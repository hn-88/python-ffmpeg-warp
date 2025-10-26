import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import subprocess
import os
import numpy as np
import threading
import json
from scipy.ndimage import zoom
from PIL import Image
import signal
import platform
import re
import shlex
import moderngl
import sys
import queue # <--- IMPORT THE QUEUE MODULE

class GPURemapper:
    """GPU-accelerated remapping using ModernGL"""
    def __init__(self, map_x_path, map_y_path, video_width, video_height):
        """
        Initialize GPU remapper with map files.
        
        Args:
            map_x_path: Path to map_x.pgm file
            map_y_path: Path to map_y.pgm file
            video_width: Input video width
            video_height: Input video height
        """
        # Create OpenGL context
        self.ctx = moderngl.create_standalone_context()
        
        self.video_width = video_width
        self.video_height = video_height
        
        # Load remap textures from PGM files
        self.map_x_texture = self._load_pgm_as_texture(map_x_path)
        self.map_y_texture = self._load_pgm_as_texture(map_y_path)
        
        # Create shader program
        self.program = self._create_shader_program()
        
        # Create fullscreen quad
        vertices = np.array([
            -1.0, -1.0,  0.0, 0.0,
             1.0, -1.0,  1.0, 0.0,
            -1.0,  1.0,  0.0, 1.0,
             1.0,  1.0,  1.0, 1.0,
        ], dtype='f4')
        
        self.vbo = self.ctx.buffer(vertices)
        self.vao = self.ctx.simple_vertex_array(
            self.program,
            self.vbo,
            'in_vert', 'in_texcoord'
        )
        
        # Input video texture (will be updated per frame)
        self.video_texture = self.ctx.texture((video_width, video_height), 3)
        self.video_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        
        # Framebuffer will be created when we know output size
        self.fbo_texture = None
        self.fbo = None
        
    def set_output_size(self, out_width, out_height):
        """Set output resolution and create framebuffer"""
        if self.fbo_texture:
            self.fbo_texture.release()
        if self.fbo:
            self.fbo.release()
            
        self.fbo_texture = self.ctx.texture((out_width, out_height), 3)
        self.fbo = self.ctx.framebuffer(color_attachments=[self.fbo_texture])
        
    def _load_pgm_as_texture(self, pgm_path):
        """Load PGM file and create OpenGL texture."""
        with open(pgm_path, 'rb') as f:
            # Read PGM header
            magic = f.readline().strip()
            if magic not in (b'P2', b'P5'):
                raise ValueError(f"Unsupported PGM format: {magic}")
            
            # Skip comments
            line = f.readline()
            while line.startswith(b'#'):
                line = f.readline()
            
            # Read dimensions
            width, height = map(int, line.split())
            max_val = int(f.readline().strip())
            
            # Read pixel data
            if magic == b'P5':  # Binary
                data = np.frombuffer(f.read(), dtype=np.uint8 if max_val < 256 else np.uint16)
            else:  # ASCII (P2)
                data = np.fromstring(f.read(), sep=' ', dtype=np.uint16)
            
            data = data.reshape((height, width))
            data = data.astype(np.float32)
        
        # Create texture (single channel, float32)
        texture = self.ctx.texture((width, height), 1, dtype='f4')
        texture.write(data.tobytes())
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        
        return texture
    
    def _create_shader_program(self):
        """Create shader program for remapping."""
        vertex_shader = '''
        #version 330
        
        in vec2 in_vert;
        in vec2 in_texcoord;
        out vec2 v_texcoord;
        
        void main() {
            gl_Position = vec4(in_vert, 0.0, 1.0);
            v_texcoord = in_texcoord;
        }
        '''
        
        fragment_shader = '''
        #version 330
        
        uniform sampler2D video_texture;
        uniform sampler2D map_x;
        uniform sampler2D map_y;
        uniform sampler2D mask_texture;
        uniform vec2 video_size;
        uniform int use_mask;
        
        in vec2 v_texcoord;
        out vec4 fragColor;
        
        void main() {
            // Read map values (these are pixel coordinates)
            float x_coord = texture(map_x, v_texcoord).r;
            float y_coord = texture(map_y, v_texcoord).r;
            
            // Normalize to 0-1 range (convert from pixel coords to texture coords)
            vec2 remap_coord = vec2(x_coord, y_coord) / video_size;
            
            // Sample from video texture at remapped coordinates
            remap_coord = clamp(remap_coord, 0.0, 1.0);
            vec4 color = texture(video_texture, remap_coord);
            
            // Apply mask if enabled
            if (use_mask == 1) {
                float mask_value = texture(mask_texture, v_texcoord).r;
                color.rgb *= mask_value;
            }
            
            fragColor = color;
        }
        '''
        
        return self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )
    
    def set_mask_texture(self, mask_path):
        """Load and set mask texture"""
        mask_img = Image.open(mask_path).convert('L')
        mask_data = np.array(mask_img, dtype=np.uint8)
        
        self.mask_texture = self.ctx.texture(mask_data.shape[::-1], 1)
        self.mask_texture.write(mask_data.tobytes())
        self.mask_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
    
    def remap_frame(self, frame_rgb, use_mask=False):
        """
        Remap a single frame.
        
        Args:
            frame_rgb: numpy array of shape (height, width, 3), dtype=uint8
            use_mask: Whether to apply mask
            
        Returns:
            Remapped frame as numpy array
        """
        # Upload frame to GPU
        self.video_texture.write(frame_rgb.tobytes())
        
        # Bind textures
        self.video_texture.use(location=0)
        self.map_x_texture.use(location=1)
        self.map_y_texture.use(location=2)
        if use_mask and hasattr(self, 'mask_texture'):
            self.mask_texture.use(location=3)
        
        # Set uniforms
        self.program['video_texture'].value = 0
        self.program['map_x'].value = 1
        self.program['map_y'].value = 2
        self.program['mask_texture'].value = 3
        self.program['video_size'].value = (self.video_width, self.video_height)
        self.program['use_mask'].value = 1 if use_mask else 0
        
        # Render to framebuffer
        self.fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0)
        self.vao.render(moderngl.TRIANGLE_STRIP)
        
        # Read result
        result = np.frombuffer(self.fbo_texture.read(), dtype=np.uint8)
        result = result.reshape((self.fbo_texture.height, self.fbo_texture.width, 3))
        
        # Flip vertically (OpenGL has origin at bottom-left)
        result = np.flipud(result)
        
        return result


class VideoWarpGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Warping Tool (GPU Accelerated)")
        self.root.geometry("700x700")
        
        self.warp_file = tk.StringVar()
        self.input_video = tk.StringVar()
        self.output_video = tk.StringVar()
        self.output_resolution = tk.StringVar(value="3840x2160")
        
        self.video_width = 0
        self.video_height = 0
        self.is_square = False

        self.codec_names = {
            'ffvhuff': 'Huffyuv (lossless)',
            'libx264': 'H.264 (libx264) - Best compatibility',
            'libx265': 'H.265/HEVC (libx265) - Better compression',
            'h264_nvenc': 'H.264 (NVIDIA GPU)',
            'hevc_nvenc': 'H.265 (NVIDIA GPU)',
            'h264_qsv': 'H.264 (Intel QuickSync)',
            'hevc_qsv': 'H.265 (Intel QuickSync)',
            'h264_amf': 'H.264 (AMD GPU)',
            'hevc_amf': 'H.265 (AMD GPU)',
            'libvpx': 'VP8',
            'libvpx-vp9': 'VP9',
            'libaom-av1': 'AV1 (libaom)',
            'libsvtav1': 'AV1 (SVT)',
            'mpeg4': 'MPEG-4',
            'prores_ks': 'ProRes',
            'dnxhd': 'DNxHD',
            'libxvid': 'XVID',
        }

        self.codec_params = {
            'libx264': ['-preset', 'fast', '-crf', '23', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k'],
            'libx265': ['-preset', 'medium', '-crf', '28', '-pix_fmt', 'yuv420p', '-maxrate', '8M', '-c:a', 'aac', '-b:a', '128k'],
            'hevc_nvenc': ['-preset', 'medium', '-cq', '23', '-rc', 'vbr', '-maxrate', '8M', '-bufsize', '16M', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k'],
            'h264_nvenc': ['-preset', 'medium', '-cq', '23', '-rc', 'vbr', '-maxrate', '8M', '-bufsize', '16M', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k'],
            'ffvhuff': ['-c:v', 'ffvhuff', '-pix_fmt', 'yuv420p'],
            'mpeg4': ['-q:v', '5', '-c:a', 'aac', '-b:a', '128k'],
        }

        self.output_codec = tk.StringVar(value="libx264")
        self.available_codecs = self.get_available_codecs()
        self.crop_to_4k = tk.BooleanVar(value=False)
        
        self.create_widgets()

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = tk.Label(root, textvariable=self.status_var)
        self.status_label.grid(row=10, column=0, columnspan=3, sticky='ew', padx=5, pady=5)
        self.ffmpeg_process = None
        self.cancelling = False
        self.gpu_remapper = None

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
    def get_ffmpeg_params(self, codec):
        """Return list of ffmpeg parameters for the selected codec."""
        return self.codec_params.get(codec, ['-preset', 'fast', '-crf', '23', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k'])

    def get_available_codecs(self):
        """Query ffmpeg for available video encoders and return a curated list"""
        try:
            result = subprocess.run(['ffmpeg', '-encoders', '-hide_banner'], capture_output=True, text=True, timeout=5)
            lines = result.stdout.split('\n')
            video_encoders = []
            encoder_pattern = re.compile(r'^\s*V\S*\s+(\S+)')
            
            for line in lines:
                match = encoder_pattern.match(line)
                if match:
                    video_encoders.append(match.group(1))
            
            preferred_codecs = ['ffvhuff', 'libx264', 'libx265', 'h264_nvenc', 'hevc_nvenc', 'h264_qsv', 'hevc_qsv', 
                              'h264_amf', 'hevc_amf', 'libvpx', 'libvpx-vp9', 'libaom-av1', 'libsvtav1', 'libxvid', 'mpeg4', 'prores_ks', 'dnxhd']
            available = [codec for codec in preferred_codecs if codec in video_encoders]
            
            if not available:
                available = ['libx264', 'mpeg4']
            
            return available
            
        except Exception as e:
            print(f"Warning: Could not query ffmpeg codecs: {e}")
            return ['libx264', 'libx265', 'mpeg4']
    
    def get_codec_display_names(self):
        """Return user-friendly names for codecs"""        
        display_list = []
        for codec in self.available_codecs:
            if codec in self.codec_names:
                display_list.append(f"{self.codec_names[codec]}")
            else:
                display_list.append(codec)
        return display_list
    
    def get_codec_from_display_name(self, display_name):
        """Return codec key for a given user-friendly display name."""
        for codec, name in self.codec_names.items():
            if name == display_name:
                return codec
        for codec in self.available_codecs:
            if codec in display_name:
                return codec
        return self.available_codecs[0] if self.available_codecs else 'libx264'

    def update_status(self, message):
        """Safely updates the dedicated status label in the main thread."""
        self.status_var.set(f"Processing: {message}")

    def monitor_ffmpeg_thread(self, thread):
        """Checks if the thread is alive and re-runs itself."""
        if thread.is_alive():
            self.root.after(100, self.monitor_ffmpeg_thread, thread)
        else: 
            print("Processing thread completed.")
        
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        ttk.Label(main_frame, text="Warp File (.map):").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.warp_file, width=50).grid(row=0, column=1, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_warp_file).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(main_frame, text="Input Video:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.input_video, width=50).grid(row=1, column=1, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_input_video).grid(row=1, column=2, padx=5, pady=5)
        ttk.Checkbutton(main_frame, text="Crop to 4K", variable=self.crop_to_4k).grid(row=1, column=3, sticky=tk.W, padx=10, pady=5)
        
        self.info_label = ttk.Label(main_frame, text="", foreground="blue")
        self.info_label.grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        self.resolution_frame = ttk.LabelFrame(main_frame, text="Output Resolution", padding="10")
        self.resolution_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        self.resolution_frame.grid_remove()
        
        ttk.Label(self.resolution_frame, text="Select output resolution:").grid(row=0, column=0, sticky=tk.W, pady=5)
        resolution_combo = ttk.Combobox(self.resolution_frame, textvariable=self.output_resolution, 
                                        values=["3840x2160", "1920x1080"], state="readonly", width=15)
        resolution_combo.grid(row=0, column=1, padx=10, pady=5)
        resolution_combo.current(0)

        self.codec_frame = ttk.LabelFrame(main_frame, text="Output Codec", padding="10")
        self.codec_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(self.codec_frame, text="Select video codec:").grid(row=0, column=0, sticky=tk.W, pady=5)
        codec_display_names = self.get_codec_display_names()
        self.codec_combo = ttk.Combobox(self.codec_frame, values=codec_display_names, state="readonly", width=40)
        self.codec_combo.grid(row=0, column=1, padx=10, pady=5, sticky=(tk.W, tk.E))
        self.codec_combo.current(0)
        self.codec_combo.bind('<<ComboboxSelected>>', self.on_codec_selected)
        
        ttk.Label(main_frame, text="Output Video:").grid(row=5, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_video, width=50).grid(row=5, column=1, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_output_video).grid(row=5, column=2, padx=5, pady=5)
        
        self.process_button = ttk.Button(main_frame, text="Start Processing", command=self.start_processing, state=tk.DISABLED)
        self.process_button.grid(row=6, column=0, columnspan=3, pady=20)
        
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate', length=600)
        self.progress.grid(row=7, column=0, columnspan=3, pady=10)
        
        ttk.Label(main_frame, text="Log Output:").grid(row=8, column=0, columnspan=3, sticky=tk.W)
        self.log_text = scrolledtext.ScrolledText(main_frame, height=15, width=80)
        self.log_text.grid(row=9, column=0, columnspan=3, pady=5)
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        self.codec_frame.columnconfigure(1, weight=1)
    
    def on_codec_selected(self, event=None):
        """Handle codec selection change"""
        display_name = self.codec_combo.get()
        codec = self.get_codec_from_display_name(display_name)
        self.output_codec.set(codec)
        self.log(f"Selected codec: {codec}")
        
    def browse_warp_file(self):
        filename = filedialog.askopenfilename(title="Select Warp File", filetypes=[("Map files", "*.map"), ("All files", "*.*")])
        if filename:
            self.warp_file.set(filename)
            self.check_ready()
            
    def browse_input_video(self):
        filename = filedialog.askopenfilename(title="Select Input Video", 
                                             filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
        if filename:
            self.input_video.set(filename)
            self.check_video_resolution(filename)
            self.check_ready()
            
    def browse_output_video(self):
        filename = filedialog.asksaveasfilename(title="Save Output Video As", defaultextension=".mp4",
                                               filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
        if filename:
            self.output_video.set(filename)
            self.check_ready()
            
    def check_video_resolution(self, video_path):
        try:
            cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", video_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            info = json.loads(result.stdout)
            self.video_width = int(info['streams'][0]['width'])
            self.video_height = int(info['streams'][0]['height'])
            
            self.is_square = (self.video_width == self.video_height)
            
            info_text = f"Video Resolution: {self.video_width}x{self.video_height}"
            if self.is_square:
                info_text += " (Square - Output resolution required)"
                self.resolution_frame.grid()
            else:
                info_text += " (Non-square)"
                self.resolution_frame.grid_remove()
                
            self.info_label.config(text=info_text)
            self.log(info_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read video resolution: {str(e)}")
            self.log(f"Error reading video: {str(e)}")
            
    def check_ready(self):
        if self.warp_file.get() and self.input_video.get() and self.output_video.get():
            self.process_button.config(state=tk.NORMAL)
        else:
            self.process_button.config(state=tk.DISABLED)
            
    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def save_pgm_p2(self, path, arr):
        h, w = arr.shape
        with open(path, "w") as f:
            f.write(f"P2\n{w} {h}\n65535\n")
            for row in arr:
                f.write(" ".join(map(str, row.tolist())) + "\n")
        
    def generate_maps(self, warp_file, input_w, input_h, out_w, out_h):
        """Generate maps from warp file"""
        self.log("Generating maps...")
        
        try:
            with open(warp_file, 'rb') as f:
                lines = f.readlines()
            
            nx, ny = map(int, lines[1].split())
            data = np.array([[float(x) for x in l.split()] for l in lines[2:]])
            grid = data.reshape(ny, nx, 5)
            
            u = grid[::-1, :, 2]
            v = 1 - grid[::-1, :, 3]
            weight = grid[::-1, :, 4]
            
            scale_x = out_w / nx
            scale_y = out_h / ny
            u_hr = zoom(u, (scale_y, scale_x), order=1)
            v_hr = zoom(v, (scale_y, scale_x), order=1)
            weight_hr = zoom(weight, (scale_y, scale_x), order=1)
            
            map_x = np.round(u_hr * (input_w - 1)).astype(np.uint16)
            map_y = np.round(v_hr * (input_h - 1)).astype(np.uint16)
            
            self.save_pgm_p2("map_x_directp2.pgm", map_x)
            self.save_pgm_p2("map_y_directp2.pgm", map_y)
            
            weight_img = (np.clip(weight_hr, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(weight_img, mode='L').save("weight_alpha_mask.png")
            
            self.log("Maps generated successfully")
            return True
            
        except Exception as e:
            self.log(f"Error generating maps: {str(e)}")
            messagebox.showerror("Error", f"Failed to generate maps: {str(e)}")
            return False

    def read_frames(self, input_proc, frame_size, frames_to_process_q):
        """Thread function to read frames from ffmpeg."""
        while True:
            frame_data = input_proc.stdout.read(frame_size)
            if len(frame_data) != frame_size:
                break
            frames_to_process_q.put(frame_data)
        frames_to_process_q.put(None)  # Sentinel to signal end of stream

    def write_frames(self, output_proc, frames_to_write_q):
        """Thread function to write frames to ffmpeg."""
        while True:
            frame = frames_to_write_q.get()
            if frame is None:
                break
            output_proc.stdin.write(frame)

    def gpu_process_video(self, input_video, output_video, out_w, out_h, framerate):
        """GPU-accelerated video processing with a multi-threaded pipeline."""
        try:
            # Get input video info
            probe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                         '-show_entries', 'stream=r_frame_rate', '-of', 'default=noprint_wrappers=1', input_video]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            for line in probe_result.stdout.strip().split('\n'):
                if line.startswith('r_frame_rate='):
                    fps_str = line.split('=')[1]
                    num, den = map(int, fps_str.split('/'))
                    detected_fps = num / den
                    framerate = framerate or detected_fps
        except:
            framerate = framerate or 30

        self.log(f"Processing at {framerate} fps with GPU acceleration")

        input_w = self.video_width
        input_h = self.video_height
        if self.crop_to_4k.get():
            input_w = 4096
            input_h = 4096

        self.gpu_remapper = GPURemapper("map_x_directp2.pgm", "map_y_directp2.pgm", input_w, input_h)
        self.gpu_remapper.set_output_size(out_w, out_h)
        self.gpu_remapper.set_mask_texture("weight_alpha_mask.png")

        if self.crop_to_4k.get():
            input_cmd = ['ffmpeg', '-i', input_video, '-vf', 'crop=4096:4096', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
        else:
            input_cmd = ['ffmpeg', '-i', input_video, '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']

        params = self.get_ffmpeg_params(self.output_codec.get())
        output_cmd = ['ffmpeg', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{out_w}x{out_h}',
                      '-r', str(framerate), '-i', 'pipe:0', '-c:v', self.output_codec.get()] + params + ['-y', output_video]

        self.log(f"Input command: {' '.join(input_cmd)}")
        self.log(f"Output command: {' '.join(output_cmd)}")

        input_proc = subprocess.Popen(input_cmd, stdout=subprocess.PIPE, stderr=sys.stderr, bufsize=10**8)
        output_proc = subprocess.Popen(output_cmd, stdin=subprocess.PIPE, stderr=sys.stderr, bufsize=10**8)

        frame_size = input_w * input_h * 3
        frame_count = 0

        # Create queues
        frames_to_process_q = queue.Queue(maxsize=30)  # maxsize helps to prevent excessive memory usage
        frames_to_write_q = queue.Queue(maxsize=30)

        # Start read and write threads
        read_thread = threading.Thread(target=self.read_frames, args=(input_proc, frame_size, frames_to_process_q))
        write_thread = threading.Thread(target=self.write_frames, args=(output_proc, frames_to_write_q))
        read_thread.start()
        write_thread.start()
        
        try:
            while True:
                frame_data = frames_to_process_q.get()
                if frame_data is None: # End of stream
                    break
                
                frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((input_h, input_w, 3))
                remapped = self.gpu_remapper.remap_frame(frame, use_mask=True)
                frames_to_write_q.put(remapped.tobytes())
                
                frame_count += 1
                if frame_count % 30 == 0:
                    self.root.after(0, self.log, f"Processed {frame_count} frames...")
                
        except Exception as e:
            self.root.after(0, self.log, f"Error processing frame {frame_count}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            frames_to_write_q.put(None) # Signal writer thread to exit
            read_thread.join()
            write_thread.join()

            try:
                input_proc.stdout.close()
                if output_proc.stdin:
                    output_proc.stdin.close()
            except:
                pass
            
            input_proc.terminate()
            output_proc.terminate()
            input_proc.wait()
            output_proc.wait()
            
        self.root.after(0, self.log, f"Finished processing {frame_count} frames")
        self.root.after(0, self.conversion_complete, True)
            
    def conversion_complete(self, isSuccess):
        """This executes when conversion stops."""
        if isSuccess:
            self.log(f"GPU conversion complete!")
        else:
            self.log(f"Error during processing!")
        self.progress.stop()
        self.process_button.config(state=tk.NORMAL)

    def process_video(self):
        """Main processing function"""
        try:
            self.progress.start()
            self.process_button.config(state=tk.DISABLED)
            
            # Determine output resolution
            if self.is_square:
                out_w, out_h = map(int, self.output_resolution.get().split('x'))
            else:
                out_w, out_h = self.video_width, self.video_height
                
            # Generate maps
            if not self.generate_maps(
                self.warp_file.get(),
                self.video_width,
                self.video_height,
                self.video_width,
                self.video_height
            ):
                return
                
            # Run GPU-accelerated processing
            self.gpu_process_video(
                self.input_video.get(),
                self.output_video.get(),
                out_w,
                out_h,
                framerate=None
            )
            
        except Exception as e:
            self.log(f"Error during processing: {str(e)}")
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
            self.progress.stop()
            self.process_button.config(state=tk.NORMAL)
            
    def start_processing(self):
        """Start processing in a separate thread"""
        thread = threading.Thread(target=self.process_video)
        thread.daemon = True
        thread.start()
        self.monitor_ffmpeg_thread(thread) # Keep the original monitor for the main processing thread
        
    def on_close(self):
        """Handle the window close event."""
        if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
            if not self.cancelling:
                self.cancelling = True
                self.status_label.config(text="Cancelling FFmpeg, please wait...")
                self.process_button.config(state="disabled")
                self.root.after(100, self.terminate_ffmpeg_and_exit)
                return
        self.root.destroy()

    def terminate_ffmpeg_and_exit(self):
        """Terminate FFmpeg gracefully, then close window."""
        if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
            try:
                if platform.system() == "Windows":
                    self.ffmpeg_process.terminate()
                else:
                    self.ffmpeg_process.send_signal(signal.SIGINT)
                self.root.after(3000, self.force_kill_if_still_running)
                return
            except Exception as e:
                print(f"Error terminating FFmpeg: {e}")
        self.root.destroy()

    def force_kill_if_still_running(self):
        """Force kill if FFmpeg didn't exit after 3 seconds."""
        if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
            print("Force killing FFmpeg...")
            try:
                self.ffmpeg_process.kill()
            except Exception as e:
                print(f"Error killing FFmpeg: {e}")
        self.root.destroy()


def main():
    root = tk.Tk()
    app = VideoWarpGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
