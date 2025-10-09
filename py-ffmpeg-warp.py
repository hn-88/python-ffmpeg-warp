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

class VideoWarpGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Warping Tool")
        self.root.geometry("700x600")
        
        self.warp_file = tk.StringVar()
        self.input_video = tk.StringVar()
        self.output_video = tk.StringVar()
        self.output_resolution = tk.StringVar(value="3840x2160")
        
        self.video_width = 0
        self.video_height = 0
        self.is_square = False

        # codec selection
        self.output_codec = tk.StringVar(value="libx264")
        
        # Store available codecs
        self.available_codecs = self.get_available_codecs()
        
        self.create_widgets()

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = tk.Label(root, textvariable=self.status_var)
        self.status_label.grid(row=10, column=0, columnspan=3, sticky='ew', padx=5, pady=5)
        self.ffmpeg_process = None
        self.cancelling = False

        # Intercept the close button
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        
    
    def get_available_codecs(self):
        """Query ffmpeg for available video encoders and return a curated list"""
        try:
            # Run ffmpeg -encoders command
            result = subprocess.run(
                ['ffmpeg', '-encoders', '-hide_banner'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            # Parse the output to find video encoders
            lines = result.stdout.split('\n')
            video_encoders = []
            
            # Look for lines that start with 'V' (video encoders)
            encoder_pattern = re.compile(r'^\s*V\S*\s+(\S+)')
            
            for line in lines:
                match = encoder_pattern.match(line)
                if match:
                    codec_name = match.group(1)
                    video_encoders.append(codec_name)
            
            # Define a curated list of commonly used codecs
            preferred_codecs = [
                'ffvhuff',      # Huffyuv FFmpeg variant
                'libx264',      # H.264 (most compatible)
                'libx265',      # H.265/HEVC (better compression)
                'h264_nvenc',   # NVIDIA H.264 hardware encoding
                'hevc_nvenc',   # NVIDIA H.265 hardware encoding
                'h264_qsv',     # Intel QuickSync H.264
                'hevc_qsv',     # Intel QuickSync H.265
                'h264_amf',     # AMD H.264 hardware encoding
                'hevc_amf',     # AMD H.265 hardware encoding
                'libvpx',       # VP8
                'libvpx-vp9',   # VP9
                'libaom-av1',   # AV1 (libaom)
                'libsvtav1',    # AV1 (SVT)
                'libxvid',      # libxvidcore MPEG-4 part 2 (codec mpeg4)
                'mpeg4',        # MPEG-4
                'prores_ks',    # ProRes
                'dnxhd',        # DNxHD
            ]
            
            # Filter to only include codecs that are actually available
            available = [codec for codec in preferred_codecs if codec in video_encoders]
            
            # If no preferred codecs found, fall back to basic set
            if not available:
                available = ['libx264', 'mpeg4']
            
            return available
            
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            # Fallback to common codecs if ffmpeg query fails
            print(f"Warning: Could not query ffmpeg codecs: {e}")
            return ['libx264', 'libx265', 'mpeg4']
    
    def get_codec_display_names(self):
        """Return user-friendly names for codecs"""
        codec_names = {
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
        
        # Create display list with friendly names
        display_list = []
        for codec in self.available_codecs:
            if codec in codec_names:
                display_list.append(f"{codec_names[codec]}")
            else:
                display_list.append(codec)
        
        return display_list
    
    def get_codec_from_display_name(self, display_name):
        """Extract the actual codec name from the display name"""
        # Extract codec name from parentheses or use the whole string
        match = re.search(r'\(([^)]+)\)', display_name)
        if match:
            codec = match.group(1).split()[0]  # Get first word in parentheses
            return codec
        
        # If no match, try to find it in our available codecs
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
            # Schedule the check again in 100 milliseconds
            self.root.after(100, self.monitor_ffmpeg_thread, thread)
        else: 
            #The thread has finished, and conversion_complete/conversion_error
            print("FFmpeg thread completed.")
        
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Warp file selection
        ttk.Label(main_frame, text="Warp File (.map):").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.warp_file, width=50).grid(row=0, column=1, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_warp_file).grid(row=0, column=2, padx=5, pady=5)
        
        # Input video selection
        ttk.Label(main_frame, text="Input Video:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.input_video, width=50).grid(row=1, column=1, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_input_video).grid(row=1, column=2, padx=5, pady=5)
        
        # Video info label
        self.info_label = ttk.Label(main_frame, text="", foreground="blue")
        self.info_label.grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # Output resolution frame (initially hidden)
        self.resolution_frame = ttk.LabelFrame(main_frame, text="Output Resolution", padding="10")
        self.resolution_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        self.resolution_frame.grid_remove()
        
        ttk.Label(self.resolution_frame, text="Select output resolution:").grid(row=0, column=0, sticky=tk.W, pady=5)
        resolution_combo = ttk.Combobox(self.resolution_frame, textvariable=self.output_resolution, 
                                        values=["3840x2160", "1920x1080"], state="readonly", width=15)
        resolution_combo.grid(row=0, column=1, padx=10, pady=5)
        resolution_combo.current(0)

         # ===== Codec selection frame =====
        self.codec_frame = ttk.LabelFrame(main_frame, text="Output Codec", padding="10")
        self.codec_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(self.codec_frame, text="Select video codec:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        # Create combobox with friendly names
        codec_display_names = self.get_codec_display_names()
        self.codec_combo = ttk.Combobox(self.codec_frame, 
                                        values=codec_display_names,
                                        state="readonly", width=40)
        self.codec_combo.grid(row=0, column=1, padx=10, pady=5, sticky=(tk.W, tk.E))
        self.codec_combo.current(0)  # Select first codec by default
        
        # Bind selection event
        self.codec_combo.bind('<<ComboboxSelected>>', self.on_codec_selected)

        # =====
        
        # Output video
        ttk.Label(main_frame, text="Output Video:").grid(row=5, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_video, width=50).grid(row=5, column=1, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_output_video).grid(row=5, column=2, padx=5, pady=5)
        
        # Process button
        self.process_button = ttk.Button(main_frame, text="Start Processing", command=self.start_processing, state=tk.DISABLED)
        self.process_button.grid(row=6, column=0, columnspan=3, pady=20)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate', length=600)
        self.progress.grid(row=7, column=0, columnspan=3, pady=10)
        
        # Log output
        ttk.Label(main_frame, text="Log Output:").grid(row=8, column=0, columnspan=3, sticky=tk.W)
        self.log_text = scrolledtext.ScrolledText(main_frame, height=15, width=80)
        self.log_text.grid(row=9, column=0, columnspan=3, pady=5)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        self.codec_frame.columnconfigure(1, weight=1)  # Make codec combo expandable
    
    def on_codec_selected(self, event=None):
        """Handle codec selection change"""
        display_name = self.codec_combo.get()
        codec = self.get_codec_from_display_name(display_name)
        self.output_codec.set(codec)
        self.log_message(f"Selected codec: {codec}")
        
    def browse_warp_file(self):
        filename = filedialog.askopenfilename(
            title="Select Warp File",
            filetypes=[("Map files", "*.map"), ("All files", "*.*")]
        )
        if filename:
            self.warp_file.set(filename)
            self.check_ready()
            
    def browse_input_video(self):
        filename = filedialog.askopenfilename(
            title="Select Input Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if filename:
            self.input_video.set(filename)
            self.check_video_resolution(filename)
            self.check_ready()
            
    def browse_output_video(self):
        filename = filedialog.asksaveasfilename(
            title="Save Output Video As",
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        if filename:
            self.output_video.set(filename)
            self.check_ready()
            
    def check_video_resolution(self, video_path):
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "json",
                video_path
            ]
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

    # --- Save as ASCII PGM (P2) with maxval=65535 ---
    def save_pgm_p2(self, path, arr):
        h, w = arr.shape
        with open(path, "w") as f:
            f.write(f"P2\n{w} {h}\n65535\n")
            for row in arr:
                f.write(" ".join(map(str, row.tolist())) + "\n")
        
    def generate_maps(self, warp_file, input_w, input_h, out_w, out_h):
        """Generate maps from warp file based on OCVWarp generate_masks.py"""
        self.log("Generating maps...")
        
        try:
            # Read warp file
            with open(warp_file, 'rb') as f:
                lines = f.readlines()
            
            nx, ny = map(int, lines[1].split())
            data = np.array([[float(x) for x in l.split()] for l in lines[2:]])
            grid = data.reshape(ny, nx, 5)  # 5 columns: x, y, u, v, weight
            
            # --- Extract normalized u,v ---
            u = grid[::-1, :, 2]  # vertical flip
            v = 1 - grid[::-1, :, 3]  # flip for ffmpeg remap
            
            # --- Extract weight (fifth column) ---
            weight = grid[::-1, :, 4]
            
            # --- Interpolate u,v to desired resolution ---
            scale_x = out_w / nx
            scale_y = out_h / ny
            u_hr = zoom(u, (scale_y, scale_x), order=1)
            v_hr = zoom(v, (scale_y, scale_x), order=1)
            weight_hr = zoom(weight, (scale_y, scale_x), order=1)
            
            # --- Convert normalized -> integer source pixel coordinates ---
            map_x = np.round(u_hr * (input_w - 1)).astype(np.uint16)
            map_y = np.round(v_hr * (input_h - 1)).astype(np.uint16)
            
            self.save_pgm_p2("map_x_directp2.pgm", map_x)
            self.save_pgm_p2("map_y_directp2.pgm", map_y)
            
            # --- Save weight as greyscale PNG (0..255) ---
            weight_img = (np.clip(weight_hr, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(weight_img, mode='L').save("weight_alpha_mask.png")
            
            self.log("Maps generated successfully")
            return True
            
        except Exception as e:
            self.log(f"Error generating maps: {str(e)}")
            messagebox.showerror("Error", f"Failed to generate maps: {str(e)}")
            return False
            
    def start_ffmpeg_conversion(self, input_video, output_video, out_w, out_h):
        """Starts the FFmpeg process in a non-blocking thread."""
        # Disable the button/UI to prevent multiple runs
        self.process_button.config(state="disabled") 
        
        # 1. Get the FFmpeg command (cmd) 
        filter_complex = (
            f"[0:v][1:v][2:v]remap[remapped];"
            f"[3:v]format=gray,scale={self.video_width}:{self.video_height},colorchannelmixer=rr=1:gg=1:bb=1[mask_rgb];"
            f"[remapped][mask_rgb]blend=all_mode=multiply[blended];"
            f"[blended]scale={out_w}:{out_h},format=yuv420p[out]"
        )
        
        cmd = [
            'ffmpeg', '-y', '-i', input_video,
            '-i', 'map_x_directp2.pgm',
            '-i', 'map_y_directp2.pgm',
            '-i', 'weight_alpha_mask.png',
            '-filter_complex', filter_complex,
            '-map', '[out]',
            '-map', '0:a',
            '-c:v', self.output_codec,
            '-preset', 'p4',
            '-cq', '23',
            '-rc', 'vbr',
            '-maxrate', '8M',
            '-bufsize', '16M',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-b:a', '128k',
            output_video
        ]
        
        
        # 2. Create and start the thread
        ffmpeg_thread = threading.Thread(target=self.run_ffmpeg_process, args=(cmd,))
        ffmpeg_thread.daemon = True
        ffmpeg_thread.start()
        
        # 3. Start a monitor function to check if the thread is finished
        self.root.after(100, self.monitor_ffmpeg_thread, ffmpeg_thread)

    def conversion_complete(self, isSuccess):
        """
        This executes when ffmpeg conversion stops.
        """
        if isSuccess == True:
            self.log(f"ffmpeg conversion complete.")
        else:
            # self.log(f"Error during processing: {str(e)}")
            self.log(f"Error during processing!")

    def run_ffmpeg_process(self, cmd):
        """
        Executes the FFmpeg process and logs output.
        This method MUST be called from a separate thread.
        """
        process = None
        try:
            # Start the process
            self.ffmpeg_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            # Read the output line by line in the background thread
            for line in self.ffmpeg_process.stdout:
                # Use root.after() to safely pass the data to the main thread for logging.
                self.root.after(0, self.log, line.strip())
                
            # Wait for the process to finish
            self.ffmpeg_process.wait()
            
            # Handle success/failure
            if self.ffmpeg_process.returncode == 0:
                self.root.after(0, lambda: self.conversion_complete(True))
            else:
                # self.root.after(0, lambda: self.conversion_complete(False, self.ffmpeg_process.returncode))
                self.root.after(0, lambda: self.conversion_complete(False))
                
        except FileNotFoundError:
            self.root.after(0, lambda: self.conversion_error("Error: ffmpeg not found. Please install ffmpeg and ensure it's in PATH.", "ffmpeg not found."))
        except Exception as e:
            self.root.after(0, lambda: self.conversion_error(f"Error running ffmpeg: {str(e)}", f"Failed to run ffmpeg: {str(e)}"))

                
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
                
            # Run ffmpeg
            self.start_ffmpeg_conversion(
                self.input_video.get(),
                self.output_video.get(),
                out_w,
                out_h
            )
            
        except Exception as e:
            self.log(f"Error during processing: {str(e)}")
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
        finally:
            self.progress.stop()
            self.process_button.config(state=tk.NORMAL)
            
    def start_processing(self):
        """Start processing in a separate thread"""
        thread = threading.Thread(target=self.process_video)
        thread.daemon = True
        thread.start()
        
    # -------------------------------
    # Graceful shutdown with cancel message
    # -------------------------------
    def on_close(self):
        """Handle the window close event."""
        if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
            if not self.cancelling:
                self.cancelling = True
                self.status_label.config(text="Cancelling FFmpeg, please wait...")
                self.process_button.config(state="disabled")

                # Terminate FFmpeg in background
                self.root.after(100, self.terminate_ffmpeg_and_exit)
                return

        # If FFmpeg not running, just close immediately
        self.root.destroy()

    def terminate_ffmpeg_and_exit(self):
        """Terminate FFmpeg gracefully, then close window."""
        if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
            try:
                if platform.system() == "Windows":
                    # Graceful-ish stop
                    self.ffmpeg_process.terminate()
                else:
                    # Proper Ctrl-C behavior
                    self.ffmpeg_process.send_signal(signal.SIGINT)
                self.root.after(3000, self.force_kill_if_still_running)
                return
            except Exception as e:
                print(f"Error terminating FFmpeg: {e}")

        self.root.destroy()

    def force_kill_if_still_running(self):
        """Force kill if FFmpeg didn’t exit after 3 seconds."""
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
