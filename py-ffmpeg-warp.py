import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import subprocess
import os
import numpy as np
import threading
import json
from scipy.ndimage import zoom, map_coordinates
from PIL import Image
import signal
import platform
import re
import shlex
from fractions import Fraction

class VideoWarpGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Warping Tool")
        
        # Desired window size
        desired_width = 850
        desired_height = 900

        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Calculate the actual window size
        actual_width = min(desired_width, screen_width-20)
        actual_height = min(desired_height, screen_height-20)

        # Set the geometry
        self.root.geometry(f"{actual_width}x{actual_height}")
        
        # OCVWarp Options
        self.transform_options = {
            "0: Equirectangular 360 to 360 degree fisheye": 0,
            "1: Equirectangular 360 to 180 degree fisheye": 1,
            "2: 360 degree fisheye to Equirectangular 360": 2,
            "3: 180 degree fisheye to Equirectangular (Parallel)": 3,
            "4: 180 fisheye to Warped (.map required)": 4,
            "5: Equirect 360 to 180 fisheye to Warped (.map required)": 5
        }
        
        self.transform_type = tk.StringVar(value="4: 180 fisheye to Warped (.map required)")
        self.angle_x = tk.DoubleVar(value=0.0)
        self.angle_y = tk.DoubleVar(value=0.0)

        self.warp_file = tk.StringVar()
        self.input_video = tk.StringVar()
        self.output_video = tk.StringVar()
        self.output_resolution = tk.StringVar(value="3840x2160")
        
        self.video_width = 0
        self.video_height = 0
        self.is_square = False
        self.total_frames = 100000

        self.codec_names = {
            'ffvhuff': 'Huffyuv (lossless)',
            'libx264': 'H.264 (libx264) - Best compatibility',
            'libx265': 'H.265/HEVC (libx265) - Better compression',
            'h264_nvenc': 'H.264 (NVIDIA GPU)',
            'hevc_nvenc': 'H.265 (NVIDIA GPU)',
            'h264_videotoolbox': 'H.264 (APPLE GPU)',
            'hevc_videotoolbox': 'H.265 (APPLE GPU)',
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
            'hevc_nvenc': ['-preset', 'p4', '-cq', '23', '-rc', 'vbr', '-maxrate', '8M', '-bufsize', '16M', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k'],
            'hevc_videotoolbox': ['-quality', 'balanced', '-b:v', '5M', '-tag:v', 'hvc1', '-movflags', '+frag_keyframe+empty_moov', '-maxrate', '8M', '-bufsize', '16M', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k'],
            'h264_videotoolbox': ['-quality', 'balanced', '-b:v', '5M', '-tag:v', 'avc1', '-movflags', '+frag_keyframe+empty_moov+faststart', '-maxrate', '8M', '-bufsize', '16M', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k'],
            'h264_nvenc': ['-preset', 'p4', '-cq', '23', '-rc', 'vbr', '-maxrate', '8M', '-bufsize', '16M', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k'],
            'ffvhuff': ['-c:v', 'ffvhuff', '-pix_fmt', 'yuv420p'],
            'mpeg4': ['-q:v', '5', '-c:a', 'aac', '-b:a', '128k'],
        }

        # codec selection
        self.output_codec = tk.StringVar(value="libx264")
        self.available_codecs = self.get_available_codecs()

        # Crop variables
        self.crop_to_4k = tk.BooleanVar(value=False)
        self.crop_to_1k = tk.BooleanVar(value=False)
        
        self.create_widgets()

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = tk.Label(root, textvariable=self.status_var)
        self.status_label.grid(row=10, column=0, columnspan=3, sticky='ew', padx=5, pady=5)
        self.ffmpeg_process = None
        self.cancelling = False

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
    def get_current_transform_type(self):
        selection = self.transform_type.get()
        return self.transform_options.get(selection, 4)

    def on_transform_change(self, event=None):
        tt = self.get_current_transform_type()
        if tt in [4, 5]:
            self.warp_entry.config(state=tk.NORMAL)
            self.warp_btn.config(state=tk.NORMAL)
        else:
            self.warp_entry.config(state=tk.DISABLED)
            self.warp_btn.config(state=tk.DISABLED)
        self.check_ready()
        
    def get_ffmpeg_params(self, codec):
        return self.codec_params.get(codec, [
            '-preset', 'fast', '-crf', '23', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k'
        ])

    def get_available_codecs(self):
        try:
            result = subprocess.run(['ffmpeg', '-encoders', '-hide_banner'], capture_output=True, text=True, timeout=5)
            lines = result.stdout.split('\n')
            video_encoders = []
            encoder_pattern = re.compile(r'^\s*V\S*\s+(\S+)')
            
            for line in lines:
                match = encoder_pattern.match(line)
                if match:
                    codec_name = match.group(1)
                    video_encoders.append(codec_name)
            
            preferred_codecs = [
                'ffvhuff', 'libx264', 'libx265', 'h264_nvenc', 'hevc_nvenc', 'h264_videotoolbox', 
                'hevc_videotoolbox', 'h264_qsv', 'hevc_qsv', 'h264_amf', 'hevc_amf', 'libvpx', 
                'libvpx-vp9', 'libaom-av1', 'libsvtav1', 'libxvid', 'mpeg4', 'prores_ks', 'dnxhd'
            ]
            
            available = [codec for codec in preferred_codecs if codec in video_encoders]
            if not available:
                available = ['libx264', 'mpeg4']
            return available
            
        except Exception as e:
            print(f"Warning: Could not query ffmpeg codecs: {e}")
            return ['libx264', 'libx265', 'mpeg4']
    
    def get_codec_display_names(self):
        display_list = []
        for codec in self.available_codecs:
            if codec in self.codec_names:
                display_list.append(f"{self.codec_names[codec]}")
            else:
                display_list.append(codec)
        return display_list
    
    def get_codec_from_display_name(self, display_name):
        for codec, name in self.codec_names.items():
            if name == display_name:
                return codec
        for codec in self.available_codecs:
            if codec in display_name:
                return codec
        return self.available_codecs[0] if self.available_codecs else 'libx264'

    def update_status(self, message):
        self.status_var.set(f"Processing: {message}")

    def monitor_ffmpeg_thread(self, thread):
        if thread.is_alive():
            self.root.after(100, self.monitor_ffmpeg_thread, thread)
        else: 
            print("FFmpeg thread completed.")
        
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Transform Settings Frame
        self.transform_frame = ttk.LabelFrame(main_frame, text="Transform Settings", padding="10")
        self.transform_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(self.transform_frame, text="Type:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.transform_combo = ttk.Combobox(self.transform_frame, textvariable=self.transform_type, values=list(self.transform_options.keys()), state="readonly", width=55)
        self.transform_combo.grid(row=0, column=1, columnspan=2, padx=5, pady=2, sticky=tk.W)
        self.transform_combo.bind('<<ComboboxSelected>>', self.on_transform_change)

        ttk.Label(self.transform_frame, text="Angle X (deg):").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.transform_frame, textvariable=self.angle_x, width=15).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(self.transform_frame, text="Angle Y (deg):").grid(row=1, column=2, sticky=tk.W, pady=2, padx=(10, 0))
        ttk.Entry(self.transform_frame, textvariable=self.angle_y, width=15).grid(row=1, column=3, sticky=tk.W, padx=5, pady=2)

        ttk.Label(self.transform_frame, text="Warp File (.map):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.warp_entry = ttk.Entry(self.transform_frame, textvariable=self.warp_file, width=45)
        self.warp_entry.grid(row=2, column=1, columnspan=2, pady=5)
        self.warp_btn = ttk.Button(self.transform_frame, text="Browse", command=self.browse_warp_file)
        self.warp_btn.grid(row=2, column=3, padx=5, pady=5)

        # Input video selection
        ttk.Label(main_frame, text="Input Video:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.input_video, width=50).grid(row=1, column=1, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_input_video).grid(row=1, column=2, padx=5, pady=5)
        
        # Video info label
        self.info_label = ttk.Label(main_frame, text="", foreground="blue")
        self.info_label.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Output resolution frame 
        self.resolution_frame = ttk.LabelFrame(main_frame, text="Output Resolution", padding="10")
        self.resolution_frame.grid(row=3, column=2, sticky=(tk.W, tk.E), pady=10)
        ttk.Checkbutton(main_frame, text="Crop input to 4K", variable=self.crop_to_4k).grid(row=3, column=0, sticky=tk.W, padx=10, pady=5)
        ttk.Checkbutton(main_frame, text="Crop input to 1K", variable=self.crop_to_1k).grid(row=3, column=1, sticky=tk.W, padx=10, pady=5)
        
        ttk.Label(self.resolution_frame, text="Select output resolution:").grid(row=0, column=0, sticky=tk.W, pady=5)
        resolution_combo = ttk.Combobox(self.resolution_frame, textvariable=self.output_resolution, 
                                        values=["3840x2160", "1920x1080"], state="readonly", width=15)
        resolution_combo.grid(row=0, column=1, padx=10, pady=5)
        resolution_combo.current(0)

         # ===== Codec selection frame =====
        self.codec_frame = ttk.LabelFrame(main_frame, text="Output Codec", padding="10")
        self.codec_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(self.codec_frame, text="Select video codec:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        codec_display_names = self.get_codec_display_names()
        self.codec_combo = ttk.Combobox(self.codec_frame, values=codec_display_names, state="readonly", width=40)
        self.codec_combo.grid(row=0, column=1, padx=10, pady=5, sticky=(tk.W, tk.E))
        try:
            self.codec_combo.current(4)
        except Exception:
            try:
                self.codec_combo.current(0)
            except Exception:
                pass
        self.codec_combo.bind('<<ComboboxSelected>>', self.on_codec_selected)

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
        self.log_text = scrolledtext.ScrolledText(main_frame, height=12, width=80)
        self.log_text.grid(row=9, column=0, columnspan=3, pady=5)

        try:
            self.codec_combo.event_generate('<<ComboboxSelected>>')
        except Exception:
            pass
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        self.codec_frame.columnconfigure(1, weight=1)
    
    def on_codec_selected(self, event=None):
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
        filename = filedialog.askopenfilename(title="Select Input Video", filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
        if filename:
            self.input_video.set(filename)
            self.check_video_resolution(filename)
            count = self.get_frame_count(filename)
            if (count > 0):
                self.total_frames = count
            self.check_ready()
            
    def browse_output_video(self):
        filename = filedialog.asksaveasfilename(title="Save Output Video As", defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
        if filename:
            self.output_video.set(filename)
            self.check_ready()

    def get_frame_count(self, video_path):
        try:
            cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_format", "-of", "json", video_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0 or not result.stdout:
                return 0
            info = json.loads(result.stdout)
            streams = info.get("streams", [])
            if not streams:
                return 0
            stream = streams[0]
            nb_frames = stream.get("nb_frames")
            if nb_frames and nb_frames.lower() != "n/a":
                try:
                    return int(nb_frames)
                except ValueError:
                    pass
            duration = None
            if stream.get("duration") and stream["duration"] != "N/A":
                 duration = float(stream["duration"])
            elif info.get("format", {}).get("duration") and info["format"]["duration"] != "N/A":
                 duration = float(info["format"]["duration"])
            elif stream.get("tags", {}).get("DURATION"):
                try:
                    h, m, s_str = stream["tags"]["DURATION"].split(":")
                    duration = int(h)*3600 + int(m)*60 + float(s_str)
                except Exception:
                    pass
            if not duration or duration <= 0:
                return 0
            fps_str = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or ""
            if not fps_str or fps_str in ("N/A", "nan"):
                return 0
            try:
                fps = float(Fraction(fps_str))
            except Exception:
                return 0
            if not (fps > 0):
                return 0
            total_frames = int(round(duration * fps))
            return total_frames if total_frames > 0 else 0
        except Exception as e:
            print(f"Error getting frame count: {e}")
            return 0
            
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
                info_text += " (Square)"
            else:
                info_text += " (Non-square)"
                
            self.info_label.config(text=info_text)
            self.log(info_text)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read video resolution: {str(e)}")
            self.log(f"Error reading video: {str(e)}")
            
    def check_ready(self):
        tt = self.get_current_transform_type()
        needs_map = tt in [4, 5]
        
        has_in = bool(self.input_video.get())
        has_out = bool(self.output_video.get())
        has_map = bool(self.warp_file.get())
        
        if has_in and has_out and (not needs_map or has_map):
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
                
    def get_math_maps(self, transformtype, in_w, in_h, out_w, out_h, anglex, angley):
        """Generates mathematical numpy maps mimicking OCVWarp.cpp logic"""
        j, i = np.meshgrid(np.arange(out_w), np.arange(out_h))
        
        anglexrad = -np.radians(anglex)
        angleyrad = -np.radians(angley)

        xcd_out = out_w / 2.0 - 1.0
        ycd_out = out_h / 2.0 - 1.0
        xcd_in = in_w / 2.0 - 1.0
        ycd_in = in_h / 2.0 - 1.0

        mask = np.ones((out_h, out_w), dtype=np.float32)

        with np.errstate(divide='ignore', invalid='ignore'):
            if transformtype in [0, 1]:
                aperture = 2 * np.pi if transformtype == 0 else np.pi
                halfcols = out_w / 2.0
                halfrows = out_h / 2.0
                
                xfish = (j - xcd_out) / halfcols
                yfish = (i - ycd_out) / halfrows
                rfish = np.sqrt(xfish**2 + yfish**2)
                theta = np.arctan2(yfish, xfish)
                phi = rfish * aperture / 2.0
                
                Px = np.sin(phi) * np.cos(theta)
                Py = np.sin(phi) * np.sin(theta)
                Pz = np.cos(phi)
                
                if anglex != 0 or angley != 0:
                    PxR = Px
                    PyR = np.cos(angleyrad) * Py - np.sin(angleyrad) * Pz
                    PzR = np.sin(angleyrad) * Py + np.cos(angleyrad) * Pz
                    
                    Px = np.cos(anglexrad) * PxR - np.sin(anglexrad) * PyR
                    Py = np.sin(anglexrad) * PxR + np.cos(anglexrad) * PyR
                    Pz = PzR
                    
                longi = np.arctan2(Py, Px)
                lat = np.arctan2(Pz, np.sqrt(Px**2 + Py**2))
                
                xequi = longi / np.pi
                yequi = 2 * lat / np.pi
                
                if transformtype == 0:
                    mask[rfish > 1.1] = 0.0
                    
                map_x = np.abs(xequi * (in_w / 2.0) + xcd_in)
                map_y = yequi * (in_h / 2.0) + ycd_in

            elif transformtype in [2, 3]:
                aperture = 2 * np.pi if transformtype == 2 else np.pi
                
                longi = np.pi * (j - xcd_out) / (out_w / 2.0)
                lat = (np.pi / 2.0) * (i - ycd_out) / (out_h / 2.0)
                
                Px = np.cos(lat) * np.cos(longi)
                Py = np.cos(lat) * np.sin(longi)
                Pz = np.sin(lat)
                
                if anglex != 0 or angley != 0:
                    PxR = Px
                    PyR = np.cos(angleyrad) * Py - np.sin(angleyrad) * Pz
                    PzR = np.sin(angleyrad) * Py + np.cos(angleyrad) * Pz
                    
                    Px = np.cos(anglexrad) * PxR - np.sin(anglexrad) * PyR
                    Py = np.sin(anglexrad) * PxR + np.cos(anglexrad) * PyR
                    Pz = PzR
                    
                if transformtype == 2:
                    R = np.where((Px==0) & (Py==0) & (Pz==0), 0.0, 2 * np.arctan2(np.sqrt(Px**2 + Py**2), Pz) / aperture)
                    theta = np.where((Px==0) & (Pz==0), 0.0, np.arctan2(Py, Px))
                    
                    map_x = R * np.cos(theta) * (in_w / 2.0) + xcd_in
                    map_y = R * np.sin(theta) * (in_h / 2.0) + ycd_in
                else:
                    map_x = -Px * (in_w / 2.0) + xcd_in
                    map_y = Py * (in_h / 2.0) + ycd_in

        return map_x, map_y, mask
        
    def generate_all_maps(self, transformtype, warp_file, in_w, in_h, out_w, out_h, anglex, angley):
        self.log(f"Generating maps for transform type {transformtype}...")
        
        try:
            if transformtype in [4, 5]:
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
                
                if transformtype == 4:
                    map_x = u_hr * (in_w - 1)
                    map_y = v_hr * (in_h - 1)
                    mask = weight_hr
                else: # Type 5 - Compose Equirect->Fisheye with Fisheye->Warped
                    inter_w, inter_h = in_w, in_h 
                    map_x_4 = np.clip(u_hr * (inter_w - 1), 0, inter_w - 1)
                    map_y_4 = np.clip(v_hr * (inter_h - 1), 0, inter_h - 1)
                    
                    # Generate Type 1 map (Equirect -> 180 Fisheye)
                    map_x_1, map_y_1, mask_1 = self.get_math_maps(1, in_w, in_h, inter_w, inter_h, anglex, angley)
                    
                    # Compose maps using map_coordinates
                    map_x = map_coordinates(map_x_1, [map_y_4, map_x_4], order=1, mode='nearest')
                    map_y = map_coordinates(map_y_1, [map_y_4, map_x_4], order=1, mode='nearest')
                    mask = map_coordinates(mask_1, [map_y_4, map_x_4], order=1, mode='constant', cval=0.0) * weight_hr

            else:
                # Types 0, 1, 2, 3
                map_x, map_y, mask = self.get_math_maps(transformtype, in_w, in_h, out_w, out_h, anglex, angley)

            # Cap values and write out
            map_x = np.clip(map_x, 0, in_w - 1)
            map_y = np.clip(map_y, 0, in_h - 1)
            mask = np.clip(mask, 0, 1)

            map_x_uint = np.round(map_x).astype(np.uint16)
            map_y_uint = np.round(map_y).astype(np.uint16)
            
            self.save_pgm_p2("map_x_directp2.pgm", map_x_uint)
            self.save_pgm_p2("map_y_directp2.pgm", map_y_uint)
            
            mask_img = (mask * 255).astype(np.uint8)
            Image.fromarray(mask_img, mode='L').save("weight_alpha_mask.png")
            
            self.log("Maps generated successfully")
            return True
            
        except Exception as e:
            self.log(f"Error generating maps: {str(e)}")
            messagebox.showerror("Error", f"Failed to generate maps: {str(e)}")
            return False
            
    def start_ffmpeg_conversion(self, input_video, output_video, out_w, out_h):
        self.process_button.config(state=tk.DISABLED) 
        
        if self.crop_to_4k.get():
            leftcoord = self.video_width - 2048
            filter_complex = (
            f"[0:v]crop=4096:4096:{leftcoord}:0[cropped];"
            f"[cropped][1:v][2:v]remap[remapped];"
            f"[3:v]format=gray,scale={self.video_width}:{self.video_height},colorchannelmixer=rr=1:gg=1:bb=1[mask_rgb];"
            f"[remapped][mask_rgb]blend=all_mode=multiply[blended];"
            f"[blended]scale={out_w}:{out_h},setsar=1,setdar=16/9,format=yuv420p[out]"
            )
        else: 
            if self.crop_to_1k.get():
                leftcoord = self.video_width - 540
                filter_complex = (
                f"[0:v]crop=1080:1080:{leftcoord}:0[cropped];"
                f"[cropped][1:v][2:v]remap[remapped];"
                f"[3:v]format=gray,scale={self.video_width}:{self.video_height},colorchannelmixer=rr=1:gg=1:bb=1[mask_rgb];"
                f"[remapped][mask_rgb]blend=all_mode=multiply[blended];"
                f"[blended]scale={out_w}:{out_h},setsar=1,setdar=16/9,format=yuv420p[out]"
                )
            else:
                filter_complex = (
                f"[0:v][1:v][2:v]remap[remapped];"
                f"[3:v]format=gray,scale={self.video_width}:{self.video_height},colorchannelmixer=rr=1:gg=1:bb=1[mask_rgb];"
                f"[remapped][mask_rgb]blend=all_mode=multiply[blended];"
                f"[blended]scale={out_w}:{out_h},setsar=1,setdar=16/9,format=yuv420p[out]"
                )
        
        params = self.get_ffmpeg_params(self.output_codec.get())
        cmd = [
            'ffmpeg', '-y', '-i', input_video,
            '-i', 'map_x_directp2.pgm',
            '-i', 'map_y_directp2.pgm',
            '-i', 'weight_alpha_mask.png',
            '-filter_complex', filter_complex,
            '-map', '[out]',
            '-map', '0:a?',
            '-c:v', self.output_codec.get(),
            ] + params + [output_video]
            
        print("FFmpeg command to copy-paste:")
        print(shlex.join(cmd))
        
        ffmpeg_thread = threading.Thread(target=self.run_ffmpeg_process, args=(cmd,))
        ffmpeg_thread.daemon = True
        ffmpeg_thread.start()
        
        self.root.after(100, self.monitor_ffmpeg_thread, ffmpeg_thread)

    def conversion_complete(self, isSuccess):
        if isSuccess == True:
            self.log(f"ffmpeg conversion complete.")
        else:
            self.log(f"Error during processing!")

    def run_ffmpeg_process(self, cmd):
        process = None
        try:
            self.ffmpeg_process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True
            )
            
            frame_regex = re.compile(r"frame=\s*(\d+)")
            for line in self.ffmpeg_process.stdout:
                self.root.after(0, self.log, line.strip())
                match = frame_regex.search(line)
                if match and self.total_frames > 0:
                    frame_num = int(match.group(1))
                    self.progress['value'] = frame_num
                    status_text = f"Frame {frame_num} of {self.total_frames}"
                    self.status_var.set(status_text)
                
            self.root.update_idletasks()
            self.ffmpeg_process.wait()
            
            if self.ffmpeg_process.returncode == 0:
                self.root.after(0, lambda: self.conversion_complete(True))
            else:
                self.root.after(0, lambda: self.conversion_complete(False))
                
        except FileNotFoundError:
            self.root.after(0, lambda: self.log("Error: ffmpeg not found. Please install ffmpeg and ensure it's in PATH."))
        except Exception as e:
            error_msg = f"Error running ffmpeg: {str(e)}"
            self.root.after(0, lambda: self.log(error_msg))

    def process_video(self):
        try:
            self.progress.start()
            self.process_button.config(state=tk.DISABLED)
            
            if self.is_square:
                out_w, out_h = map(int, self.output_resolution.get().split('x'))
            else:
                out_w, out_h = self.video_width, self.video_height
                
            tt = self.get_current_transform_type()
            ax = self.angle_x.get()
            ay = self.angle_y.get()

            if not self.generate_all_maps(
                tt, self.warp_file.get(), self.video_width, self.video_height, out_w, out_h, ax, ay
            ):
                return
                
            self.start_ffmpeg_conversion(self.input_video.get(), self.output_video.get(), out_w, out_h)
            
        except Exception as e:
            self.log(f"Error during processing: {str(e)}")
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
        finally:
            self.progress.stop()
            self.process_button.config(state=tk.NORMAL)
            
    def start_processing(self):
        try:
            self.progress['maximum'] = self.total_frames
            self.progress['mode'] = 'determinate'
        except Exception as e:
            self.log(f"Could not determine total frames: {e}. Progress bar will be indeterminate.")
            self.progress.start()
            
        thread = threading.Thread(target=self.process_video)
        thread.daemon = True
        thread.start()
        
    def on_close(self):
        if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
            if not self.cancelling:
                self.cancelling = True
                self.status_label.config(text="Cancelling FFmpeg, please wait...")
                self.process_button.config(state=tk.DISABLED)
                self.root.after(100, self.terminate_ffmpeg_and_exit)
                return
        self.root.destroy()

    def terminate_ffmpeg_and_exit(self):
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
