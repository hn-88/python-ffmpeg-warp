import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import subprocess
import os
import cv2
import numpy as np
import threading
import json
from scipy.ndimage import zoom
from PIL import Image

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
        
        self.create_widgets()
        
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
        
        # Output video
        ttk.Label(main_frame, text="Output Video:").grid(row=4, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_video, width=50).grid(row=4, column=1, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_output_video).grid(row=4, column=2, padx=5, pady=5)
        
        # Process button
        self.process_button = ttk.Button(main_frame, text="Start Processing", command=self.start_processing, state=tk.DISABLED)
        self.process_button.grid(row=5, column=0, columnspan=3, pady=20)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate', length=600)
        self.progress.grid(row=6, column=0, columnspan=3, pady=10)
        
        # Log output
        ttk.Label(main_frame, text="Log Output:").grid(row=7, column=0, columnspan=3, sticky=tk.W)
        self.log_text = scrolledtext.ScrolledText(main_frame, height=15, width=80)
        self.log_text.grid(row=8, column=0, columnspan=3, pady=5)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
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
            
    def run_ffmpeg(self, input_video, output_video, out_w, out_h):
        """Run ffmpeg command to process video"""
        self.log("Starting ffmpeg processing...")
        
        filter_complex = (
            f"[0:v][1:v][2:v]remap[remapped];"
            f"[3:v]format=gray,scale={self.video_width}:{self.video_height},colorchannelmixer=rr=1:gg=1:bb=1[mask_rgb];"
            f"[remapped][mask_rgb]blend=all_mode=multiply[blended];"
            f"[blended]scale={out_w}:{out_h}[out]"
        )
        
        cmd = [
            'ffmpeg', '-y', '-i', input_video,
            '-i', 'map_x_directp2.pgm',
            '-i', 'map_y_directp2.pgm',
            '-i', 'weight_alpha_mask.png',
            '-filter_complex', filter_complex,
            '-map', '[out]',
            '-map', '0:a',
            '-c:v', 'hevc_nvenc',
            '-preset', 'p5',
            '-cq', '23',
            '-rc', 'vbr',
            '-maxrate', '15M',
            '-bufsize', '26M',
            '-c:a', 'aac',
            '-b:a', '128k',
            output_video
        ]
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            for line in process.stdout:
                self.log(line.strip())
                
            process.wait()
            
            if process.returncode == 0:
                self.log("Processing completed successfully!")
                messagebox.showinfo("Success", "Video processing completed!")
            else:
                self.log(f"FFmpeg exited with code {process.returncode}")
                messagebox.showerror("Error", "FFmpeg processing failed. Check log for details.")
                
        except FileNotFoundError:
            self.log("Error: ffmpeg not found. Please install ffmpeg and ensure it's in PATH.")
            messagebox.showerror("Error", "ffmpeg not found. Please install ffmpeg.")
        except Exception as e:
            self.log(f"Error running ffmpeg: {str(e)}")
            messagebox.showerror("Error", f"Failed to run ffmpeg: {str(e)}")
            
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
            self.run_ffmpeg(
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

def main():
    root = tk.Tk()
    app = VideoWarpGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
