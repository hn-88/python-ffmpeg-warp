import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import subprocess
import os
import re
import shlex
import threading
import signal
import platform
from fractions import Fraction


class ImageSeqToVideoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Sequence → Video")

        desired_width = 900
        desired_height = 920
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        actual_width = min(desired_width, screen_width - 20)
        actual_height = min(desired_height, screen_height - 20)
        self.root.geometry(f"{actual_width}x{actual_height}")

        # ── codec metadata ──────────────────────────────────────────────────
        self.codec_names = {
            'libx264':          'H.264 (libx264) – Best compatibility',
            'libx265':          'H.265/HEVC (libx265) – Better compression',
            'h264_nvenc':       'H.264 (NVIDIA GPU)',
            'hevc_nvenc':       'H.265 (NVIDIA GPU)',
            'h264_videotoolbox':'H.264 (Apple GPU)',
            'hevc_videotoolbox':'H.265 (Apple GPU)',
            'h264_qsv':         'H.264 (Intel QuickSync)',
            'hevc_qsv':         'H.265 (Intel QuickSync)',
            'h264_amf':         'H.264 (AMD GPU)',
            'hevc_amf':         'H.265 (AMD GPU)',
            'ffvhuff':          'Huffyuv (lossless)',
            'libvpx-vp9':       'VP9',
            'libaom-av1':       'AV1 (libaom)',
            'libsvtav1':        'AV1 (SVT)',
            'prores_ks':        'ProRes',
            'dnxhd':            'DNxHD',
            'mpeg4':            'MPEG-4',
        }

        # Default extra-params per codec (what goes AFTER -c:v <codec>)
        self.codec_extra_params = {
            'libx264': (
                '-preset medium\n'
                '-crf 12\n'
                '-pix_fmt yuv420p\n'
                '-g 30 -keyint_min 30\n'
                '-x264-params "keyint=30:min-keyint=30:scenecut=0:bframes=2"\n'
                '-fps_mode cfr\n'
                '-movflags +faststart'
            ),
            'libx265': (
                '-preset medium\n'
                '-crf 12\n'
                '-pix_fmt yuv420p10le -profile:v main10\n'
                '-g 30 -keyint_min 30\n'
                '-x265-params "keyint=30:min-keyint=30:scenecut=0:bframes=2"\n'
                '-fps_mode cfr\n'
                '-movflags +faststart'
            ),
            'hevc_nvenc': (
                '-preset p7\n'
                '-rc constqp -qp 12\n'
                '-pix_fmt p010le -profile:v main10\n'
                '-g 30 -keyint_min 30 -sc_threshold 0\n'
                '-r {fps} -vsync cfr\n'
                '-movflags +faststart'
            ),
            'h264_nvenc': (
                '-preset p4\n'
                '-rc constqp -qp 18\n'
                '-pix_fmt yuv420p\n'
                '-g 30 -keyint_min 30 -sc_threshold 0\n'
                '-r {fps} -vsync cfr\n'
                '-movflags +faststart'
            ),
            'hevc_videotoolbox': (
                '-quality balanced\n'
                '-b:v 5M -tag:v hvc1\n'
                '-pix_fmt yuv420p\n'
                '-movflags +frag_keyframe+empty_moov\n'
                '-maxrate 8M -bufsize 16M'
            ),
            'h264_videotoolbox': (
                '-quality balanced\n'
                '-b:v 5M -tag:v avc1\n'
                '-pix_fmt yuv420p\n'
                '-movflags +frag_keyframe+empty_moov+faststart\n'
                '-maxrate 8M -bufsize 16M'
            ),
            'ffvhuff': (
                '-pix_fmt yuv420p'
            ),
            'prores_ks': (
                '-profile:v 3\n'
                '-pix_fmt yuv422p10le\n'
                '-vendor apl0'
            ),
        }

        # ── state vars ───────────────────────────────────────────────────────
        self.input_pattern = tk.StringVar()
        self.output_video  = tk.StringVar()
        self.framerate     = tk.StringVar(value="30")
        self.output_codec  = tk.StringVar(value="libx265")

        self.ffmpeg_process = None
        self.cancelling     = False
        self.total_frames   = 0

        self.available_codecs = self._get_available_codecs()

        self._build_ui()

        self.status_var   = tk.StringVar(value="Ready")
        self.status_label = tk.Label(root, textvariable=self.status_var, anchor='w', relief=tk.SUNKEN)
        self.status_label.grid(row=1, column=0, sticky='ew', padx=5, pady=(0, 4))

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ════════════════════════════════════════════════════════════════════════
    # UI construction
    # ════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding="10")
        main.grid(row=0, column=0, sticky='nsew')
        main.columnconfigure(1, weight=1)

        row = 0

        # ── Input pattern ────────────────────────────────────────────────────
        ttk.Label(main, text="Input Pattern:").grid(row=row, column=0, sticky='w', pady=4)
        ttk.Entry(main, textvariable=self.input_pattern, width=52).grid(
            row=row, column=1, sticky='ew', padx=5, pady=4)
        ttk.Button(main, text="Browse…", command=self._browse_input).grid(
            row=row, column=2, padx=5, pady=4)
        row += 1

        self.input_info = ttk.Label(main, text="", foreground="#1a6bb5")
        self.input_info.grid(row=row, column=0, columnspan=3, sticky='w', pady=(0, 6))
        row += 1

        # ── Framerate ────────────────────────────────────────────────────────
        fps_frame = ttk.LabelFrame(main, text="Frame Rate", padding="8")
        fps_frame.grid(row=row, column=0, columnspan=3, sticky='ew', pady=5)
        ttk.Label(fps_frame, text="fps:").grid(row=0, column=0, sticky='w')
        fps_entry = ttk.Entry(fps_frame, textvariable=self.framerate, width=10)
        fps_entry.grid(row=0, column=1, sticky='w', padx=8)
        for val in ("24", "25", "29.97", "30", "48", "50", "59.94", "60", "120"):
            ttk.Button(fps_frame, text=val, width=6,
                       command=lambda v=val: self.framerate.set(v)).grid(
                row=0, column=fps_frame.grid_size()[0], padx=2)
        row += 1

        # ── Codec selection ──────────────────────────────────────────────────
        codec_frame = ttk.LabelFrame(main, text="Output Codec", padding="8")
        codec_frame.grid(row=row, column=0, columnspan=3, sticky='ew', pady=5)
        codec_frame.columnconfigure(1, weight=1)

        ttk.Label(codec_frame, text="Codec:").grid(row=0, column=0, sticky='w')
        display_names = self._get_codec_display_names()
        self.codec_combo = ttk.Combobox(
            codec_frame, values=display_names, state='readonly', width=42)
        self.codec_combo.grid(row=0, column=1, sticky='ew', padx=8)
        # select libx265 by default if available
        try:
            idx = self.available_codecs.index('libx265')
            self.codec_combo.current(idx)
        except ValueError:
            self.codec_combo.current(0)
        self.codec_combo.bind('<<ComboboxSelected>>', self._on_codec_changed)
        row += 1

        # ── Extra params text box ────────────────────────────────────────────
        params_frame = ttk.LabelFrame(main, text="Extra FFmpeg Parameters (after -c:v <codec>)", padding="8")
        params_frame.grid(row=row, column=0, columnspan=3, sticky='ew', pady=5)
        params_frame.columnconfigure(0, weight=1)

        self.params_text = tk.Text(params_frame, height=9, width=75, font=("Courier", 10),
                                   wrap=tk.WORD, relief=tk.SUNKEN, bd=1)
        self.params_text.grid(row=0, column=0, sticky='ew')
        sb = ttk.Scrollbar(params_frame, orient='vertical', command=self.params_text.yview)
        sb.grid(row=0, column=1, sticky='ns')
        self.params_text['yscrollcommand'] = sb.set

        btn_row = ttk.Frame(params_frame)
        btn_row.grid(row=1, column=0, sticky='w', pady=(4, 0))
        ttk.Button(btn_row, text="Reset to defaults", command=self._fill_default_params).grid(
            row=0, column=0, padx=(0, 6))
        ttk.Button(btn_row, text="Clear", command=lambda: self.params_text.delete('1.0', tk.END)).grid(
            row=0, column=1)
        row += 1

        # ── Output video ─────────────────────────────────────────────────────
        ttk.Label(main, text="Output File:").grid(row=row, column=0, sticky='w', pady=4)
        ttk.Entry(main, textvariable=self.output_video, width=52).grid(
            row=row, column=1, sticky='ew', padx=5, pady=4)
        ttk.Button(main, text="Browse…", command=self._browse_output).grid(
            row=row, column=2, padx=5, pady=4)
        row += 1

        # ── Preview command ──────────────────────────────────────────────────
        cmd_frame = ttk.LabelFrame(main, text="Command Preview", padding="8")
        cmd_frame.grid(row=row, column=0, columnspan=3, sticky='ew', pady=5)
        cmd_frame.columnconfigure(0, weight=1)

        self.cmd_preview = tk.Text(cmd_frame, height=5, width=75, font=("Courier", 9),
                                   wrap=tk.WORD, state=tk.DISABLED,
                                   background="#f0f4f8", relief=tk.FLAT)
        self.cmd_preview.grid(row=0, column=0, sticky='ew')
        ttk.Button(cmd_frame, text="Refresh Preview", command=self._refresh_preview).grid(
            row=1, column=0, sticky='e', pady=(4, 0))
        row += 1

        # ── Start / Cancel buttons ───────────────────────────────────────────
        btn_frame = ttk.Frame(main)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=10)
        self.start_btn = ttk.Button(btn_frame, text="▶  Start Processing",
                                    command=self._start_processing)
        self.start_btn.grid(row=0, column=0, padx=10)
        self.cancel_btn = ttk.Button(btn_frame, text="✖  Cancel",
                                     command=self._cancel_processing, state=tk.DISABLED)
        self.cancel_btn.grid(row=0, column=1, padx=10)
        row += 1

        # ── Progress ─────────────────────────────────────────────────────────
        self.progress = ttk.Progressbar(main, mode='indeterminate', length=700)
        self.progress.grid(row=row, column=0, columnspan=3, pady=6, sticky='ew')
        row += 1

        # ── Log ──────────────────────────────────────────────────────────────
        ttk.Label(main, text="Log:").grid(row=row, column=0, sticky='w')
        row += 1
        self.log_text = scrolledtext.ScrolledText(main, height=10, width=80,
                                                  font=("Courier", 9))
        self.log_text.grid(row=row, column=0, columnspan=3, pady=4, sticky='nsew')
        main.rowconfigure(row, weight=1)

        # Populate defaults for the initial codec selection
        self._fill_default_params()
        # Bind live preview refresh
        self.input_pattern.trace_add('write', lambda *_: self._refresh_preview())
        self.output_video.trace_add('write',  lambda *_: self._refresh_preview())
        self.framerate.trace_add('write',     lambda *_: self._refresh_preview())
        self.params_text.bind('<KeyRelease>',  lambda _: self._refresh_preview())

    # ════════════════════════════════════════════════════════════════════════
    # Codec helpers
    # ════════════════════════════════════════════════════════════════════════

    def _get_available_codecs(self):
        try:
            result = subprocess.run(
                ['ffmpeg', '-encoders', '-hide_banner'],
                capture_output=True, text=True, timeout=5)
            found = set(re.findall(r'^\s*V\S*\s+(\S+)', result.stdout, re.MULTILINE))
            preferred = [
                'libx264','libx265','h264_nvenc','hevc_nvenc',
                'h264_videotoolbox','hevc_videotoolbox',
                'h264_qsv','hevc_qsv','h264_amf','hevc_amf',
                'ffvhuff','libvpx-vp9','libaom-av1','libsvtav1',
                'prores_ks','dnxhd','mpeg4',
            ]
            available = [c for c in preferred if c in found]
            return available or ['libx264', 'libx265', 'mpeg4']
        except Exception:
            return ['libx264', 'libx265', 'mpeg4']

    def _get_codec_display_names(self):
        return [self.codec_names.get(c, c) for c in self.available_codecs]

    def _codec_from_display(self, display):
        for codec, name in self.codec_names.items():
            if name == display:
                return codec
        # fallback: check if display IS a codec key
        if display in self.available_codecs:
            return display
        return self.available_codecs[0] if self.available_codecs else 'libx264'

    def _current_codec(self):
        return self._codec_from_display(self.codec_combo.get())

    def _on_codec_changed(self, event=None):
        codec = self._current_codec()
        self.output_codec.set(codec)
        self._fill_default_params()
        self._refresh_preview()
        self._log(f"Codec selected: {codec}")

    def _fill_default_params(self):
        codec  = self._current_codec()
        fps    = self.framerate.get() or "30"
        params = self.codec_extra_params.get(codec, '-pix_fmt yuv420p\n-movflags +faststart')
        params = params.replace('{fps}', fps)
        self.params_text.delete('1.0', tk.END)
        self.params_text.insert(tk.END, params)
        self._refresh_preview()

    # ════════════════════════════════════════════════════════════════════════
    # Browse helpers
    # ════════════════════════════════════════════════════════════════════════

    def _browse_input(self):
        """Let user pick any one file from the sequence; auto-detect pattern."""
        path = filedialog.askopenfilename(
            title="Select any frame from the sequence",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.tiff *.tif *.exr *.dpx *.bmp"),
                ("All files", "*.*"),
            ])
        if not path:
            return

        pattern, count, start = self._detect_pattern(path)
        self.input_pattern.set(pattern)
        info = f"Detected: {pattern}  |  ~{count} frames found  |  start #{start}"
        self.input_info.config(text=info)
        self._log(info)

        if count > 0:
            self.total_frames = count
            try:
                self.progress['maximum'] = count
                self.progress['mode'] = 'determinate'
            except Exception:
                pass

        # Suggest output filename based on folder name
        folder = os.path.dirname(path)
        suggested_name = os.path.basename(folder) + ".mp4"
        suggested_out  = os.path.join(folder, suggested_name)
        if not self.output_video.get():
            self.output_video.set(suggested_out)

        self._refresh_preview()

    def _detect_pattern(self, sample_path):
        """Given one file from a sequence, return (pattern, count, first_number)."""
        folder   = os.path.dirname(sample_path)
        basename = os.path.basename(sample_path)

        # Find the numeric run in the filename (last one wins for sequences)
        matches = list(re.finditer(r'(\d+)', basename))
        if not matches:
            return sample_path, 1, 0

        # Use the last numeric group (most likely the frame counter)
        m      = matches[-1]
        prefix = basename[:m.start()]
        suffix = basename[m.end():]
        digits = len(m.group())
        number = int(m.group())

        pattern = os.path.join(folder, f"{prefix}%0{digits}d{suffix}")

        # Count matching files
        file_re = re.compile(
            r'^' + re.escape(prefix) + r'(\d{' + str(digits) + r',})' + re.escape(suffix) + r'$')
        frames = []
        try:
            for f in os.listdir(folder):
                fm = file_re.match(f)
                if fm:
                    frames.append(int(fm.group(1)))
        except Exception:
            pass

        frames.sort()
        first = frames[0] if frames else number
        return pattern, len(frames), first

    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            title="Save output video as",
            defaultextension=".mp4",
            filetypes=[
                ("MP4",  "*.mp4"),
                ("MKV",  "*.mkv"),
                ("MOV",  "*.mov"),
                ("All",  "*.*"),
            ])
        if path:
            self.output_video.set(path)
            self._refresh_preview()

    # ════════════════════════════════════════════════════════════════════════
    # Command building
    # ════════════════════════════════════════════════════════════════════════

    def _build_command(self):
        inp    = self.input_pattern.get().strip()
        out    = self.output_video.get().strip()
        fps    = self.framerate.get().strip() or "30"
        codec  = self._current_codec()
        extras = self.params_text.get('1.0', tk.END).strip()

        # Parse the extra params text into a flat list, splitting on newlines
        # and then shell-splitting each line so quoted strings stay together
        extra_tokens = []
        for line in extras.splitlines():
            line = line.strip()
            if line:
                try:
                    extra_tokens.extend(shlex.split(line))
                except ValueError:
                    extra_tokens.extend(line.split())

        cmd = [
            'ffmpeg',
            '-framerate', fps,
            '-i', inp,
            '-c:v', codec,
        ] + extra_tokens + [
            '-y', out,
        ]
        return cmd

    def _refresh_preview(self, *_):
        try:
            cmd = self._build_command()
            pretty = shlex.join(cmd)
        except Exception as e:
            pretty = f"(cannot build command: {e})"

        self.cmd_preview.config(state=tk.NORMAL)
        self.cmd_preview.delete('1.0', tk.END)
        self.cmd_preview.insert(tk.END, pretty)
        self.cmd_preview.config(state=tk.DISABLED)

    # ════════════════════════════════════════════════════════════════════════
    # Processing
    # ════════════════════════════════════════════════════════════════════════

    def _start_processing(self):
        inp = self.input_pattern.get().strip()
        out = self.output_video.get().strip()

        if not inp:
            messagebox.showwarning("Missing input", "Please select an input image sequence.")
            return
        if not out:
            messagebox.showwarning("Missing output", "Please specify an output file.")
            return

        self.start_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)

        try:
            self.progress['mode'] = 'determinate'
            self.progress['value'] = 0
        except Exception:
            self.progress.config(mode='indeterminate')
            self.progress.start()

        cmd = self._build_command()
        self._log("Starting FFmpeg…")
        self._log("Command: " + shlex.join(cmd))

        t = threading.Thread(target=self._run_ffmpeg, args=(cmd,), daemon=True)
        t.start()

    def _run_ffmpeg(self, cmd):
        try:
            self.ffmpeg_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )
            frame_re = re.compile(r'frame=\s*(\d+)')
            for line in self.ffmpeg_process.stdout:
                self.root.after(0, self._log, line.rstrip())
                m = frame_re.search(line)
                if m and self.total_frames > 0:
                    n = int(m.group(1))
                    self.root.after(0, self._set_progress, n)

            self.ffmpeg_process.wait()
            rc = self.ffmpeg_process.returncode
            if rc == 0:
                self.root.after(0, self._done, True)
            else:
                self.root.after(0, self._done, False, rc)

        except FileNotFoundError:
            self.root.after(0, self._log,
                            "ERROR: ffmpeg not found. Install ffmpeg and add it to PATH.")
            self.root.after(0, self._done, False, -1)
        except Exception as e:
            self.root.after(0, self._log, f"ERROR: {e}")
            self.root.after(0, self._done, False, -1)

    def _set_progress(self, n):
        try:
            self.progress['value'] = n
            self.status_var.set(f"Frame {n} / {self.total_frames}")
        except Exception:
            pass

    def _done(self, success, returncode=0):
        self.start_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)
        try:
            self.progress.stop()
            self.progress['value'] = self.total_frames if success else 0
        except Exception:
            pass

        if success:
            self.status_var.set("✔  Done!")
            self._log("═" * 60)
            self._log("✔  Processing complete.")
            self._log(f"   Output: {self.output_video.get()}")
            messagebox.showinfo("Done", f"Video saved to:\n{self.output_video.get()}")
        else:
            self.status_var.set(f"✖  Failed (exit code {returncode})")
            self._log(f"✖  FFmpeg exited with code {returncode}.")

    def _cancel_processing(self):
        if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
            self._log("Cancelling…")
            self.cancelling = True
            try:
                if platform.system() == "Windows":
                    self.ffmpeg_process.terminate()
                else:
                    self.ffmpeg_process.send_signal(signal.SIGINT)
            except Exception as e:
                self._log(f"Cancel error: {e}")
            self.cancel_btn.config(state=tk.DISABLED)

    def _cancel_and_wait(self):
        if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
            try:
                if platform.system() == "Windows":
                    self.ffmpeg_process.terminate()
                else:
                    self.ffmpeg_process.send_signal(signal.SIGINT)
                self.root.after(3000, self._force_kill)
                return
            except Exception:
                pass
        self.root.destroy()

    def _force_kill(self):
        if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
            try:
                self.ffmpeg_process.kill()
            except Exception:
                pass
        self.root.destroy()

    def _on_close(self):
        if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
            if not self.cancelling:
                self.cancelling = True
                self.status_var.set("Cancelling FFmpeg, please wait…")
                self.root.after(100, self._cancel_and_wait)
                return
        self.root.destroy()

    # ════════════════════════════════════════════════════════════════════════
    # Logging
    # ════════════════════════════════════════════════════════════════════════

    def _log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()


def main():
    root = tk.Tk()
    app = ImageSeqToVideoGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
