"""
Tkinter desktop UI (customtkinter) for the video analytics pipeline

Purpose
-------
Provides a minimal GUI to pick a video, launch processing on a background thread,
show progress, and display always-on analytics badges.

What it does
------------
- Select video (file dialog) → enables “Process Video”
- Spawns VideoProcessor(video_path, filters) on a thread; wires a progress callback
- Updates a progress bar and status label (orange → green/red)
- Shows non-interactive badges for the four core analytics included in the composite

Primary entry point
-------------------
- App.process_video(): prepares always-true filters and calls VideoProcessor.process_video()
- __main__: launches the Tkinter event loop

Assumptions
-----------
- The core analytics are “always on” and also enforced in the processing layer
- process_video.py (VideoProcessor) is on the Python path
"""


import customtkinter as ctk
from tkinter import filedialog
from process_video import VideoProcessor
import threading

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Pickleball Computer Vision Project")
        self.geometry("1200x500")
        self.video_path = None

        # these vars are kept for compatibility, but not shown as toggles
        self.playerHeatmap_var = ctk.BooleanVar(value=True)
        self.rallyLength_var = ctk.BooleanVar(value=True)
        self.ballHeatmap_var = ctk.BooleanVar(value=True)
        self.kitchenIntrusion_var = ctk.BooleanVar(value=True)

        self.create_widgets()

    def create_widgets(self):
        self.grid_columnconfigure(0, weight=2, uniform="a")
        self.grid_columnconfigure(1, weight=1, uniform="a")
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # LEFT — controls & status
        left_frame = ctk.CTkFrame(self, corner_radius=10)
        left_frame.grid(row=0, column=0, rowspan=3, padx=20, pady=20, sticky="nsew")
        left_frame.grid_columnconfigure(0, weight=1)

        self.select_btn = ctk.CTkButton(left_frame, text="Select Video", command=self.select_video)
        self.select_btn.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")

        self.process_btn = ctk.CTkButton(left_frame, text="Process Video", command=self.process_video_thread, state="disabled")
        self.process_btn.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        # Status
        status_frame = ctk.CTkFrame(left_frame, corner_radius=10, fg_color="#2a2d2e")
        status_frame.grid(row=2, column=0, padx=20, pady=(10, 20), sticky="nsew")
        status_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(status_frame, text="Status:", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, padx=10, pady=(10, 5), sticky="w"
        )
        self.status_label = ctk.CTkLabel(
            status_frame, text="Video not selected", text_color="red", font=ctk.CTkFont(size=13)
        )
        self.status_label.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="w")

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(status_frame)
        self.progress_bar.set(0)
        self.progress_bar.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="ew")

        # RIGHT — Analytics badges (always on)
        right = ctk.CTkFrame(self, corner_radius=10)
        right.grid(row=0, column=1, rowspan=3, padx=20, pady=20, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            right, text="Analytics (Always Included)", font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, padx=10, pady=(20, 6), sticky="w")

        ctk.CTkLabel(
            right,
            text="These analytics are baked into the main composite.\nFuture analytics may export as separate videos.",
            font=ctk.CTkFont(size=12),
            text_color="#A0A0A0",
            justify="left",
        ).grid(row=1, column=0, padx=10, pady=(0, 16), sticky="w")

        # badge helper
        def badge(parent, text, color):
            # small rounded pill with text
            pill = ctk.CTkFrame(parent, fg_color=color, corner_radius=14)
            lbl = ctk.CTkLabel(pill, text=text, font=ctk.CTkFont(size=12, weight="bold"))
            lbl.pack(padx=10, pady=6)
            return pill

        # badges grid
        badges_frame = ctk.CTkFrame(right, corner_radius=10, fg_color="#232527")
        badges_frame.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="nsew")
        badges_frame.grid_columnconfigure((0, 1), weight=1)

        b1 = badge(badges_frame, "Player Heatmap", "#155E75")       # teal-ish
        b2 = badge(badges_frame, "Ball Heatmap", "#4F46E5")         # indigo
        b3 = badge(badges_frame, "Kitchen Detection", "#B45309")    # amber-ish
        b4 = badge(badges_frame, "Rally Length", "#7C3AED")     # purple

        b1.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        b2.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        b3.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        b4.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

    def select_video(self):
        filetypes = (("MP4 files", "*.mp4"), ("All files", "*.*"))
        self.video_path = filedialog.askopenfilename(title="Open video file", filetypes=filetypes)
        if self.video_path:
            self.status_label.configure(text="Video selected", text_color="green")
            self.process_btn.configure(state="normal")
        else:
            self.status_label.configure(text="Video not selected", text_color="red")

    def process_video_thread(self):
        threading.Thread(target=self.process_video, daemon=True).start()

    def process_video(self):
        self.process_btn.configure(state="disabled")
        self.status_label.configure(text="Processing video...", text_color="orange")
        self.progress_bar.set(0)

        # These are always True and reinforced in process_video.py too
        filters = {
            "player_heatmap": True,
            "rally_length": True,
            "ball_heatmap": True,
            "kitchen_detection": True,
        }
        def update_progress(value):
            self.after(0, lambda: self.progress_bar.set(value))
        try:
            processor = VideoProcessor(self.video_path, filters)
            processor.process_video(progress_callback=update_progress)
            self.status_label.configure(text="Processing completed!", text_color="green")
        except Exception as e:
            self.status_label.configure(text=f"Error: {str(e)}", text_color="red")
            print(f"Error: {str(e)}")
        finally:
            self.process_btn.configure(state="normal")

if __name__ == "__main__":
    app = App()
    app.mainloop()
