"""
video_processor.py

High-level orchestration of the pickleball video pipeline:
- Loads a source video
- Runs court, player, and ball detection
- Projects detections into bird's-eye space via homography
- Updates analytics (heatmaps, kitchen intrusion, rally tempo)
- Renders a composite output with: Main View | Bird's-eye | 2×2 Analytics

Dependencies:
    - OpenCV (cv2)
    - numpy
    - Local modules: BallTracker, PlayerTracker, CourtDetector, Analytics

"""

import cv2
from ball_tracker import BallTracker
from player_tracker import PlayerTracker
from court_detection import CourtDetector
from analytics import Analytics
import os
from datetime import datetime
import numpy as np
import time


project_dir = os.path.dirname(os.path.abspath(__file__))

class VideoProcessor:
    def __init__(self, video_path, filters):
        self.video_path = video_path

        # Force the four core analytics to be ON by default
        defaults_on = ["player_heatmap", "ball_heatmap", "kitchen_detection", "court_zone"]
        for key in defaults_on:
            filters[key] = True

        self.filters = filters
        self.ball_tracker = BallTracker(os.path.join(project_dir, "models", "ball_tracking.pt"))
        self.player_tracker = PlayerTracker(os.path.join(project_dir, "models", "player_tracking.pt"))
        self.court_mapper = CourtDetector(os.path.join(project_dir, "models", "court_detection.pt"))
        self.analytics = Analytics(filters)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(project_dir, "video_outputs", f"run_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)

    def process_video(self, progress_callback=None):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        total_frames= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Source video dimensions
        W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        FPS = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        # -------------------------------------------------------
        # Layout ratios — shrink main video, give analytics more space
        MAIN_RATIO = 0.42   # left column: main video
        BE_RATIO   = 0.26   # middle column: bird's-eye
        GRID_RATIO = 0.32   # right column: 2x2 analytics panels (heatmaps etc.)

        OUT_W = int(W * 1.8)     # overall output width scale (tweak as desired)
        OUT_H = H                 # keep original height

        MAIN_W = int(OUT_W * MAIN_RATIO)
        BE_W   = int(OUT_W * BE_RATIO)
        GRID_W = OUT_W - MAIN_W - BE_W  # ensures exact fit

        PANEL_W = GRID_W // 2
        PANEL_H = OUT_H // 2
        # -------------------------------------------------------

        # Bird’s-eye accumulators use this canvas size
        self.analytics.set_canvas_size(W, H)
        self.analytics.set_video_context(total_frames=total_frames, fps=FPS)

        out_path = os.path.join(self.output_dir, "Main_overlay.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (OUT_W, OUT_H))
        
        # 12-keypoint connection indices (must match your court model's order)
        connections = [
            (0, 1), (1, 2),
            (3, 4), (4, 5),
            (6, 7), (7, 8),
            (9,10), (10,11),
            (0, 3), (3, 6), (6, 9),
            (1, 4), (4, 7), (7,10),
            (2, 5), (5, 8), (8,11)
        ]

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # --- Detect / project (single pass per frame) ---
            keypoints, Hmg = self.court_mapper.get_keypoints_and_homography(frame)
            players, projected_players = self.player_tracker.detect_and_project(frame, Hmg)

            # Detect ball for THIS frame only; no interpolation, no preload
            # (Assumes BallTracker.process_and_project accepts the per-frame detection output
            # returned by BallTracker.detect_frame)
            ball_det = self.ball_tracker.detect_frame(frame)
            ball_bbox, ball_proj = self.ball_tracker.process_and_project(ball_det, frame, Hmg)

            # --- LEFT: main annotated view ---
            main_view = frame.copy()
            for p in players or []:
                x1, y1, x2, y2 = map(int, p)
                cv2.rectangle(main_view, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if ball_bbox:
                x1, y1, x2, y2 = map(int, ball_bbox)
                cv2.rectangle(main_view, (x1, y1), (x2, y2), (0, 0, 255), 2)

            if keypoints is not None:
                for i, pt in enumerate(keypoints):
                    x, y = map(int, pt)
                    cv2.circle(main_view, (x, y), 5, (255, 0, 0), -1)
                    cv2.putText(main_view, str(i), (x + 5, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                for a, b in connections:
                    pt1 = tuple(map(int, keypoints[a]))
                    pt2 = tuple(map(int, keypoints[b]))
                    cv2.line(main_view, pt1, pt2, (0, 255, 255), 2)

            # --- MIDDLE: bird's-eye projection ---
            bird = np.zeros((H, W, 3), dtype=np.uint8)
            if keypoints is not None and Hmg is not None:
                pts = np.array(keypoints, dtype='float32').reshape(-1, 1, 2)
                proj_kpts = cv2.perspectiveTransform(pts, Hmg).reshape(-1, 2)
                # Teach analytics the kitchen band from the current projected keypoints
                self.analytics.update_kitchen_from_keypoints(proj_kpts)
                self.analytics.update_court_bounds_from_keypoints(proj_kpts)
                self.analytics.update_zones_from_keypoints(proj_kpts)

                for i, pt in enumerate(proj_kpts):
                    x, y = map(int, pt)
                    if 0 <= x < W and 0 <= y < H:
                        cv2.circle(bird, (x, y), 5, (255, 0, 0), -1)
                        cv2.putText(bird, str(i), (x + 5, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                for a, b in connections:
                    pt1 = tuple(map(int, proj_kpts[a]))
                    pt2 = tuple(map(int, proj_kpts[b]))
                    cv2.line(bird, pt1, pt2, (255, 255, 255), 2)

            for pt in projected_players or []:
                x, y = map(int, pt)
                if 0 <= x < W and 0 <= y < H:
                    cv2.circle(bird, (x, y), 8, (0, 255, 0), -1)
            if ball_proj is not None:
                x, y = map(int, ball_proj)
                if 0 <= x < W and 0 <= y < H:
                    cv2.circle(bird, (x, y), 6, (0, 0, 255), -1)

            # --- Update analytics state ---
            self.analytics.update_counters(frame_idx, projected_players, ball_proj)

            # --- RIGHT: analytics 2×2 grid, with heatmaps blended over bird ---
            ph = self.analytics.panel_player_heatmap((PANEL_W, PANEL_H), bird_reference=bird)
            bh = self.analytics.panel_ball_heatmap((PANEL_W, PANEL_H),   bird_reference=bird)
            kd = self.analytics.panel_kitchen_intrusion(projected_players, (PANEL_W, PANEL_H))
            rl = self.analytics.panel_rally_tempo((PANEL_W, PANEL_H))

            top_row = cv2.hconcat([ph, bh])
            bot_row = cv2.hconcat([kd, rl])
            grid = cv2.vconcat([top_row, bot_row])

            # --- Resize columns to their target widths/heights ---
            main_col = cv2.resize(main_view, (MAIN_W, OUT_H), interpolation=cv2.INTER_AREA)
            bird_col = cv2.resize(bird,     (BE_W,   OUT_H), interpolation=cv2.INTER_AREA)
            grid_col = cv2.resize(grid,     (GRID_W, OUT_H), interpolation=cv2.INTER_AREA)

            composite = cv2.hconcat([main_col, bird_col, grid_col])
            writer.write(composite)

            frame_idx+=1
            if progress_callback:
                progress_callback(min(frame_idx / total_frames, 1.0))
            time.sleep(0.001)
        
        cap.release()
        writer.release()
        self.analytics.save_outputs()
        if progress_callback:
            progress_callback(1.0)
        print(f"Saved: {out_path}")
