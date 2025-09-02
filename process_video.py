"""
video_processor.py (refactored)

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

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np

from ball_tracker import BallTracker
from player_tracker import PlayerTracker
from court_detection import CourtDetector
from analytics import Analytics

# ==============================================================================
# Module‑level constants (easy to tweak and reuse)
# ==============================================================================
PROJECT_DIR: str = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR: str = os.path.join(PROJECT_DIR, "models")
OUTPUT_ROOT: str = os.path.join(PROJECT_DIR, "video_outputs")

# Default analytics that should always be ON unless explicitly disabled
DEFAULT_ENABLED: Tuple[str, ...] = (
    "player_heatmap",
    "ball_heatmap",
    "kitchen_detection",
    "court_zone",
)

# Output layout ratios
MAIN_RATIO: float = 0.42  # left column: main video
BE_RATIO: float = 0.26    # middle column: bird's‑eye
GRID_RATIO: float = 0.32  # right column: 2×2 analytics
OUTPUT_WIDTH_SCALE: float = 1.8

# Drawing styles
COLOR_PLAYER: Tuple[int, int, int] = (0, 255, 0)
COLOR_BALL: Tuple[int, int, int] = (0, 0, 255)
COLOR_KP: Tuple[int, int, int] = (255, 0, 0)
COLOR_KP_ID: Tuple[int, int, int] = (0, 255, 255)
COLOR_COURT_EDGE: Tuple[int, int, int] = (255, 255, 255)
BIRD_BG: Tuple[int, int, int] = (0, 0, 0)
KP_RADIUS: int = 5
PLAYER_DOT_RADIUS: int = 8
BALL_DOT_RADIUS: int = 6
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE: float = 0.7
FONT_THICKNESS: int = 2

# Video
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")
FRAME_SLEEP_SEC: float = 0.001  # UI breathing room

# Court keypoint connection topology (must match your court model)
CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2),
    (3, 4), (4, 5),
    (6, 7), (7, 8),
    (9, 10), (10, 11),
    (0, 3), (3, 6), (6, 9),
    (1, 4), (4, 7), (7, 10),
    (2, 5), (5, 8), (8, 11),
)


# ==============================================================================
# Video Processor
# ==============================================================================
class VideoProcessor:
    def __init__(self, video_path: str, filters: dict):
        self.video_path = video_path
        self.filters = self._apply_default_filters(filters)

        self.ball_tracker = BallTracker(os.path.join(MODELS_DIR, "ball_tracking.pt"))
        self.player_tracker = PlayerTracker(os.path.join(MODELS_DIR, "player_tracking.pt"))
        self.court_mapper = CourtDetector(os.path.join(MODELS_DIR, "court_detection.pt"))
        self.analytics = Analytics(self.filters)

        self.output_dir = self._make_output_dir()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_video(self, progress_callback=None) -> str:
        cap = self._open_capture(self.video_path)
        total_frames, src_w, src_h, fps = self._read_video_meta(cap)
        out_w, out_h, main_w, be_w, grid_w, panel_w, panel_h = self._compute_layout(src_w, src_h)

        # Init analytics context
        self.analytics.set_canvas_size(src_w, src_h)
        self.analytics.set_video_context(total_frames=total_frames, fps=fps)

        out_path, writer = self._create_writer(out_w, out_h, fps)

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # --- Detection & projection (single pass) ---
                kps, Hmg = self.court_mapper.get_keypoints_and_homography(frame)
                players, proj_players = self.player_tracker.detect_and_project(frame, Hmg)

                ball_det = self.ball_tracker.detect_frame(frame)
                ball_bbox, ball_proj = self.ball_tracker.process_and_project(ball_det, frame, Hmg)

                # Columns
                main_col = self._render_main_view(frame, players, ball_bbox, kps, (main_w, out_h))
                bird_col = self._render_birdseye(src_w, src_h, kps, Hmg, proj_players, ball_proj, (be_w, out_h))

                # Analytics update + panels
                self.analytics.update_counters(frame_idx, proj_players, ball_proj)
                grid_col = self._render_analytics_grid((panel_w, panel_h), (grid_w, out_h), bird_reference=bird_col)

                # Compose & write
                composite = cv2.hconcat([main_col, bird_col, grid_col])
                writer.write(composite)

                frame_idx += 1
                self._report_progress(progress_callback, frame_idx, total_frames)
        finally:
            cap.release()
            writer.release()
            self.analytics.save_outputs()
            self._report_progress(progress_callback, total_frames, total_frames)

        print(f"Saved: {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # Helpers — configuration & IO
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_default_filters(filters: dict) -> dict:
        filters = dict(filters or {})
        for k in DEFAULT_ENABLED:
            filters[k] = True
        return filters

    @staticmethod
    def _open_capture(path: str) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        return cap

    @staticmethod
    def _read_video_meta(cap: cv2.VideoCapture) -> Tuple[int, int, int, int]:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        return total, w, h, fps

    @staticmethod
    def _compute_layout(src_w: int, src_h: int) -> Tuple[int, int, int, int, int, int, int]:
        out_w = int(src_w * OUTPUT_WIDTH_SCALE)
        out_h = src_h
        main_w = int(out_w * MAIN_RATIO)
        be_w = int(out_w * BE_RATIO)
        grid_w = out_w - main_w - be_w
        panel_w = grid_w // 2
        panel_h = out_h // 2
        return out_w, out_h, main_w, be_w, grid_w, panel_w, panel_h

    def _make_output_dir(self) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(OUTPUT_ROOT, f"run_{ts}")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _create_writer(self, out_w: int, out_h: int, fps: int) -> Tuple[str, cv2.VideoWriter]:
        out_path = os.path.join(self.output_dir, "Main_overlay.mp4")
        writer = cv2.VideoWriter(out_path, FOURCC, fps, (out_w, out_h))
        if not writer.isOpened():
            raise RuntimeError("Failed to open VideoWriter")
        return out_path, writer

    @staticmethod
    def _report_progress(cb, idx: int, total: int) -> None:
        if cb is None or total <= 0:
            return
        cb(min(idx / total, 1.0))
        time.sleep(FRAME_SLEEP_SEC)

    # ------------------------------------------------------------------
    # Helpers — rendering
    # ------------------------------------------------------------------
    def _render_main_view(
        self,
        frame: np.ndarray,
        players: Optional[Iterable[Tuple[float, float, float, float]]],
        ball_bbox: Optional[Tuple[float, float, float, float]],
        keypoints: Optional[np.ndarray],
        target_size: Tuple[int, int],
    ) -> np.ndarray:
        canvas = frame.copy()

        # Players
        for p in players or []:
            x1, y1, x2, y2 = map(int, p)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_PLAYER, 2)

        # Ball
        if ball_bbox:
            x1, y1, x2, y2 = map(int, ball_bbox)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_BALL, 2)

        # Court keypoints
        if keypoints is not None:
            for i, pt in enumerate(keypoints):
                x, y = map(int, pt)
                cv2.circle(canvas, (x, y), KP_RADIUS, COLOR_KP, -1)
                cv2.putText(canvas, str(i), (x + 5, y - 10), FONT, FONT_SCALE, COLOR_KP_ID, FONT_THICKNESS)
            for a, b in CONNECTIONS:
                pt1 = tuple(map(int, keypoints[a]))
                pt2 = tuple(map(int, keypoints[b]))
                cv2.line(canvas, pt1, pt2, COLOR_KP_ID, 2)

        w, h = target_size
        return cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)

    def _render_birdseye(
        self,
        src_w: int,
        src_h: int,
        keypoints: Optional[np.ndarray],
        Hmg: Optional[np.ndarray],
        projected_players: Optional[Iterable[Tuple[float, float]]],
        ball_proj: Optional[Tuple[float, float]],
        target_size: Tuple[int, int],
    ) -> np.ndarray:
        bird = np.zeros((src_h, src_w, 3), dtype=np.uint8)
        bird[:, :] = BIRD_BG

        if keypoints is not None and Hmg is not None:
            pts = np.array(keypoints, dtype=np.float32).reshape(-1, 1, 2)
            proj_kps = cv2.perspectiveTransform(pts, Hmg).reshape(-1, 2)

            # Teach analytics dynamic zones based on projected keypoints
            self.analytics.update_kitchen_from_keypoints(proj_kps)
            self.analytics.update_court_bounds_from_keypoints(proj_kps)
            self.analytics.update_zones_from_keypoints(proj_kps)

            for i, pt in enumerate(proj_kps):
                x, y = map(int, pt)
                if 0 <= x < src_w and 0 <= y < src_h:
                    cv2.circle(bird, (x, y), KP_RADIUS, COLOR_KP, -1)
                    cv2.putText(bird, str(i), (x + 5, y - 10), FONT, FONT_SCALE, COLOR_KP_ID, FONT_THICKNESS)
            for a, b in CONNECTIONS:
                pt1 = tuple(map(int, proj_kps[a]))
                pt2 = tuple(map(int, proj_kps[b]))
                cv2.line(bird, pt1, pt2, COLOR_COURT_EDGE, 2)

        # Players
        for pt in projected_players or []:
            x, y = map(int, pt)
            if 0 <= x < src_w and 0 <= y < src_h:
                cv2.circle(bird, (x, y), PLAYER_DOT_RADIUS, COLOR_PLAYER, -1)

        # Ball
        if ball_proj is not None:
            x, y = map(int, ball_proj)
            if 0 <= x < src_w and 0 <= y < src_h:
                cv2.circle(bird, (x, y), BALL_DOT_RADIUS, COLOR_BALL, -1)

        w, h = target_size
        return cv2.resize(bird, (w, h), interpolation=cv2.INTER_AREA)

    def _render_analytics_grid(
        self,
        panel_size: Tuple[int, int],
        target_size: Tuple[int, int],
        bird_reference: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        panel_w, panel_h = panel_size

        ph = self.analytics.panel_player_heatmap((panel_w, panel_h), bird_reference=bird_reference)
        bh = self.analytics.panel_ball_heatmap((panel_w, panel_h), bird_reference=bird_reference)
        kd = self.analytics.panel_kitchen_intrusion(None, (panel_w, panel_h))  # players provided via update_counters
        rl = self.analytics.panel_rally_tempo((panel_w, panel_h))

        top = cv2.hconcat([ph, bh])
        bot = cv2.hconcat([kd, rl])
        grid = cv2.vconcat([top, bot])

        w, h = target_size
        return cv2.resize(grid, (w, h), interpolation=cv2.INTER_AREA)