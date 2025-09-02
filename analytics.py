"""
Analytics module

Purpose
-------
Maintains running, bird’s-eye analytics derived from per-frame detections and homography,
then renders small dashboard panels for the composite video.

What it does
------------
- Accumulates player/ball heatmaps with Gaussian stamping
- Learns kitchen band and court bounds from projected 12-keypoint layouts (EMA smoothing)
- Tracks who is in the kitchen *this frame* (no entry counts) and rough zone usage
- Estimates rally length and tempo (rallies/min) with simple in-bounds heuristics
- Renders 2×2 panel tiles: player heatmap, ball heatmap, kitchen overlay, rally stats

Key APIs
--------
- set_canvas_size(width, height): initialize accumulators and kernels
- set_video_context(total_frames, fps): tune contrast/gamma and per-hit weights
- update_kitchen_from_keypoints(proj_kpts), update_court_bounds_from_keypoints(...),
  update_zones_from_keypoints(...): learn geometry from homography-projected keypoints
- update_counters(frame_idx, projected_players, ball_proj): per-frame analytics update
- panel_*(): return panel images sized to (w, h) for composition
- save_outputs(): placeholder for future persistence

Inputs
------
- Projected player points (bird coords), projected ball point, projected 12 court keypoints

Outputs
-------
- Numpy images for each analytics panel; internal heatmaps/metrics for display

Assumptions
-----------
- 12 keypoints are arranged as 4 rows × 3 columns in row-major order
- Coordinates passed to updates are already homography-projected to bird space
"""


import numpy as np
import cv2
from collections import defaultdict

class Analytics:
    def __init__(self, filters):
        self.filters = filters

        # Accumulators (bird’s-eye coords)
        self.canvas_w = None
        self.canvas_h = None
        self.player_heat_accum = None
        self.ball_heat_accum   = None

       # --- Kitchen / zone tracking ---
        self.zone_counts = defaultdict(int)      # <-- total intrusions over the whole video
        self.players_in_kitchen = set()         # who is inside this frame (indices)
        self.kitchen_hits_last_frame = 0        # display count of who is in kitchen now

        # ===== Base Tunables =====
        # Gaussian stamps per hit
        self.stamp_radius_player = 10
        self.stamp_sigma_player  = 6.0
        self.stamp_radius_ball   = 8
        self.stamp_sigma_ball    = 4.0

        # Base per-hit energy (will be scaled adaptively)
        self._base_inc_player = 1.5
        self._base_inc_ball   = 2.0
        self.increment_per_hit_player = self._base_inc_player
        self.increment_per_hit_ball   = self._base_inc_ball

        # Blur on accumulators (we already stamp Gaussians)
        self.blur_kernel = 15  # must be odd

        # Render controls (adaptive overrides will tweak these)
        self.clip_percentiles = (2.0, 98.0)
        self.gamma = 0.8  # <1 brightens midtones

        # Overlay alpha
        self.heat_alpha_player = 0.60
        self.heat_alpha_ball   = 0.60

        # Precomputed kernels
        self._player_kernel = None
        self._ball_kernel   = None

        # Video context (set by set_video_context)
        self._total_frames = None
        self._fps = None

        # --- Dynamic kitchen band learned from keypoints (bird coords) ---
        self._kitchen_y_min = None   # float (bird Y)
        self._kitchen_y_max = None   # float (bird Y)
        self._kitchen_ema_alpha = 0.2  # smoothing for stability

        # --- Rally length & tempo tracking ---
        self._fps = None                 # set via set_video_context
        self._court_bounds = None        # (xmin, ymin, xmax, ymax) in bird coords (smoothed)
        self._bounds_alpha = 0.2         # EMA smoothing for bounds
        self._rally_active = False
        self._rally_frames = 0
        self._gap_frames = 0
        self._gap_threshold = 18         # ~0.6s at 30 fps (tune)
        self._rallies = []               # list of rally lengths in frames
        self._elapsed_frames = 0         # for tempo calculation (overall)

        # Learned court zone polygons in bird coords (updated every frame)
        self._zone_polys = {
            "backcourt_top": None,     # np.ndarray of shape (4,2)
            "kitchen": None,           # np.ndarray of shape (4,2)
            "backcourt_bottom": None,  # np.ndarray of shape (4,2)
}

    # ---------- context ----------
    def set_canvas_size(self, width, height):
        self.canvas_w, self.canvas_h = width, height
        if self.filters.get("player_heatmap"):
            self.player_heat_accum = np.zeros((height, width), dtype=np.float32)
        if self.filters.get("ball_heatmap"):
            self.ball_heat_accum = np.zeros((height, width), dtype=np.float32)

        self._player_kernel = self._make_gaussian_kernel(self.stamp_radius_player, self.stamp_sigma_player)
        self._ball_kernel   = self._make_gaussian_kernel(self.stamp_radius_ball,   self.stamp_sigma_ball)

    def set_video_context(self, total_frames: int, fps: int = 30):
        """
        Softer adaptation: short clips get a modest boost, not a blast.
        """
        self._total_frames = max(int(total_frames or 1), 1)
        self._fps = max(int(fps or 30), 1)

        # Reference ~5 minutes
        ref_frames = self._fps * 300  # 300s

        # ↓ Softer inverse-power gain and tighter clamp
        beta = 0.5                                    # was 0.7
        raw_gain = (ref_frames / self._total_frames) ** beta
        gain = float(np.clip(raw_gain, 0.80, 1.80))   # was [0.75, 2.5]

        # Apply to per-hit increments
        self.increment_per_hit_player = self._base_inc_player * gain
        self.increment_per_hit_ball   = self._base_inc_ball   * gain

        # ↓ Gentler gamma curve (closer to neutral even for short clips)
        # Map N in [~30s, ~10min] -> gamma in [0.90, 1.05]   (was [0.75, 1.05])
        n30s = self._fps * 30
        n10m = self._fps * 600
        self.gamma = float(np.interp(self._total_frames, [n30s, n10m], [0.90, 1.05]))
        self.gamma = float(np.clip(self.gamma, 0.90, 1.05))

        # ↓ Less aggressive contrast stretch for short clips
        # [short] (4, 98.5) -> [long] (1.5, 99.2)
        lo = np.interp(self._total_frames, [n30s, n10m], [4.0, 1.5])
        hi = np.interp(self._total_frames, [n30s, n10m], [98.5, 99.2])
        self.clip_percentiles = (float(lo), float(hi))


    # ---------- per-frame updates ----------
    def update_counters(self, frame_idx, projected_players, ball_proj):
        # Zone usage counts (based on mid-band kitchen)
        for pt in projected_players or []:
            zone = self._get_zone(pt)
            if zone:
                self.zone_counts[zone] += 1

        # Who is in the kitchen THIS frame (no history/entries)
        current_in = {i for i, pt in enumerate(projected_players or []) if self._in_kitchen(pt)}
        self.players_in_kitchen = current_in
        self.kitchen_hits_last_frame = len(current_in)

        # Heat accumulation (Gaussian stamps)
        if self.player_heat_accum is not None:
            for pt in projected_players or []:
                x, y = map(int, pt)
                self._stamp(self.player_heat_accum, x, y, self._player_kernel, self.increment_per_hit_player)

        if self.ball_heat_accum is not None and ball_proj is not None:
            x, y = map(int, ball_proj)
            self._stamp(self.ball_heat_accum, x, y, self._ball_kernel, self.increment_per_hit_ball)

        # --- Rally state update ---
        self._elapsed_frames += 1

        in_play = self._ball_in_bounds(ball_proj)
        if in_play:
            # ball present & in-bounds → rally is active/continues
            self._rally_active = True
            self._rally_frames += 1
            self._gap_frames = 0
        else:
            if self._rally_active:
                # grace period before we call the rally "over"
                self._gap_frames += 1
                if self._gap_frames >= self._gap_threshold:
                    # finalize rally
                    if self._rally_frames > 0:
                        self._rallies.append(self._rally_frames)
                    self._rally_active = False
                    self._rally_frames = 0
                    self._gap_frames = 0


    # ---------- panel renderers ----------
    def panel_player_heatmap(self, panel_size, bird_reference=None):
        return self._render_heatmap_panel(self.player_heat_accum, panel_size, "Player heatmap", bird_reference, self.heat_alpha_player)

    def panel_ball_heatmap(self, panel_size, bird_reference=None):
        return self._render_heatmap_panel(self.ball_heat_accum, panel_size, "Ball heatmap", bird_reference, self.heat_alpha_ball)

    def panel_kitchen_intrusion(self, projected_players, panel_size):
        """
        Colors the actual court zones via polygons from keypoints:
        - backcourts: darker gray
        - kitchen (midcourt): tinted band
        Also lists who is currently in the kitchen (this frame).
        """
        w, h = panel_size
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # scale from bird canvas -> panel
        sx = w / max(self.canvas_w or 1, 1)
        sy = h / max(self.canvas_h or 1, 1)

        # --- Draw zones if we have them; otherwise fall back to simple band ---
        top_poly    = self._scale_poly(self._zone_polys.get("backcourt_top"), sx, sy)
        kitchen_poly= self._scale_poly(self._zone_polys.get("kitchen"), sx, sy)
        bot_poly    = self._scale_poly(self._zone_polys.get("backcourt_bottom"), sx, sy)

        if top_poly is not None and kitchen_poly is not None and bot_poly is not None:
            cv2.fillPoly(img, [top_poly],     (45, 45, 45))   # darker gray
            cv2.fillPoly(img, [kitchen_poly], (0, 0, 200))  # bluish tint for kitchen
            cv2.fillPoly(img, [bot_poly],     (45, 45, 45))
        else:
            # fallback: scaled mid-band if polygons aren't available yet
            y_kitchen_min = int((300 / 880.0) * (self.canvas_h or 0) * sy)
            y_kitchen_max = int((580 / 880.0) * (self.canvas_h or 0) * sy)
            cv2.rectangle(img, (0, y_kitchen_min), (w, y_kitchen_max), (60, 60, 60), -1)

        # --- Draw players & labels; mark who is in kitchen for this frame ---
        current_in = []
        for i, pt in enumerate(projected_players or []):
            x, y = pt
            px, py = int(x * sx), int(y * sy)
            in_k = self._in_kitchen(pt)
            if in_k:
                current_in.append(i)
            color = (0, 0, 255) if in_k else (0, 200, 0)
            cv2.circle(img, (px, py), 6, color, -1)
            cv2.putText(img, f"P{i}", (px + 8, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        # Header + who is in kitchen now
        cv2.putText(img, "Kitchen (midcourt)",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230), 2, cv2.LINE_AA)

        if current_in:
            cv2.putText(img, f"In kitchen: {', '.join(f'P{pid}' for pid in current_in)}",
                        (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,200,200), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, "In kitchen: none",
                        (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160,160,160), 2, cv2.LINE_AA)

        return img


    def panel_rally_tempo(self, panel_size):
        """
        Rally Length & Tempo Tracker.
        Shows current rally time, average rally time, longest rally, and rallies/minute.
        """
        w, h = panel_size
        img = np.zeros((h, w, 3), dtype=np.uint8)

        fps = float(self._fps or 30)
        # Compose stats
        # Include the current (active) rally for "current" only; history for avg/max.
        current_s = self._rally_frames / fps
        hist_frames = self._rallies[:]  # copy
        avg_s = (np.mean(hist_frames) / fps) if hist_frames else 0.0
        max_s = (np.max(hist_frames) / fps) if hist_frames else 0.0

        elapsed_min = (self._elapsed_frames / fps) / 60.0
        tempo = (len(hist_frames) / elapsed_min) if elapsed_min > 1e-6 else 0.0

        # Header
        cv2.putText(img, "Rally length & tempo",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230,230,230), 2, cv2.LINE_AA)

        # Current rally box
        box_t = 50
        cv2.rectangle(img, (10, box_t), (w-10, box_t+56), (40,40,40), 2)
        cv2.putText(img, f"Current rally: {current_s:4.1f}s",
                    (20, box_t+38), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (70, 180, 255) if self._rally_active else (160,160,160), 2, cv2.LINE_AA)

        # Bars for avg and max
        bar_left = 10
        bar_right = w - 10
        bar_width = bar_right - bar_left
        base_y = box_t + 56 + 18
        bar_h = 18
        gap = 10

        # Choose a time scale (seconds) for bar normalization: dynamic to what's seen.
        scale_s = max(5.0, max(current_s, avg_s, max_s, 1.0))  # at least 5s
        def draw_bar(label, value_s, row):
            y_top = base_y + row * (bar_h + gap)
            y_bot = y_top + bar_h
            frac = float(np.clip(value_s / scale_s, 0.0, 1.0))
            x_end = bar_left + int(frac * bar_width)
            cv2.rectangle(img, (bar_left, y_top), (bar_right, y_bot), (40,40,40), 2)
            cv2.rectangle(img, (bar_left, y_top), (x_end, y_bot), (70,180,255), -1)
            cv2.putText(img, f"{label}: {value_s:4.1f}s",
                        (bar_left + 6, y_bot - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (230,230,230), 2, cv2.LINE_AA)

        draw_bar("Average rally", avg_s, 0)
        draw_bar("Longest rally", max_s, 1)

        # Tempo row
        tempo_y = base_y + 2 * (bar_h + gap) + 30
        cv2.putText(img, f"Tempo: {tempo:4.1f} rallies/min",
                    (10, tempo_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 2, cv2.LINE_AA)

        # Optional footer: detected court bounds debug
        if self._court_bounds is not None:
            xmin, ymin, xmax, ymax = self._court_bounds
            cv2.putText(img, "court bounds learned", (10, h-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120,120,120), 1, cv2.LINE_AA)

        return img

    # ---------- helpers ----------
    def _render_heatmap_panel(self, accum, panel_size, title, bird_reference=None, alpha=0.55):
        w, h = panel_size
        out = np.zeros((h, w, 3), dtype=np.uint8)

        if accum is None:
            cv2.putText(out, f"{title} (off)", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 2, cv2.LINE_AA)
            return out

        k = self.blur_kernel if self.blur_kernel % 2 == 1 else self.blur_kernel + 1
        blurred = cv2.GaussianBlur(accum, (k, k), 0) if k > 1 else accum

        heat8 = self._to_uint8_with_auto_contrast(blurred)

        if self.gamma and self.gamma != 1.0:
            f = (heat8.astype(np.float32) / 255.0) ** (1.0 / self.gamma)
            heat8 = np.clip(f * 255.0, 0, 255).astype(np.uint8)

        color = cv2.applyColorMap(heat8, cv2.COLORMAP_JET)
        color = cv2.resize(color, (w, h), interpolation=cv2.INTER_LINEAR)

        if bird_reference is not None:
            bird_resized = cv2.resize(bird_reference, (w, h), interpolation=cv2.INTER_AREA)
            blended = cv2.addWeighted(bird_resized, 1 - alpha, color, alpha, 0)
            cv2.putText(blended, title, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230), 2, cv2.LINE_AA)
            return blended

        cv2.putText(color, title, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230), 2, cv2.LINE_AA)
        return color

    def _to_uint8_with_auto_contrast(self, arr: np.ndarray) -> np.ndarray:
        nz = arr[arr > 0]
        if nz.size == 0:
            return np.zeros_like(arr, dtype=np.uint8)
        lo_p, hi_p = self.clip_percentiles
        lo = np.percentile(nz, lo_p)
        hi = np.percentile(nz, hi_p)
        if hi <= lo:
            hi, lo = nz.max(), nz.min()
        arr_clip = np.clip(arr, lo, hi)
        norm = (arr_clip - lo) / max(hi - lo, 1e-6)
        return np.clip(norm * 255.0, 0, 255).astype(np.uint8)

    def _make_gaussian_kernel(self, radius: int, sigma: float) -> np.ndarray:
        size = 2 * radius + 1
        ax = np.arange(-radius, radius + 1, dtype=np.float32)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        kernel /= kernel.sum() + 1e-12
        return kernel.astype(np.float32)

    def _stamp(self, accum: np.ndarray, x: int, y: int, kernel: np.ndarray, weight: float):
        if not (0 <= x < (self.canvas_w or 0) and 0 <= y < (self.canvas_h or 0)):
            return
        r = (kernel.shape[0] - 1) // 2
        x0 = max(x - r, 0); y0 = max(y - r, 0)
        x1 = min(x + r + 1, self.canvas_w); y1 = min(y + r + 1, self.canvas_h)
        kx0 = r - (x - x0); ky0 = r - (y - y0)
        kx1 = kx0 + (x1 - x0); ky1 = ky0 + (y1 - y0)
        accum[y0:y1, x0:x1] += kernel[ky0:ky1, kx0:kx1] * weight

    # ---------- zones ----------
    def _get_zone(self, pt):
        """
        Return 'backcourt_top' | 'kitchen' | 'backcourt_bottom' | None
        using dynamic Y boundaries from keypoints if available; otherwise fallback.
        """
        if not (self.canvas_w and self.canvas_h):
            return None
        _, y = pt

        y_min, y_max = self._current_kitchen_bounds_with_fallback()

        if y <= y_min:
            return "backcourt_top"
        if y_min < y <= y_max:
            return "kitchen"
        if y > y_max:
            return "backcourt_bottom"
        return None

    def _in_kitchen(self, pt):
        """True iff point lies in the (smoothed) kitchen band; uses small hysteresis."""
        if not (self.canvas_w and self.canvas_h):
            return False
        _, y = pt

        y_min, y_max = self._current_kitchen_bounds_with_fallback()
        y_eps = 0.01 * self.canvas_h  # ~1% hysteresis
        return (y_min - y_eps) <= y <= (y_max + y_eps)
    
    def _current_kitchen_bounds_with_fallback(self):
        """Return (y_min, y_max) for kitchen band in bird coords."""
        if self._kitchen_y_min is not None and self._kitchen_y_max is not None:
            return self._kitchen_y_min, self._kitchen_y_max
        # Fallback to old proportional thresholds until keypoints arrive
        y_kitchen_min = 300 / 880.0 * (self.canvas_h or 0)
        y_kitchen_max = 580 / 880.0 * (self.canvas_h or 0)
        return y_kitchen_min, y_kitchen_max

    def update_kitchen_from_keypoints(self, proj_kpts: np.ndarray):
        """
        proj_kpts: shape (12,2) or (N,2) in bird space. Assumes 4 rows x 3 cols:
        rows: [0..2], [3..5], [6..8], [9..11]
        Defines the kitchen band as the Y region between row1 and row2.
        Applies exponential moving average for stability.
        """
        if proj_kpts is None:
            return
        pts = np.asarray(proj_kpts, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 12:
            return

        # Average Y for each row
        row0_y = float(np.mean(pts[0:3, 1]))
        row1_y = float(np.mean(pts[3:6, 1]))
        row2_y = float(np.mean(pts[6:9, 1]))
        row3_y = float(np.mean(pts[9:12, 1]))

        # The kitchen is the mid band between row1 and row2
        y_min_new = min(row1_y, row2_y)
        y_max_new = max(row1_y, row2_y)

        # Initialize or EMA-smooth
        if self._kitchen_y_min is None or self._kitchen_y_max is None:
            self._kitchen_y_min = y_min_new
            self._kitchen_y_max = y_max_new
        else:
            a = self._kitchen_ema_alpha
            self._kitchen_y_min = (1 - a) * self._kitchen_y_min + a * y_min_new
            self._kitchen_y_max = (1 - a) * self._kitchen_y_max + a * y_max_new

    def update_court_bounds_from_keypoints(self, proj_kpts):
        """
        Update/smooth the court bounding box (bird coords) from projected keypoints.
        Uses an EMA to avoid jitter. Expects shape (N,2), typically N=12.
        """
        if proj_kpts is None:
            return
        pts = np.asarray(proj_kpts, dtype=np.float32).reshape(-1, 2)
        if pts.size == 0:
            return

        xmin, ymin = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
        xmax, ymax = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))

        if self._court_bounds is None:
            self._court_bounds = (xmin, ymin, xmax, ymax)
        else:
            ax = self._bounds_alpha
            pxmin, pymin, pxmax, pymax = self._court_bounds
            self._court_bounds = (
                (1-ax)*pxmin + ax*xmin,
                (1-ax)*pymin + ax*ymin,
                (1-ax)*pxmax + ax*xmax,
                (1-ax)*pymax + ax*ymax,
            )
    def _ball_in_bounds(self, ball_proj):
        if ball_proj is None or self._court_bounds is None:
            return False
        x, y = ball_proj
        xmin, ymin, xmax, ymax = self._court_bounds
        # small tolerance to avoid bouncing on the edge
        tol = 2.0
        return (xmin - tol) <= x <= (xmax + tol) and (ymin - tol) <= y <= (ymax + tol)
     
    def update_zones_from_keypoints(self, proj_kpts):
        """
        Build three quad polygons (top backcourt, kitchen/midcourt, bottom backcourt)
        from the projected 12 keypoints (assumed 4 rows x 3 cols, row-major order).
        Polys are in bird coords and updated every frame.
        """
        if proj_kpts is None:
            return
        pts = np.asarray(proj_kpts, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 12:
            return

        # Rows (top->bottom), cols (left, mid, right)
        # row i indices: i*3 + [0,1,2]
        def row_lr(i):
            left = pts[i*3 + 0]
            right = pts[i*3 + 2]
            return left, right

        # Bands between rows:
        #   backcourt_top   = between row0 and row1
        #   kitchen (mid)   = between row1 and row2
        #   backcourt_bottom= between row2 and row3
        l0, r0 = row_lr(0)
        l1, r1 = row_lr(1)
        l2, r2 = row_lr(2)
        l3, r3 = row_lr(3)

        # Each band is a quad: [top-left, top-right, bot-right, bot-left]
        self._zone_polys["backcourt_top"]    = np.array([l0, r0, r1, l1], dtype=np.float32)
        self._zone_polys["kitchen"]          = np.array([l1, r1, r2, l2], dtype=np.float32)
        self._zone_polys["backcourt_bottom"] = np.array([l2, r2, r3, l3], dtype=np.float32)

    def _scale_poly(self, poly, sx, sy):
        if poly is None:
            return None
        out = poly.copy()
        out[:, 0] *= sx
        out[:, 1] *= sy
        return out.astype(np.int32)

    # ---------- outputs ----------
    def save_outputs(self):
        # No persistence right now
        pass