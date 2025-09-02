"""
Ball tracking module

Purpose
-------
Runs YOLO ball detection per frame, draws the box on the main view, and projects
the ball center into bird’s-eye coordinates using a homography.

What it does
------------
- detect_frame(frame): YOLO inference → {1: [x1, y1, x2, y2]} if ball found
- interpolate_ball_positions(list_of_dicts): fills gaps across frames (NaN interpolation)
- draw_bbox(frame, bbox): annotate main view
- project_ball(bbox, H): perspectiveTransform of bbox center with homography H
- process_and_project(ball_dict, frame, H): convenience wrapper returning (bbox, proj_point)

Inputs
------
- BGR frame (OpenCV), homography matrix H (3×3)

Outputs
-------
- Ball bbox on the main frame (visual) and (x, y) point in bird coordinates

Assumptions
-----------
- Single ball tracked; dictionary uses key=1 for consistency
- Homography H is valid when projection is requested
"""


import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frame(self, frame):
        """Detect ball in a single frame, return dict with bbox if found"""
        results = self.model.predict(frame, conf=0.15)[0]
        ball_dict = {}
        for box in results.boxes:
            bbox = box.xyxy.tolist()[0]
            ball_dict[1] = bbox  # using key=1 to be consistent
        return ball_dict

    def interpolate_ball_positions(self, ball_positions):
        """
        Given list of dicts per frame, interpolate missing detections (NaNs).
        Returns list of dicts with interpolated bboxes.
        """
        # Extract bbox coords into dataframe, fill missing with NaN
        data = []
        for d in ball_positions:
            if 1 in d:
                data.append(d[1])
            else:
                data.append([np.nan, np.nan, np.nan, np.nan])
        df = pd.DataFrame(data, columns=['x1', 'y1', 'x2', 'y2'])
        df_interp = df.interpolate().bfill().ffill()

        # Reconstruct list of dicts with interpolated coords
        interp_positions = [{1: row.tolist()} for idx, row in df_interp.iterrows()]
        return interp_positions

    def draw_bbox(self, frame, bbox):
        """Draw bounding box and label on frame"""
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, "Ball", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    def project_ball(self, bbox, H):
        """Project the ball center to bird's eye view"""
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        point = np.array([[[cx, cy]]], dtype=np.float32)
        proj = cv2.perspectiveTransform(point, H)[0][0]
        return tuple(proj)

    def process_and_project(self, ball_dict, frame, H):
        """
        Convenience method: draw bbox on frame and return projected point (if available)
        """
        if 1 in ball_dict:
            bbox = ball_dict[1]
            self.draw_bbox(frame, bbox)
            if H is not None:
                proj = self.project_ball(bbox, H)
                return bbox, proj
            return bbox, None
        return None, None
