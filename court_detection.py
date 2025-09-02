"""
Court detection & homography module

Purpose
-------
Detects 12 court keypoints with YOLO, computes a homography into a canonical
bird’s-eye plane, and provides utilities to draw/project court geometry.

What it does
------------
- get_keypoints_and_homography(frame): YOLO keypoints → H via RANSAC to fixed dst layout
- project_points(points, H): project arbitrary (x, y) points to bird space
- draw_court_overlay(frame, keypoints): annotate original frame with points/edges
- draw_homography_overlay(canvas, projected_kpts): draw court in bird space

Key data
--------
- dst_pts: canonical 12-point layout (4 rows × 3 columns) defining the target plane
- connect_pairs: edges for visualizing the court grid

Inputs
------
- BGR frame (OpenCV)

Outputs
-------
- (src_pts, H) where src_pts are detected image-space keypoints, H is 3×3 homography

Assumptions
-----------
- Model returns exactly 12 ordered keypoints matching the canonical dst_pts order
"""


from ultralytics import YOLO
import cv2
import numpy as np

class CourtDetector:
    def __init__(self, model_path, homography_size = (400,900)):
        self.model = YOLO(model_path)
        self.birdseye_size = homography_size
        self.dst_pts = self.get_reference_points()
        self.connect_pairs = self.get_connect_pairs()

    def get_reference_points(self):
        return np.array([
            [0,0],  # top left
            [200, 0], # top middle
            [400, 0], # top right
            [0, 300], # top left kitchen
            [200, 300], #top middle kitchen
            [400, 400], #top right kitchen
            [0, 580], # botttom left kitchen
            [200, 580], #bottom middle kitchen
            [400, 580], #bottom right kitchen
            [0, 880], #bottom left
            [200, 880], #bottom midle
            [400, 880] #bottom right
        ], dtype=np.float32)

    def get_connect_pairs(self):
        return [
            (0, 1), (1, 2),  # top row
            (3, 4), (4, 5),  # 2nd row
            (6, 7), (7, 8),  # 3rd row
            (9,10), (10,11), # bottom row
            (0, 3), (3, 6), (6, 9),  # left column
            (1, 4), (4, 7), (7,10),  # center column
            (2, 5), (5, 8), (8,11)   # right column
        ]

    def get_keypoints_and_homography(self, frame):
        """Detects court keypoints and computes the homography matrix"""
        results = self.model(frame, conf=0.9)
        keypoints = results[0].keypoints

        if keypoints is None or keypoints.shape[1] != len(self.dst_pts):
            return None, None

        src_pts = keypoints.xy[0].cpu().numpy().astype(np.float32)
        H, _ = cv2.findHomography(src_pts, self.dst_pts, method=cv2.RANSAC)
        return src_pts, H

    def draw_court_overlay(self, frame, keypoints):
        """Draw keypoints and connections on original frame"""
        for pt in keypoints:
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        for i, j in self.connect_pairs:
            pt1 = tuple(np.round(keypoints[i]).astype(int))
            pt2 = tuple(np.round(keypoints[j]).astype(int))
            cv2.line(frame, pt1, pt2, (255, 255, 255), 2)

    def draw_homography_overlay(self, canvas, projected_kpts):
        """Draws the projected court keypoints on bird’s-eye canvas"""
        for pt in projected_kpts:
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(canvas, (x, y), 5, (0, 255, 0), -1)
        for i, j in self.connect_pairs:
            pt1 = tuple(np.round(projected_kpts[i]).astype(int))
            pt2 = tuple(np.round(projected_kpts[j]).astype(int))
            cv2.line(canvas, pt1, pt2, (255, 255, 255), 2)

    def project_points(self, points, H):
        """Projects a list of (x, y) points using the homography H"""
        points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(points, H)
        return projected.reshape(-1, 2)