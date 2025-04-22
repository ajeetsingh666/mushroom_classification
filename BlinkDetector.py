import cv2
import mediapipe as mp
import os
import shutil
import math
import numpy as np

class BlinkDetector:
    def __init__(self):
        self.blink_threshold = 0.04

        # MediaPipe Eye Landmark Indices (for EAR calculation)
        self.left_eye_indices = [33, 160, 158, 133, 153, 144]   # p1, p2, p3, p4, p5, p6
        self.right_eye_indices = [362, 385, 387, 263, 373, 380]

    def _euclidean_distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _extract_eye_points(self, landmarks, indices, w, h):
        return [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]

    def _compute_ear(self, eye_points):
        p1, p2, p3, p4, p5, p6 = eye_points
        vertical_1 = self._euclidean_distance(p2, p6)
        vertical_2 = self._euclidean_distance(p3, p5)
        horizontal = self._euclidean_distance(p1, p4)

        if horizontal == 0:
            return None
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def get_eye_aspect_ratios(self, landmarks, w, h):
        left_eye_pts = self._extract_eye_points(landmarks, self.left_eye_indices, w, h)
        right_eye_pts = self._extract_eye_points(landmarks, self.right_eye_indices, w, h)

        left_ear = self._compute_ear(left_eye_pts)
        right_ear = self._compute_ear(right_eye_pts)

        return left_ear, right_ear

    def is_blinking(self, landmarks, w, h):
        left_ear, right_ear = self.get_eye_aspect_ratios(landmarks, w, h)

        if left_ear is None or right_ear is None:
            return None, None, (None, None)

        avg_ear = (left_ear + right_ear) / 2.0
        blinking = avg_ear < self.blink_threshold

        return blinking, avg_ear, (left_ear, right_ear)
