"""
TARA ADAS — Traffic Light Recognition (TLR)
CPU-efficient HSV color filtering with shape validation.

Improvements over pure pixel-count approach:
  1. Full red hue range (wraps around 0/180 in HSV)
  2. Circularity check — filters out non-circular colored objects
  3. Minimum contour area — ignores tiny reflections
  4. Faster temporal smoothing (2 frames instead of 3)
"""
import cv2
import numpy as np
import time
from collections import deque
from utils.logger import get_logger

log = get_logger("TLR")


class TrafficLightResult:
    """Detection result for Traffic Light Recognition (TLR)."""
    def __init__(self):
        self.detected = False
        self.state = "UNKNOWN"  # RED, YELLOW, GREEN, UNKNOWN
        self.confidence = 0.0
        self.inference_ms = 0.0


class TrafficLightDetector:
    """
    Traffic Light Recognition using CPU-efficient HSV color filtering
    with shape validation for false-positive reduction.
    Designed for real-time inference on Raspberry Pi 4B.
    """
    def __init__(self, config):
        self.cfg = config
        self.enabled = getattr(config, 'TL_ENABLED', True)

        # Color Thresholds (HSV)
        # Red wraps around H=0/180 — need TWO ranges
        self.red_low_1 = getattr(config, 'TL_RED_LOW', (0, 100, 100))
        self.red_high_1 = getattr(config, 'TL_RED_HIGH', (10, 255, 255))
        self.red_low_2 = getattr(config, 'TL_RED_LOW_2', (170, 100, 100))
        self.red_high_2 = getattr(config, 'TL_RED_HIGH_2', (180, 255, 255))

        self.green_low = getattr(config, 'TL_GREEN_LOW', (40, 50, 50))
        self.green_high = getattr(config, 'TL_GREEN_HIGH', (90, 255, 255))
        self.yellow_low = getattr(config, 'TL_YELLOW_LOW', (15, 100, 100))
        self.yellow_high = getattr(config, 'TL_YELLOW_HIGH', (35, 255, 255))

        # Minimum pixels to confirm detection
        self.min_pixels = getattr(config, 'TL_MIN_PIXELS', 800)

        # Circularity threshold (0.0–1.0, 1.0 = perfect circle)
        self.min_circularity = getattr(config, 'TL_MIN_CIRCULARITY', 0.5)

        # Minimum contour area (in pixels)
        self.min_contour_area = getattr(config, 'TL_MIN_CONTOUR_AREA', 200)

        # State Smoothing (reduced from 3 to 2 for faster response)
        self.history = deque(maxlen=2)

    def detect(self, frame):
        """
        Detect traffic light state in the 'Sky ROI' of the frame.

        Pipeline:
          1. Crop top 30% (sky region where lights hang)
          2. Convert to HSV
          3. Color mask (red, green, yellow) with shape validation
          4. Pick strongest detection
          5. Temporal smoothing (2 frames)
        """
        result = TrafficLightResult()
        if not self.enabled:
            return result

        t_start = time.monotonic()

        # ── [1] ROI Crop (Top 30% of the frame) ──────────────────────
        h, w = frame.shape[:2]
        roi_h = int(h * 0.3)
        sky_roi = frame[0:roi_h, 0:w]

        # ── [2] Convert to HSV ───────────────────────────────────────
        hsv = cv2.cvtColor(sky_roi, cv2.COLOR_BGR2HSV)

        # ── [3] Apply Color Masks with shape validation ───────────────
        # Red has TWO ranges (hue wraps around 0/180)
        mask_red_1 = cv2.inRange(hsv, np.array(self.red_low_1), np.array(self.red_high_1))
        mask_red_2 = cv2.inRange(hsv, np.array(self.red_low_2), np.array(self.red_high_2))
        mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)

        mask_green = cv2.inRange(hsv, np.array(self.green_low), np.array(self.green_high))
        mask_yellow = cv2.inRange(hsv, np.array(self.yellow_low), np.array(self.yellow_high))

        # ── [4] Validate each color with shape analysis ──────────────
        red_score = self._validate_light(mask_red)
        green_score = self._validate_light(mask_green)
        yellow_score = self._validate_light(mask_yellow)

        # ── [5] Determine State ──────────────────────────────────────
        current_state = "UNKNOWN"
        max_score = max(red_score, green_score, yellow_score)

        if max_score > 0:
            if max_score == red_score:
                current_state = "RED"
            elif max_score == green_score:
                current_state = "GREEN"
            elif max_score == yellow_score:
                current_state = "YELLOW"

        # ── [6] Temporal Smoothing ───────────────────────────────────
        self.history.append(current_state)

        # Confirm if last 2 frames agree (faster than requiring 3)
        if len(self.history) >= 2 and len(set(self.history)) == 1:
            stable_state = self.history[0]
            if stable_state != "UNKNOWN":
                result.detected = True
                result.state = stable_state
                result.confidence = max_score

        result.inference_ms = (time.monotonic() - t_start) * 1000
        return result

    def _validate_light(self, mask):
        """
        Validate a color mask by checking for circular blobs.
        Returns a confidence score (0.0 = no valid light, >0 = detected).

        This eliminates false positives from non-circular colored objects
        (walls, decorations, clothing, etc.).

        Args:
            mask: Binary color mask

        Returns:
            float confidence score (0.0 if no valid detection)
        """
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_score = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_contour_area:
                continue

            # Compute circularity: 4π * area / perimeter²
            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1:
                continue

            circularity = (4.0 * np.pi * area) / (perimeter * perimeter)

            if circularity >= self.min_circularity:
                # Score based on area — larger = more confident
                score = area / 1000.0  # normalize
                if score > best_score:
                    best_score = score

        return best_score
