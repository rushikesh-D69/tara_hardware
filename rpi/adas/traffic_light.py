"""
TARA ADAS — Traffic Light Recognition (TLR)
CPU-efficient HSV colour filtering with shape validation.

Pipeline:
  1. Crop top 30% of frame (sky ROI)
  2. Convert to HSV
  3. Apply colour masks (red×2, green, yellow) with circularity validation
  4. Pick strongest detection
  5. Temporal smoothing over 2 consecutive frames
"""
import cv2
import numpy as np
import time
from collections import deque
from utils.logger import get_logger

log = get_logger("TLR")


class TrafficLightResult:
    """Detection result for Traffic Light Recognition."""
    def __init__(self):
        self.detected     = False
        self.state        = "UNKNOWN"
        self.confidence   = 0.0
        self.inference_ms = 0.0


class TrafficLightDetector:
    """
    Traffic Light Recognition using HSV colour filtering and
    circularity-based shape validation for false-positive reduction.
    """

    def __init__(self, config):
        self.cfg     = config
        self.enabled = getattr(config, 'TL_ENABLED', True)

        self.red_low_1  = getattr(config, 'TL_RED_LOW',    (0,   100, 100))
        self.red_high_1 = getattr(config, 'TL_RED_HIGH',   (10,  255, 255))
        self.red_low_2  = getattr(config, 'TL_RED_LOW_2',  (170, 100, 100))
        self.red_high_2 = getattr(config, 'TL_RED_HIGH_2', (180, 255, 255))

        self.green_low   = getattr(config, 'TL_GREEN_LOW',   (40, 50,  50))
        self.green_high  = getattr(config, 'TL_GREEN_HIGH',  (90, 255, 255))
        self.yellow_low  = getattr(config, 'TL_YELLOW_LOW',  (15, 100, 100))
        self.yellow_high = getattr(config, 'TL_YELLOW_HIGH', (35, 255, 255))

        self.min_pixels      = getattr(config, 'TL_MIN_PIXELS',       800)
        self.min_circularity = getattr(config, 'TL_MIN_CIRCULARITY',  0.5)
        self.min_contour_area = getattr(config, 'TL_MIN_CONTOUR_AREA', 200)

        self.history = deque(maxlen=2)

    def detect(self, frame):
        """
        Detect traffic light state in the upper 30% (sky ROI) of the frame.

        Returns:
            TrafficLightResult
        """
        result = TrafficLightResult()
        if not self.enabled:
            return result

        t_start = time.monotonic()

        h, w    = frame.shape[:2]
        sky_roi = frame[0:int(h * 0.3), 0:w]
        hsv     = cv2.cvtColor(sky_roi, cv2.COLOR_BGR2HSV)

        mask_red_1 = cv2.inRange(hsv, np.array(self.red_low_1), np.array(self.red_high_1))
        mask_red_2 = cv2.inRange(hsv, np.array(self.red_low_2), np.array(self.red_high_2))
        mask_red   = cv2.bitwise_or(mask_red_1, mask_red_2)

        mask_green  = cv2.inRange(hsv, np.array(self.green_low),  np.array(self.green_high))
        mask_yellow = cv2.inRange(hsv, np.array(self.yellow_low), np.array(self.yellow_high))

        red_score    = self._validate_light(mask_red)
        green_score  = self._validate_light(mask_green)
        yellow_score = self._validate_light(mask_yellow)

        current_state = "UNKNOWN"
        max_score     = max(red_score, green_score, yellow_score)

        if max_score > 0:
            if max_score == red_score:
                current_state = "RED"
            elif max_score == green_score:
                current_state = "GREEN"
            else:
                current_state = "YELLOW"

        self.history.append(current_state)

        if len(self.history) >= 2 and len(set(self.history)) == 1:
            stable_state = self.history[0]
            if stable_state != "UNKNOWN":
                result.detected   = True
                result.state      = stable_state
                result.confidence = max_score

        result.inference_ms = (time.monotonic() - t_start) * 1000
        return result

    def _validate_light(self, mask):
        """
        Validate a colour mask by checking for circular blobs.

        Returns a confidence score (0.0 = no valid light, >0 = detected).
        Eliminates false positives from non-circular coloured objects.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_score  = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_contour_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1:
                continue

            circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
            if circularity >= self.min_circularity:
                score = area / 1000.0
                if score > best_score:
                    best_score = score

        return best_score
