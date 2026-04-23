import cv2
import numpy as np
import time
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
    Traffic Light Recognition using CPU-efficient HSV color filtering.
    Designed for real-time inference on Raspberry Pi 4B.
    """
    def __init__(self, config):
        self.cfg = config
        self.enabled = getattr(config, 'TL_ENABLED', True)
        
        # Color Thresholds (HSV)
        self.red_low = getattr(config, 'TL_RED_LOW', (0, 100, 100))
        self.red_high = getattr(config, 'TL_RED_HIGH', (10, 255, 255))
        self.green_low = getattr(config, 'TL_GREEN_LOW', (40, 50, 50))
        self.green_high = getattr(config, 'TL_GREEN_HIGH', (90, 255, 255))
        self.yellow_low = getattr(config, 'TL_YELLOW_LOW', (15, 100, 100))
        self.yellow_high = getattr(config, 'TL_YELLOW_HIGH', (35, 255, 255))
        
        # Minimum pixels to confirm detection
        self.min_pixels = getattr(config, 'TL_MIN_PIXELS', 500)
        
        # State Smoothing (Must see same light for N frames)
        self.history = []
        self.history_len = 3

    def detect(self, frame):
        """
        Detect traffic light state in the 'Sky ROI' of the frame.
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
        
        # ── [3] Apply Color Masks ────────────────────────────────────
        mask_red = cv2.inRange(hsv, self.red_low, self.red_high)
        mask_green = cv2.inRange(hsv, self.green_low, self.green_high)
        mask_yellow = cv2.inRange(hsv, self.yellow_low, self.yellow_high)
        
        # ── [4] Count Pixels ─────────────────────────────────────────
        red_count = cv2.countNonZero(mask_red)
        green_count = cv2.countNonZero(mask_green)
        yellow_count = cv2.countNonZero(mask_yellow)
        
        # ── [5] Determine State ──────────────────────────────────────
        current_state = "UNKNOWN"
        max_count = max(red_count, green_count, yellow_count)
        
        if max_count > self.min_pixels:
            if max_count == red_count:
                current_state = "RED"
            elif max_count == green_count:
                current_state = "GREEN"
            elif max_count == yellow_count:
                current_state = "YELLOW"
        
        # ── [6] Temporal Smoothing ───────────────────────────────────
        self.history.append(current_state)
        if len(self.history) > self.history_len:
            self.history.pop(0)
            
        # Only confirm if last N frames were the same
        if len(self.history) == self.history_len and len(set(self.history)) == 1:
            stable_state = self.history[0]
            if stable_state != "UNKNOWN":
                result.detected = True
                result.state = stable_state
                result.confidence = float(max_count / (roi_h * w))
        
        result.inference_ms = (time.monotonic() - t_start) * 1000
        return result
