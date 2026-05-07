"""
TARA ADAS — OpenCV Sign Detector (Directional Signs)
Detects blue circular directional signs without ML.
Uses HSV colour filtering + white-arrow centroid analysis.
"""
import cv2
import numpy as np
import logging

log = logging.getLogger("TARA.SignDetectorCV")


class SignDetectorCV:
    """
    OpenCV-based detector for blue circular directional signs.
    Returns 'LEFT', 'RIGHT', or None based on internal arrow centroid offset.
    """

    def __init__(self, config):
        self.cfg        = config
        self.blue_low   = np.array(config.SIGN_BLUE_LOW)
        self.blue_high  = np.array(config.SIGN_BLUE_HIGH)
        self.white_low  = np.array(config.SIGN_WHITE_LOW)
        self.white_high = np.array(config.SIGN_WHITE_HIGH)

    def detect(self, frame):
        """
        Detect directional sign and return 'LEFT', 'RIGHT', or None.

        Args:
            frame: BGR image (ideally bird's-eye view from LaneDetector.warp_bev)

        Returns:
            'LEFT' | 'RIGHT' | None
        """
        if frame is None:
            return None

        small = cv2.resize(frame, (320, 240))
        hsv   = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        blue_mask = cv2.inRange(hsv, self.blue_low, self.blue_high)

        kernel    = np.ones((5, 5), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        blue_mask = cv2.dilate(blue_mask, kernel, iterations=1)

        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.cfg.SIGN_MIN_AREA:
                continue

            peri = cv2.arcLength(cnt, True)
            if peri == 0:
                continue

            circularity = 4 * np.pi * area / (peri * peri)
            if circularity <= self.cfg.SIGN_CIRCULARITY_THRESHOLD:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            roi          = small[y:y + bh, x:x + bw]
            roi_hsv      = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            white_mask   = cv2.inRange(roi_hsv, self.white_low, self.white_high)

            M = cv2.moments(white_mask)
            if M["m00"] > 50:
                cX       = int(M["m10"] / M["m00"])
                offset_x = (cX - bw / 2) / bw

                if offset_x > 0.05:
                    log.info(f"Sign: RIGHT (offset={offset_x:.2f})")
                    return "RIGHT"
                elif offset_x < -0.05:
                    log.info(f"Sign: LEFT (offset={offset_x:.2f})")
                    return "LEFT"

        return None
