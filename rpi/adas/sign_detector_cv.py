import cv2
import numpy as np
import logging

log = logging.getLogger("TARA.SignDetectorCV")

class SignDetectorCV:
    """
    OpenCV-based Traffic Sign Recognition (TSR) for Blue Directional Signs.
    No Machine Learning model required — uses color filtering and centroid analysis.
    """
    def __init__(self, config):
        self.cfg = config
        
        # Blue HSV range — tuned for standard printed blue paper
        # Hue: 100-140 (Blue), Saturation: 100-255, Value: 50-255
        self.blue_low = np.array([100, 100, 50])
        self.blue_high = np.array([140, 255, 255])
        
        # White HSV range (for the arrow inside the blue circle)
        self.white_low = np.array([0, 0, 180])
        self.white_high = np.array([180, 50, 255])

    def detect(self, frame):
        """
        Detects blue circular signs and returns 'LEFT', 'RIGHT', or None.
        """
        if frame is None: return None
        
        # 1. Downscale for speed
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (320, 240))
        
        # 2. Color filtering for Blue
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, self.blue_low, self.blue_high)
        
        # 3. Morphology to clean up noise
        kernel = np.ones((5, 5), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        
        # 4. Find circular contours
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 400: continue # Skip small blobs
            
            # Check circularity: 4*pi*Area / Perimeter^2
            peri = cv2.arcLength(cnt, True)
            if peri == 0: continue
            circularity = 4 * np.pi * area / (peri * peri)
            
            if circularity > 0.6: # Good enough for a circle
                # 5. Extract the region of interest (ROI)
                x, y, bw, bh = cv2.boundingRect(cnt)
                roi = small[y:y+bh, x:x+bw]
                
                # 6. Analyze the white pixels (the arrow) inside the blue circle
                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                white_mask = cv2.inRange(roi_hsv, self.white_low, self.white_high)
                
                # Check pixel count in left vs right half of the ROI
                mid = bw // 2
                left_mass = cv2.countNonZero(white_mask[:, :mid])
                right_mass = cv2.countNonZero(white_mask[:, mid:])
                
                # If one side has significantly more white pixels, it's a turn sign
                # Usually the "head" of the arrow adds more mass to that side
                if right_mass > left_mass * 1.3:
                    log.info(f"Sign Detected: TURN RIGHT (L:{left_mass} R:{right_mass})")
                    return "RIGHT"
                elif left_mass > right_mass * 1.3:
                    log.info(f"Sign Detected: TURN LEFT (L:{left_mass} R:{right_mass})")
                    return "LEFT"
                    
        return None
