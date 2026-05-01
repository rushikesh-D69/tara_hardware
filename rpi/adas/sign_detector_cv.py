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
        self.blue_low = np.array(config.SIGN_BLUE_LOW)
        self.blue_high = np.array(config.SIGN_BLUE_HIGH)
        
        # White HSV range (for the arrow inside the blue circle)
        self.white_low = np.array(config.SIGN_WHITE_LOW)
        self.white_high = np.array(config.SIGN_WHITE_HIGH)

    def detect(self, frame):
        """
        Detects blue circular signs and returns 'LEFT', 'RIGHT', or None.
        Uses centroid-based direction analysis for the internal white arrow.
        """
        if frame is None: return None
        
        # 1. Downscale for speed
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (320, 240))
        
        # 2. Color filtering for Blue
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, self.blue_low, self.blue_high)
        
        # 3. Morphology to clean up noise and close gaps
        kernel = np.ones((5, 5), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        blue_mask = cv2.dilate(blue_mask, kernel, iterations=1) # Close gaps from reflections
        
        # 4. Find circular contours
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.cfg.SIGN_MIN_AREA: continue # Skip small blobs
            
            # Check circularity: 4*pi*Area / Perimeter^2
            peri = cv2.arcLength(cnt, True)
            if peri == 0: continue
            circularity = 4 * np.pi * area / (peri * peri)
            
            # Relaxed circularity check to handle tilted signs
            if circularity > self.cfg.SIGN_CIRCULARITY_THRESHOLD: 
                # 5. Extract the region of interest (ROI)
                x, y, bw, bh = cv2.boundingRect(cnt)
                roi = small[y:y+bh, x:x+bw]
                
                # 6. Analyze the white pixels (the arrow) inside the blue circle
                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                white_mask = cv2.inRange(roi_hsv, self.white_low, self.white_high)
                
                # ── Centroid-based Direction Analysis ──────────────────────────
                # Find the center of mass of the white pixels
                M = cv2.moments(white_mask)
                if M["m00"] > 50: # Minimum white mass to be an arrow
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    # The arrow "head" or the way it points usually shifts the 
                    # centroid away from the geometric center.
                    center_x = bw / 2
                    
                    # If centroid is significantly left or right of center
                    offset_x = (cX - center_x) / bw # Normalized offset (-0.5 to 0.5)
                    
                    if offset_x > 0.05: # Centroid is on the right
                        log.info(f"Sign Detected: RIGHT (centroid offset: {offset_x:.2f})")
                        return "RIGHT"
                    elif offset_x < -0.05: # Centroid is on the left
                        log.info(f"Sign Detected: LEFT (centroid offset: {offset_x:.2f})")
                        return "LEFT"
                    
        return None
