"""
TARA ADAS — Lane Detection Module (LDW + LKA)
Pure OpenCV classical computer vision pipeline.
No ML model needed — saves compute budget for other features.

Optimized for INDOOR TRACK:
  - Black chart paper floor
  - White insulation tape lane markings
  - Short lanes, sharp turns
  - Low-mounted, downward-tilted camera
  - Indoor lighting with shadows/reflections

Pipeline:
  1. Resize → 2. Grayscale → 3. Adaptive threshold (white on black)
  → 4. Morphological cleanup → 5. Bird's-eye warp
  → 6. Sliding window lane search → 7. Polynomial fit
  → 8. Compute offset → 9. Inverse-warp for debug

Outputs (via LaneDetectionResult):
  - lane_center_offset: float (pixels, + = drifting right)
  - departure_warning: bool
  - steering_correction: float (-1.0 to 1.0)
      Consumed by DecisionManager which routes it through
      serial_bridge to the ESP32.
  - lane_detected: bool
"""
import cv2
import numpy as np
from collections import deque
from utils.logger import get_logger

log = get_logger("LaneDet")


class LaneDetectionResult:
    """Container for lane detection outputs."""

    def __init__(self):
        self.lane_detected = False
        self.left_lane = None        # (x1, y1, x2, y2)
        self.right_lane = None       # (x1, y1, x2, y2)
        self.lane_center_offset = 0.0  # pixels from frame center
        self.departure_warning = False
        self.steering_correction = 0.0  # -1.0 (left) to 1.0 (right)
        self.confidence = 0.0        # 0.0–1.0, based on pixel count
        self.debug_frame = None      # Visualization overlay


class LaneDetector:
    """
    Lane detection using classical computer vision.
    Optimized for indoor track with black floor + white tape lanes.
    Designed to run at 30+ FPS on Raspberry Pi 4B.

    Steering output is a normalized proportional value (-1.0 to 1.0)
    sent directly to the ESP32 as jd.x. The ESP32 handles all closed-loop
    motor control — no PID runs here.
    """

    def __init__(self, config):
        """
        Args:
            config: Config module with lane detection parameters
        """
        self.cfg = config
        self.proc_w = config.PROC_WIDTH
        self.proc_h = config.PROC_HEIGHT

        # Precompute perspective transform matrices
        self._M = None
        self._M_inv = None
        self._compute_perspective_transform()

        # Sliding window parameters
        self._n_windows = 12           # number of sliding windows
        self._window_margin = 70       # half-width of each window (pixels)
        self._min_pix_recenter = 30    # minimum pixels to recenter window

        # Lane smoothing — deques for O(1) popleft
        self._left_fit_history = deque(maxlen=5)
        self._right_fit_history = deque(maxlen=5)
        self._left_base_history = deque(maxlen=8)
        self._right_base_history = deque(maxlen=8)

        # Steering deadband (pixels)
        self._steering_deadband = getattr(config, 'LANE_STEERING_DEADBAND', 5)

        # Lane recovery state
        self._frames_since_left = 0
        self._frames_since_right = 0
        self._recovery_mode = False
        self._recovery_threshold = 4   # frames lost before recovery activates

        # Morphological kernel (5×5 handles glossy chart-paper reflections)
        morph_size = getattr(config, 'LANE_MORPH_KERNEL', 5)
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (morph_size, morph_size))

        # Horizontal rejection kernel — removes horizontal structures
        # (transparent tape strips joining chart paper sheets)
        self._horiz_reject_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (15, 1))  # wide horizontal line detector

        # Adaptive threshold parameters
        self._adaptive_block = getattr(config, 'LANE_ADAPTIVE_BLOCK_SIZE', 51)
        self._adaptive_c = getattr(config, 'LANE_ADAPTIVE_C', -25)
        self._use_adaptive = getattr(config, 'LANE_USE_ADAPTIVE_THRESH', True)

        log.info(f"LaneDetector initialized: process at {self.proc_w}x{self.proc_h}, "
                 f"adaptive_thresh={self._use_adaptive}")

    def _compute_perspective_transform(self):
        """Precompute the bird's-eye view perspective matrices."""
        w, h = self.proc_w, self.proc_h

        src = np.float32([
            [r[0] * w, r[1] * h] for r in self.cfg.BEV_SRC_RATIOS
        ])
        dst = np.float32([
            [r[0] * w, r[1] * h] for r in self.cfg.BEV_DST_RATIOS
        ])

        self._M = cv2.getPerspectiveTransform(src, dst)
        self._M_inv = cv2.getPerspectiveTransform(dst, src)

    def detect(self, frame, debug=False):
        """
        Run the full lane detection pipeline on a single frame.

        Args:
            frame: BGR image from camera (640x480)
            debug: If True, generate debug visualization

        Returns:
            LaneDetectionResult with all outputs.
        """
        result = LaneDetectionResult()

        # Step 1: Resize for faster processing
        small = cv2.resize(frame, (self.proc_w, self.proc_h))

        # Step 2: Create binary mask of lane markings
        lane_mask = self._create_lane_mask(small)

        # Step 3: Warp to bird's-eye view
        bev_mask = cv2.warpPerspective(
            lane_mask, self._M, (self.proc_w, self.proc_h))

        # Step 4: Morphological cleanup on BEV
        # Close: fill small gaps in tape lines
        bev_mask = cv2.morphologyEx(bev_mask, cv2.MORPH_CLOSE, self._morph_kernel)
        # Open: remove small noise spots (reflections, debris)
        bev_mask = cv2.morphologyEx(bev_mask, cv2.MORPH_OPEN, self._morph_kernel)

        # Remove horizontal structures (transparent tape joining chart sheets).
        # Detect horizontal lines, then subtract them from the mask.
        horiz_lines = cv2.morphologyEx(bev_mask, cv2.MORPH_OPEN, self._horiz_reject_kernel)
        bev_mask = cv2.subtract(bev_mask, horiz_lines)

        # Mask top 10% of BEV — distant region is noisy after warp
        sky_cutoff = int(self.proc_h * 0.10)
        bev_mask[:sky_cutoff, :] = 0

        # Step 5: Find lane bases using histogram
        left_base, right_base = self._find_lane_bases(bev_mask)

        # Step 6: Sliding window lane search
        left_fit, right_fit, left_pixels, right_pixels = self._sliding_window_search(
            bev_mask, left_base, right_base)

        # Step 7: Compute lane positions and offset
        result = self._compute_steering(
            left_fit, right_fit, left_pixels, right_pixels, result)

        # Step 8: Compute camera-space lane lines for debug overlay
        if left_fit is not None:
            result.left_lane = self._fit_to_camera_line(left_fit)
        if right_fit is not None:
            result.right_lane = self._fit_to_camera_line(right_fit)

        # Debug visualization
        if debug:
            result.debug_frame = self._draw_debug(small, result, bev_mask, left_fit, right_fit)

        return result

    def _create_lane_mask(self, frame):
        """
        Create a binary mask of lane markings optimized for
        white tape on black chart paper floor.

        Primary:  Adaptive threshold (robust to uneven indoor lighting).
        Recovery: Merges adaptive result with a LAB-space white detector.
                  LAB is perceptually uniform — white tape has consistently
                  high L regardless of floor color or lighting temperature.
                  Borrowed from TurboPi's LineFollower approach.

        Args:
            frame: BGR image

        Returns:
            Binary mask (uint8, 0 or 255)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._use_adaptive:
            # Primary: adaptive threshold — locally bright (white tape) on dark floor
            mask = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                self._adaptive_block,
                self._adaptive_c,
            )

            # Recovery mode: merge adaptive result with LAB-space white detection
            if self._recovery_mode:
                # More permissive adaptive pass
                mask_adaptive_loose = cv2.adaptiveThreshold(
                    gray, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    self._adaptive_block,
                    self._adaptive_c + 10,   # more permissive
                )

                # LAB white detection — L > 180 (bright), a near 128, b near 128
                # OpenCV LAB: L in [0,255], a and b in [0,255] (128=neutral)
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                mask_lab = cv2.inRange(
                    lab,
                    (170,  110, 110),   # L_min, a_min, b_min
                    (255,  145, 145),   # L_max, a_max, b_max
                )
                # Morphological cleanup on LAB mask — remove noise spots
                mask_lab = cv2.morphologyEx(
                    mask_lab, cv2.MORPH_OPEN,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                )

                # Merge all three passes — any vote counts
                mask = cv2.bitwise_or(mask, mask_adaptive_loose)
                mask = cv2.bitwise_or(mask, mask_lab)
        else:
            # Fallback: HSV-based white detection (config-driven)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(
                hsv,
                np.array(self.cfg.LANE_WHITE_HSV_LOW),
                np.array(self.cfg.LANE_WHITE_HSV_HIGH),
            )

        # Light Gaussian blur to merge nearby detections
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        return mask



    def _find_lane_bases(self, bev_mask):
        """
        Find the starting x-position of left and right lanes using
        a histogram of the bottom half of the BEV image.

        Smooths with historical base positions to avoid jumping.

        Args:
            bev_mask: Binary BEV image

        Returns:
            (left_base_x, right_base_x)
        """
        # Histogram of bottom half
        bottom_half = bev_mask[self.proc_h // 2:, :]
        histogram = np.sum(bottom_half, axis=0)

        midpoint = self.proc_w // 2

        # Find peaks in left and right halves
        left_half = histogram[:midpoint]
        right_half = histogram[midpoint:]

        # Require minimum pixel count to avoid noise peaks
        min_peak = 200

        if np.max(left_half) > min_peak:
            left_base = int(np.argmax(left_half))
        else:
            left_base = None

        if np.max(right_half) > min_peak:
            right_base = int(np.argmax(right_half)) + midpoint
        else:
            right_base = None

        # Smooth with history
        if left_base is not None:
            self._left_base_history.append(left_base)
        if right_base is not None:
            self._right_base_history.append(right_base)

        # Use smoothed base if available
        if self._left_base_history:
            left_base = int(np.mean(self._left_base_history))
        elif left_base is None:
            left_base = self.proc_w // 4  # default guess

        if self._right_base_history:
            right_base = int(np.mean(self._right_base_history))
        elif right_base is None:
            right_base = 3 * self.proc_w // 4  # default guess

        return left_base, right_base

    def _sliding_window_search(self, bev_mask, left_base, right_base):
        """
        Sliding window search for lane pixels in BEV space.

        More robust than Hough lines for curves and short lane segments.
        Adapts window position as it moves up the image.

        Args:
            bev_mask: Binary BEV image
            left_base: Starting x for left lane
            right_base: Starting x for right lane

        Returns:
            (left_fit, right_fit, left_pixel_count, right_pixel_count)
            Each fit is a 2nd-order polynomial [a, b, c] or None.
        """
        window_height = (self.proc_h - int(self.proc_h * 0.25)) // self._n_windows
        nonzero = bev_mask.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        left_current = left_base
        right_current = right_base

        left_lane_inds = []
        right_lane_inds = []

        margin = self._window_margin
        # Widen margin in recovery mode
        if self._recovery_mode:
            margin = int(margin * 1.5)

        for window_idx in range(self._n_windows):
            # Window boundaries
            win_y_low = self.proc_h - (window_idx + 1) * window_height
            win_y_high = self.proc_h - window_idx * window_height

            # Left window
            win_xleft_low = max(0, left_current - margin)
            win_xleft_high = min(self.proc_w, left_current + margin)

            # Right window
            win_xright_low = max(0, right_current - margin)
            win_xright_high = min(self.proc_w, right_current + margin)

            # Identify pixels within windows
            good_left = (
                (nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)
            ).nonzero()[0]

            good_right = (
                (nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)
            ).nonzero()[0]

            left_lane_inds.append(good_left)
            right_lane_inds.append(good_right)

            # Recenter windows if enough pixels found
            if len(good_left) > self._min_pix_recenter:
                left_current = int(np.mean(nonzero_x[good_left]))
            if len(good_right) > self._min_pix_recenter:
                right_current = int(np.mean(nonzero_x[good_right]))

        # Concatenate indices
        left_lane_inds = np.concatenate(left_lane_inds) if left_lane_inds else np.array([])
        right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([])

        left_pixel_count = len(left_lane_inds)
        right_pixel_count = len(right_lane_inds)

        # Minimum pixel count for a valid lane
        min_lane_pixels = 150
        if self._recovery_mode:
            min_lane_pixels = 80  # more permissive during recovery

        # Fit polynomials
        left_fit = None
        right_fit = None

        if left_pixel_count > min_lane_pixels:
            left_x = nonzero_x[left_lane_inds]
            left_y = nonzero_y[left_lane_inds]
            try:
                left_fit = np.polyfit(left_y, left_x, 2)
                self._left_fit_history.append(left_fit)
                self._frames_since_left = 0
            except (np.linalg.LinAlgError, ValueError):
                left_fit = None
        else:
            self._frames_since_left += 1

        if right_pixel_count > min_lane_pixels:
            right_x = nonzero_x[right_lane_inds]
            right_y = nonzero_y[right_lane_inds]
            try:
                right_fit = np.polyfit(right_y, right_x, 2)
                self._right_fit_history.append(right_fit)
                self._frames_since_right = 0
            except (np.linalg.LinAlgError, ValueError):
                right_fit = None
        else:
            self._frames_since_right += 1

        # Use averaged historical fit if current frame missed
        if left_fit is None and self._left_fit_history:
            left_fit = np.mean(self._left_fit_history, axis=0)

        if right_fit is None and self._right_fit_history:
            right_fit = np.mean(self._right_fit_history, axis=0)

        # Update recovery mode
        both_lost = (self._frames_since_left >= self._recovery_threshold and
                     self._frames_since_right >= self._recovery_threshold)
        either_lost = (self._frames_since_left >= self._recovery_threshold or
                       self._frames_since_right >= self._recovery_threshold)

        if both_lost:
            if not self._recovery_mode:
                log.warning("Both lanes lost — entering recovery mode")
            self._recovery_mode = True
        elif not either_lost:
            if self._recovery_mode:
                log.info("Lanes recovered — exiting recovery mode")
            self._recovery_mode = False

        return left_fit, right_fit, left_pixel_count, right_pixel_count

    def _compute_steering(self, left_fit, right_fit, left_pixels, right_pixels, result):
        """
        Compute lane center offset and steering correction from polynomial fits.

        Args:
            left_fit: Left lane polynomial coefficients or None
            right_fit: Right lane polynomial coefficients or None
            left_pixels: Number of pixels supporting left lane
            right_pixels: Number of pixels supporting right lane
            result: LaneDetectionResult to populate

        Returns:
            Populated LaneDetectionResult
        """
        eval_y = self.proc_h  # evaluate at bottom of frame
        frame_center_x = self.proc_w / 2

        if left_fit is not None and right_fit is not None:
            # Both lanes detected
            result.lane_detected = True
            left_x = np.polyval(left_fit, eval_y)
            right_x = np.polyval(right_fit, eval_y)

            # Sanity check: right lane should be to the right of left lane
            if right_x <= left_x:
                # Lanes are crossed — likely noise. Use historical data only.
                result.lane_detected = False
                result.confidence = 0.1
                return result

            lane_center_x = (left_x + right_x) / 2
            result.lane_center_offset = lane_center_x - frame_center_x
            result.confidence = min(1.0, (left_pixels + right_pixels) / 1000.0)

        elif left_fit is not None:
            # Only left lane — estimate center
            result.lane_detected = True
            left_x = np.polyval(left_fit, eval_y)
            # Assume lane width is ~55% of BEV frame width
            estimated_lane_width = 0.45 * self.proc_w
            lane_center_x = left_x + estimated_lane_width / 2
            result.lane_center_offset = lane_center_x - frame_center_x
            result.confidence = min(0.7, left_pixels / 500.0)

        elif right_fit is not None:
            # Only right lane — estimate center
            result.lane_detected = True
            right_x = np.polyval(right_fit, eval_y)
            estimated_lane_width = 0.45 * self.proc_w
            lane_center_x = right_x - estimated_lane_width / 2
            result.lane_center_offset = lane_center_x - frame_center_x
            result.confidence = min(0.7, right_pixels / 500.0)

        else:
            # No lanes at all
            result.lane_detected = False
            result.confidence = 0.0
            return result

        # Lane Departure Warning
        if abs(result.lane_center_offset) > self.cfg.LANE_DEPARTURE_THRESHOLD:
            result.departure_warning = True

        # Lane Keeping Assist — proportional steering with deadband
        if abs(result.lane_center_offset) <= self._steering_deadband:
            # Inside deadband — no correction needed (eliminates jitter)
            result.steering_correction = 0.0
        else:
            # Remove deadband from offset before normalizing
            effective_offset = result.lane_center_offset
            if effective_offset > 0:
                effective_offset -= self._steering_deadband
            else:
                effective_offset += self._steering_deadband

            # Normalize to [-1.0, 1.0]
            normalized = effective_offset / (self.proc_w / 2)
            result.steering_correction = max(-1.0, min(1.0, normalized))

        return result

    def _fit_to_camera_line(self, fit):
        """
        Convert a BEV polynomial fit to camera-space line endpoints.

        Args:
            fit: Polynomial coefficients [a, b, c] for x = a*y^2 + b*y + c

        Returns:
            (x1, y1, x2, y2) in camera space, or None
        """
        if fit is None:
            return None

        # Generate two points in BEV space (bottom and top)
        y_bottom = self.proc_h
        y_top = int(self.proc_h * 0.3)

        x_bottom = int(np.polyval(fit, y_bottom))
        x_top = int(np.polyval(fit, y_top))

        # Transform back to camera space
        pts_bev = np.float32([[x_bottom, y_bottom], [x_top, y_top]]).reshape(-1, 1, 2)
        try:
            pts_cam = cv2.perspectiveTransform(pts_bev, self._M_inv)
            cx1, cy1 = pts_cam[0][0]
            cx2, cy2 = pts_cam[1][0]
            return (int(cx1), int(cy1), int(cx2), int(cy2))
        except cv2.error:
            return None

    def _draw_debug(self, frame, result, bev_mask, left_fit, right_fit):
        """
        Draw debug visualization with detected lanes overlay.

        Args:
            frame: Original frame (small)
            result: LaneDetectionResult
            bev_mask: Binary BEV mask
            left_fit: Left lane polynomial or None
            right_fit: Right lane polynomial or None

        Returns:
            BGR debug frame
        """
        debug = frame.copy()

        # Draw lane lines
        if result.left_lane is not None:
            x1, y1, x2, y2 = result.left_lane
            cv2.line(debug, (x1, y1), (x2, y2), (0, 255, 0), 3)

        if result.right_lane is not None:
            x1, y1, x2, y2 = result.right_lane
            cv2.line(debug, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Draw lane area polygon if both lanes visible
        if result.left_lane is not None and result.right_lane is not None:
            pts = np.array([
                [result.left_lane[0], result.left_lane[1]],
                [result.left_lane[2], result.left_lane[3]],
                [result.right_lane[2], result.right_lane[3]],
                [result.right_lane[0], result.right_lane[1]],
            ], dtype=np.int32)
            lane_overlay = debug.copy()
            cv2.fillPoly(lane_overlay, [pts], (0, 80, 0))
            debug = cv2.addWeighted(debug, 0.7, lane_overlay, 0.3, 0)

            # Draw center markers
            lane_center_x = int((result.left_lane[0] + result.right_lane[0]) / 2)
            frame_center_x = self.proc_w // 2
            cv2.circle(debug, (lane_center_x, self.proc_h - 20), 5, (0, 0, 255), -1)
            cv2.circle(debug, (frame_center_x, self.proc_h - 20), 5, (255, 0, 0), -1)

        # Status text
        status_color = (0, 0, 255) if result.departure_warning else (0, 255, 0)
        status_text = "LDW: DEPARTURE!" if result.departure_warning else "LDW: OK"
        if self._recovery_mode:
            status_text = "LDW: RECOVERY"
            status_color = (0, 165, 255)

        cv2.putText(debug, status_text, (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)

        offset_text = f"Offset: {result.lane_center_offset:.1f}px"
        cv2.putText(debug, offset_text, (5, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        steer_text = f"Steer: {result.steering_correction:.2f}"
        cv2.putText(debug, steer_text, (5, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

        conf_text = f"Conf: {result.confidence:.2f}"
        cv2.putText(debug, conf_text, (5, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        return debug
