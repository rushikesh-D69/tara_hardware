"""
TARA ADAS — Lane Detection Module (LDW + LKA)
Pure OpenCV classical computer vision pipeline.
No ML model needed — saves compute budget for other features.

Pipeline:
  1. Resize → 2. HSV mask → 3. Bird's-eye warp → 4. Blur
  → 5. Canny → 6. Hough lines → 7. Average & fit lanes
  → 8. Inverse-warp lane points → 9. Compute offset

Outputs (via LaneDetectionResult):
  - lane_center_offset: float (pixels, + = drifting right)
  - departure_warning: bool
  - steering_correction: float (-1.0 to 1.0)
      Consumed by DecisionManager which scales it to [-100, 100]
      and routes it through serial_bridge to the ESP32.
  - lane_detected: bool
"""
import cv2
import numpy as np
import math
import threading
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
        self.debug_frame = None      # Visualization overlay


class LaneDetector:
    """
    Lane detection using classical computer vision.
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

        # Lane smoothing — keep history of recent detections
        # Protected by a lock to prevent corruption under concurrent access
        # (e.g. future multi-threaded scheduling on Pi 5).
        self._history_lock = threading.Lock()
        self._left_history = []
        self._right_history = []
        self._history_len = 5

        log.info(f"LaneDetector initialized: process at {self.proc_w}x{self.proc_h}")

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
            steering_correction is consumed downstream by
            DecisionManager → serial_bridge → ESP32.
        """
        result = LaneDetectionResult()

        # Step 1: Resize for faster processing
        small = cv2.resize(frame, (self.proc_w, self.proc_h))

        # Step 2: Convert to HSV and create lane mask
        lane_mask = self._create_lane_mask(small)

        # Step 3: Warp lane mask to bird's-eye view.
        # In BEV, lane lines become near-vertical / parallel, which makes
        # Hough slope-based L/R separation work reliably on curves.
        bev_mask = cv2.warpPerspective(
            lane_mask, self._M, (self.proc_w, self.proc_h))

        # Step 4: Gaussian blur (on BEV)
        blurred = cv2.GaussianBlur(bev_mask, self.cfg.BLUR_KERNEL, 0)

        # Step 5: Canny edge detection (on BEV)
        edges = cv2.Canny(blurred, self.cfg.CANNY_LOW, self.cfg.CANNY_HIGH)

        # Mask the top 40% of the BEV image — this is sky/horizon that leaked
        # through the perspective warp. Keeping it causes false Hough lines.
        sky_cutoff = int(self.proc_h * 0.40)
        edges[:sky_cutoff, :] = 0

        # No ROI trapezoid needed — the perspective warp already discards
        # sky/horizon; the full BEV rectangle is the road surface.

        # Step 6: Hough line detection (on BEV)
        lines = cv2.HoughLinesP(
            edges,
            rho=self.cfg.HOUGH_RHO,
            theta=np.pi / self.cfg.HOUGH_THETA_DIVISOR,
            threshold=self.cfg.HOUGH_THRESHOLD,
            minLineLength=self.cfg.HOUGH_MIN_LINE_LEN,
            maxLineGap=self.cfg.HOUGH_MAX_LINE_GAP,
        )

        # Step 7: Separate left/right lanes and average (in BEV space)
        left_bev, right_bev = self._average_lane_lines_bev(lines)

        # Apply temporal smoothing (thread-safe)
        with self._history_lock:
            left_bev = self._smooth_lane(left_bev, self._left_history)
            right_bev = self._smooth_lane(right_bev, self._right_history)

        # Step 8: Inverse-warp lane endpoints back to camera perspective
        # so offset calculations and debug overlays align with the
        # original image.
        left_lane = self._warp_lane_to_camera(left_bev)
        right_lane = self._warp_lane_to_camera(right_bev)

        result.left_lane = left_lane
        result.right_lane = right_lane

        # Step 9: Compute lane center offset and warnings
        if left_lane is not None and right_lane is not None:
            result.lane_detected = True

            # Calculate the center of the detected lane at the bottom of the frame
            lane_center_x = (left_lane[0] + right_lane[0]) / 2  # x coords at bottom (y1)
            frame_center_x = self.proc_w / 2

            result.lane_center_offset = lane_center_x - frame_center_x

            # Lane Departure Warning
            if abs(result.lane_center_offset) > self.cfg.LANE_DEPARTURE_THRESHOLD:
                result.departure_warning = True

            # Lane Keeping Assist — proportional normalized steering
            # steering_correction ∈ [-1.0, 1.0] maps directly to ESP32 jd.x
            normalized_offset = result.lane_center_offset / (self.proc_w / 2)
            result.steering_correction = max(-1.0, min(1.0, normalized_offset))

        elif left_lane is not None or right_lane is not None:
            # Only one lane visible — estimate center from single lane
            result.lane_detected = True
            
            # Assuming lane width is ~70% of frame width in BEV, which maps
            # roughly to a similar proportion at the bottom of the camera view.
            half_lane = 0.35 * self.proc_w
            
            if left_lane is not None:
                # Only left lane visible — estimate center by adding half-width
                est_center = left_lane[0] + half_lane
            else:
                # Only right lane visible — estimate center by subtracting half-width
                est_center = right_lane[0] - half_lane
                
            result.lane_center_offset = est_center - (self.proc_w / 2)
            
            if abs(result.lane_center_offset) > self.cfg.LANE_DEPARTURE_THRESHOLD:
                result.departure_warning = True
                
            normalized_offset = result.lane_center_offset / (self.proc_w / 2)
            result.steering_correction = max(-1.0, min(1.0, normalized_offset))
        else:
            # No lanes detected
            result.lane_detected = False

        # Debug visualization
        if debug:
            result.debug_frame = self._draw_debug(small, result, edges)

        return result

    def _create_lane_mask(self, frame):
        """
        Create a binary mask of lane markings using HSV color filtering.

        Args:
            frame: BGR image

        Returns:
            Binary mask (uint8)
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # White lanes
        white_mask = cv2.inRange(
            hsv,
            np.array(self.cfg.LANE_WHITE_HSV_LOW),
            np.array(self.cfg.LANE_WHITE_HSV_HIGH),
        )

        # Yellow lanes
        yellow_mask = cv2.inRange(
            hsv,
            np.array(self.cfg.LANE_YELLOW_HSV_LOW),
            np.array(self.cfg.LANE_YELLOW_HSV_HIGH),
        )

        # Combine masks
        combined = cv2.bitwise_or(white_mask, yellow_mask)
        return combined

    def _apply_roi_mask(self, edges):
        """
        Apply a trapezoidal Region of Interest mask.
        Only keeps the bottom 60% of the frame where road is visible.

        Args:
            edges: Edge-detected image

        Returns:
            Masked edge image
        """
        h, w = edges.shape
        mask = np.zeros_like(edges)

        # Trapezoid vertices (bottom-heavy, covers road area)
        polygon = np.array([
            [
                (int(w * 0.0), h),          # bottom-left
                (int(w * 0.35), int(h * 0.55)),  # top-left
                (int(w * 0.65), int(h * 0.55)),  # top-right
                (int(w * 1.0), h),          # bottom-right
            ]
        ], dtype=np.int32)

        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(edges, mask)

    def _average_lane_lines_bev(self, lines):
        """
        Separate Hough lines into left and right lanes by x-position
        in bird's-eye view space, then average each group.

        In BEV the lanes are near-vertical, so we separate by which
        half of the frame the line sits in rather than by slope sign
        (which is fragile in perspective view for curves).

        Args:
            lines: Output from cv2.HoughLinesP (in BEV coordinates)

        Returns:
            (left_line, right_line) each as (x1, y1, x2, y2) or None
        """
        if lines is None:
            return None, None

        mid_x = self.proc_w / 2
        left_xs = []      # collect all x coords for averaging
        left_slopes = []
        left_intercepts = []
        right_xs = []
        right_slopes = []
        right_intercepts = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                # Perfectly vertical — still useful in BEV
                avg_x = (x1 + x2) / 2
            else:
                avg_x = (x1 + x2) / 2

            dx = x2 - x1
            if dx == 0:
                slope = 1e6  # near-infinite
                intercept = x1   # not meaningful for vertical, but store x
            else:
                slope = (y2 - y1) / dx
                intercept = y1 - slope * x1

            # In BEV, reject near-horizontal lines (slope angle < 45°)
            # Lane lines must be steeply angled — < 45° means noise/road edge.
            if dx != 0:
                angle = abs(math.degrees(math.atan(slope)))
                if angle < 45:
                    continue

            if avg_x < mid_x:
                left_slopes.append(slope)
                left_intercepts.append(intercept)
            else:
                right_slopes.append(slope)
                right_intercepts.append(intercept)

        left_lane = self._make_lane_points_bev(left_slopes, left_intercepts)
        right_lane = self._make_lane_points_bev(right_slopes, right_intercepts)

        return left_lane, right_lane

    def _make_lane_points_bev(self, slopes, intercepts):
        """
        Average slopes/intercepts and convert to line coordinates
        within the bird's-eye view frame.

        Returns:
            (x1, y1, x2, y2) — line spanning the BEV image, or None
        """
        if not slopes:
            return None

        avg_slope = np.mean(slopes)
        avg_intercept = np.mean(intercepts)

        # Span the full height of the BEV image
        y1 = self.proc_h           # bottom
        y2 = 0                     # top

        if abs(avg_slope) < 1e-6:
            return None

        x1 = int((y1 - avg_intercept) / avg_slope)
        x2 = int((y2 - avg_intercept) / avg_slope)

        # Sanity check — line should be within frame bounds (with margin)
        if x1 < -self.proc_w or x1 > 2 * self.proc_w:
            return None
        if x2 < -self.proc_w or x2 > 2 * self.proc_w:
            return None

        return (x1, y1, x2, y2)

    def _warp_lane_to_camera(self, bev_lane):
        """
        Transform a lane line from bird's-eye coordinates back to
        original camera perspective using the inverse warp matrix.

        Args:
            bev_lane: (x1, y1, x2, y2) in BEV space, or None

        Returns:
            (x1, y1, x2, y2) in camera space, or None
        """
        if bev_lane is None:
            return None

        x1, y1, x2, y2 = bev_lane
        pts_bev = np.float32([[x1, y1], [x2, y2]]).reshape(-1, 1, 2)
        pts_cam = cv2.perspectiveTransform(pts_bev, self._M_inv)

        cx1, cy1 = pts_cam[0][0]
        cx2, cy2 = pts_cam[1][0]
        return (int(cx1), int(cy1), int(cx2), int(cy2))

    def _smooth_lane(self, current, history):
        """
        Temporal smoothing using a history buffer.
        Averages recent detections to reduce jitter.

        MUST be called while holding self._history_lock.

        Args:
            current: Current lane line or None
            history: List of recent lane lines

        Returns:
            Smoothed lane line
        """
        if current is not None:
            history.append(current)
        while len(history) > self._history_len:
            history.pop(0)

        if not history:
            return None

        # Snapshot the list to a local tuple so np.mean never sees a
        # partially-mutated list (defensive even under the lock).
        snapshot = list(history)

        avg_x1 = int(np.mean([l[0] for l in snapshot]))
        avg_y1 = int(np.mean([l[1] for l in snapshot]))
        avg_x2 = int(np.mean([l[2] for l in snapshot]))
        avg_y2 = int(np.mean([l[3] for l in snapshot]))

        return (avg_x1, avg_y1, avg_x2, avg_y2)

    def _draw_debug(self, frame, result, roi_edges):
        """
        Draw debug visualization with detected lanes overlay.

        Args:
            frame: Original frame (small)
            result: LaneDetectionResult
            roi_edges: Edge-detected ROI image

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

        # Draw lane center and frame center
        if result.left_lane is not None and result.right_lane is not None:
            lane_center_x = int((result.left_lane[2] + result.right_lane[2]) / 2)
            frame_center_x = self.proc_w // 2

            # Fill lane area (green polygon)
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
            cv2.circle(debug, (lane_center_x, self.proc_h - 20), 5, (0, 0, 255), -1)
            cv2.circle(debug, (frame_center_x, self.proc_h - 20), 5, (255, 0, 0), -1)

        # Status text
        status_color = (0, 0, 255) if result.departure_warning else (0, 255, 0)
        status_text = "LDW: DEPARTURE!" if result.departure_warning else "LDW: OK"
        cv2.putText(debug, status_text, (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)

        offset_text = f"Offset: {result.lane_center_offset:.1f}px"
        cv2.putText(debug, offset_text, (5, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        steer_text = f"Steer: {result.steering_correction:.2f}"
        cv2.putText(debug, steer_text, (5, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

        return debug
