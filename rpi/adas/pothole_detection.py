"""
TARA ADAS — Pothole Detection & Avoidance
Two modes:
  1. Binary classifier (fast, ~20ms) — MobileNetV2 α=0.35, 128x128
  2. SSD detector (precise, ~100ms) — SSD-MobileNetV2, 300x300

Default: Binary classifier for RPi 4B performance.
"""
import cv2
import numpy as np
from utils.logger import get_logger

log = get_logger("Pothole")


class PotholeResult:
    """Container for pothole detection outputs."""

    def __init__(self):
        self.pothole_detected = False
        self.confidence = 0.0
        self.position = "center"     # "left", "center", "right"
        self.avoidance_steer = 0     # Steering offset: -ve=left, +ve=right
        self.bounding_box = None     # (x1, y1, x2, y2) if using SSD


class PotholeDetector:
    """
    Pothole detection optimized for Raspberry Pi 4B.
    Uses a MobileNetV2 binary classifier by default for speed.
    """

    def __init__(self, config):
        """
        Args:
            config: Config module with pothole detection parameters
        """
        self.cfg = config
        self.use_ssd = config.POTHOLE_USE_SSD
        self.conf_threshold = config.POTHOLE_CONFIDENCE_THRESHOLD
        self.steer_magnitude = config.POTHOLE_STEER_MAGNITUDE

        if self.use_ssd:
            self.input_size = config.POTHOLE_SSD_INPUT_SIZE
            self.model_path = config.POTHOLE_SSD_MODEL_PATH
        else:
            self.input_size = config.POTHOLE_INPUT_SIZE
            self.model_path = config.POTHOLE_MODEL_PATH

        # Absolute fallback path — used if config path doesn't resolve to a real file.
        # This covers cases where the project is run from a different working directory.
        _ABSOLUTE_MODEL_PATH = "rpi/models/pothole_mobilenetv2_int8.tflite"

        import os as _os
        if not _os.path.isfile(self.model_path):
            log.warning(
                f"Model not found at config path: {self.model_path!r}. "
                f"Falling back to absolute path: {_ABSOLUTE_MODEL_PATH!r}"
            )
            self.model_path = _ABSOLUTE_MODEL_PATH

        # TFLite interpreter
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._loaded = False

        mode = "SSD Object Detection" if self.use_ssd else "Binary Classifier"
        log.info(f"PotholeDetector initialized: {mode}, input {self.input_size}x{self.input_size}")

    def load_model(self):
        """Load the TFLite model.
        On Raspberry Pi 4B, tflite-runtime is preferred over full TensorFlow.
        Install: pip install tflite-runtime
        """
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            try:
                import tensorflow.lite as tflite
                log.warning("tflite-runtime not found — using full TensorFlow (slow on RPi). "
                            "Install: pip install tflite-runtime")
            except ImportError:
                log.error("No TFLite backend found. Run: pip install tflite-runtime")
                return False

        try:
            self._interpreter = tflite.Interpreter(
                model_path=self.model_path,
                num_threads=4,   # RPi 4B has 4 cores — use all of them
            )
            self._interpreter.allocate_tensors()
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()

            in_shape = self._input_details[0]['shape']
            log.info(f"Pothole model loaded: input={in_shape}")

            self._loaded = True
            return True

        except Exception as e:
            log.error(f"Failed to load pothole model: {e}")
            return False

    def detect(self, frame):
        """
        Run pothole detection on a frame.

        Args:
            frame: BGR image from camera

        Returns:
            PotholeResult with detection info and avoidance command
        """
        result = PotholeResult()

        if not self._loaded:
            return result

        # Extract the lower half of the frame (road surface)
        h, w = frame.shape[:2]
        road_roi = frame[int(h * 0.5):, :]

        if self.use_ssd:
            return self._detect_ssd(road_roi, result, w)
        else:
            return self._detect_classifier(road_roi, result, w)

    def _detect_classifier(self, road_roi, result, frame_width):
        """
        Binary classification approach: "pothole" vs "clear road".
        Then estimate pothole position from the ROI.

        Args:
            road_roi: Bottom half of frame (road surface)
            result: PotholeResult to populate
            frame_width: Full frame width for position calculation

        Returns:
            Populated PotholeResult
        """
        # Resize for model input
        input_img = cv2.resize(road_roi, (self.input_size, self.input_size))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        # Prepare input tensor
        input_dtype = self._input_details[0]['dtype']
        if input_dtype == np.uint8:
            input_data = np.expand_dims(input_img.astype(np.uint8), axis=0)
        else:
            input_data = np.expand_dims(input_img.astype(np.float32) / 255.0, axis=0)

        # Run inference
        self._interpreter.set_tensor(self._input_details[0]['index'], input_data)
        self._interpreter.invoke()

        output = self._interpreter.get_tensor(self._output_details[0]['index'])[0]

        # Dequantize if needed
        output_dtype = self._output_details[0]['dtype']
        if output_dtype == np.uint8 or output_dtype == np.int8:
            quant = self._output_details[0].get('quantization', None)
            if quant and len(quant) >= 2:
                scale, zero_point = quant
                output = (output.astype(np.float32) - zero_point) * scale

        # Apply softmax if output isn't already a probability distribution
        if np.any(output < 0) or not (0.9 <= np.sum(output) <= 1.1):
            exp_out = np.exp(output - np.max(output))
            output = exp_out / np.sum(exp_out)

        # Binary classifier — MobileNetV2 standard ordering:
        #   output[0] = clear_road probability
        #   output[1] = pothole probability
        # (max() was WRONG — it always picked the dominant clear-road score)
        if len(output) >= 2:
            score_clear  = float(output[0])
            pothole_prob = float(output[1])
            log.debug(
                f"Pothole scores — clear={score_clear:.3f}  pothole={pothole_prob:.3f}  "
                f"threshold={self.conf_threshold}"
            )
        else:
            pothole_prob = float(output[0])
            log.debug(f"Pothole score — pothole={pothole_prob:.3f}  threshold={self.conf_threshold}")

        if pothole_prob >= self.conf_threshold:
            result.pothole_detected = True
            result.confidence = pothole_prob

            # Estimate pothole position using simple intensity analysis
            # Divide ROI into left, center, right thirds
            result.position = self._estimate_position(road_roi)

            # Generate avoidance steering command
            if result.position == "left":
                result.avoidance_steer = self.steer_magnitude  # Steer right
            elif result.position == "right":
                result.avoidance_steer = -self.steer_magnitude  # Steer left
            else:
                # Center pothole — steer to whichever side has more space
                result.avoidance_steer = self.steer_magnitude  # Default: dodge right

            log.info(f"POTHOLE detected! pos={result.position}, "
                     f"conf={pothole_prob:.2f}, steer={result.avoidance_steer}")

        return result

    def _detect_ssd(self, road_roi, result, frame_width):
        """
        SSD object detection approach: precise bounding box localization.

        Args:
            road_roi: Bottom half of frame
            result: PotholeResult to populate
            frame_width: Full frame width

        Returns:
            Populated PotholeResult
        """
        # Resize for SSD input
        input_img = cv2.resize(road_roi, (self.input_size, self.input_size))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        input_dtype = self._input_details[0]['dtype']
        if input_dtype == np.uint8:
            input_data = np.expand_dims(input_img.astype(np.uint8), axis=0)
        else:
            input_data = np.expand_dims(input_img.astype(np.float32) / 255.0, axis=0)

        # Run inference
        self._interpreter.set_tensor(self._input_details[0]['index'], input_data)
        self._interpreter.invoke()

        # SSD outputs: [boxes, classes, scores, num_detections]
        boxes = self._interpreter.get_tensor(self._output_details[0]['index'])[0]
        classes = self._interpreter.get_tensor(self._output_details[1]['index'])[0]
        scores = self._interpreter.get_tensor(self._output_details[2]['index'])[0]

        # Find best detection
        best_idx = np.argmax(scores)
        best_score = float(scores[best_idx])

        if best_score >= self.conf_threshold:
            result.pothole_detected = True
            result.confidence = best_score

            # Get bounding box
            ymin, xmin, ymax, xmax = boxes[best_idx]
            roi_h, roi_w = road_roi.shape[:2]
            result.bounding_box = (
                int(xmin * roi_w),
                int(ymin * roi_h),
                int(xmax * roi_w),
                int(ymax * roi_h),
            )

            # Determine position from bounding box center
            box_center_x = (xmin + xmax) / 2
            if box_center_x < 0.33:
                result.position = "left"
                result.avoidance_steer = self.steer_magnitude
            elif box_center_x > 0.66:
                result.position = "right"
                result.avoidance_steer = -self.steer_magnitude
            else:
                result.position = "center"
                result.avoidance_steer = self.steer_magnitude

            log.info(f"POTHOLE (SSD) detected! pos={result.position}, "
                     f"conf={best_score:.2f}, bbox={result.bounding_box}")

        return result

    def _estimate_position(self, road_roi):
        """
        Estimate pothole position by analyzing intensity in thirds.
        Potholes appear darker than surrounding road.

        Args:
            road_roi: Road surface image

        Returns:
            "left", "center", or "right"
        """
        gray = cv2.cvtColor(road_roi, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding to highlight dark regions (potholes)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        h, w = thresh.shape
        third = w // 3

        # Count dark pixels in each third
        left_dark = np.sum(thresh[:, :third] > 0)
        center_dark = np.sum(thresh[:, third:2*third] > 0)
        right_dark = np.sum(thresh[:, 2*third:] > 0)

        max_dark = max(left_dark, center_dark, right_dark)
        if max_dark == left_dark:
            return "left"
        elif max_dark == right_dark:
            return "right"
        else:
            return "center"

    @property
    def is_loaded(self):
        """Whether the model is loaded and ready."""
        return self._loaded
