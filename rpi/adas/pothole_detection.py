"""
TARA ADAS — Pothole Detection & Avoidance
Two operating modes:
  Binary classifier (default) — MobileNetV2 α=0.35, 128×128, ~20 ms
  SSD detector (optional)     — SSD-MobileNetV2, 300×300, ~100 ms
"""
import cv2
import numpy as np
from utils.logger import get_logger

log = get_logger("Pothole")


class PotholeResult:
    """Container for pothole detection outputs."""

    def __init__(self):
        self.pothole_detected = False
        self.confidence       = 0.0
        self.position         = "center"
        self.avoidance_steer  = 0
        self.bounding_box     = None


class PotholeDetector:
    """
    Pothole detection optimised for Raspberry Pi 4B.
    Uses MobileNetV2 binary classifier by default; SSD optional.
    """

    def __init__(self, config):
        self.cfg            = config
        self.use_ssd        = config.POTHOLE_USE_SSD
        self.conf_threshold = config.POTHOLE_CONFIDENCE_THRESHOLD
        self.steer_magnitude = config.POTHOLE_STEER_MAGNITUDE

        if self.use_ssd:
            self.input_size = config.POTHOLE_SSD_INPUT_SIZE
            self.model_path = config.POTHOLE_SSD_MODEL_PATH
        else:
            self.input_size = config.POTHOLE_INPUT_SIZE
            self.model_path = config.POTHOLE_MODEL_PATH

        import os as _os
        _FALLBACK_PATH = "rpi/models/pothole_mobilenetv2_int8.tflite"
        if not _os.path.isfile(self.model_path):
            log.warning(f"Model not found at {self.model_path!r} — using fallback")
            self.model_path = _FALLBACK_PATH

        self._interpreter    = None
        self._input_details  = None
        self._output_details = None
        self._loaded         = False

        mode = "SSD Object Detection" if self.use_ssd else "Binary Classifier"
        log.info(f"PotholeDetector initialized: {mode}, {self.input_size}x{self.input_size}")

    def load_model(self):
        """Load TFLite model via ai-edge-litert. Install: pip install ai-edge-litert"""
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            log.error("ai-edge-litert not installed. Run: pip install ai-edge-litert")
            return False

        try:
            self._interpreter = Interpreter(model_path=self.model_path, num_threads=4)
            self._interpreter.allocate_tensors()
            self._input_details  = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()

            log.info(f"Pothole model loaded: input={self._input_details[0]['shape']}")
            self._loaded = True
            return True

        except Exception as e:
            log.error(f"Failed to load pothole model: {e}")
            return False

    def detect(self, frame):
        """
        Run pothole detection on the lower half of the frame (road surface).

        Args:
            frame: BGR image from camera

        Returns:
            PotholeResult
        """
        result = PotholeResult()
        if not self._loaded:
            return result

        h, w      = frame.shape[:2]
        road_roi  = frame[int(h * 0.5):, :]

        if self.use_ssd:
            return self._detect_ssd(road_roi, result, w)
        return self._detect_classifier(road_roi, result, w)

    def _detect_classifier(self, road_roi, result, frame_width):
        """Binary classification: pothole vs. clear road."""
        input_img   = cv2.resize(road_roi, (self.input_size, self.input_size))
        input_img   = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_dtype = self._input_details[0]['dtype']

        if input_dtype == np.uint8:
            input_data = np.expand_dims(input_img.astype(np.uint8), axis=0)
        else:
            input_data = np.expand_dims(input_img.astype(np.float32) / 255.0, axis=0)

        self._interpreter.set_tensor(self._input_details[0]['index'], input_data)
        self._interpreter.invoke()

        output      = self._interpreter.get_tensor(self._output_details[0]['index'])[0]
        output_dtype = self._output_details[0]['dtype']

        if output_dtype in (np.uint8, np.int8):
            quant_params = self._output_details[0].get('quantization_parameters', {})
            scales       = quant_params.get('scales', None)
            zero_points  = quant_params.get('zero_points', None)
            if scales is not None and len(scales) > 0:
                output = (output.astype(np.float32) - zero_points[0]) * scales[0]
            else:
                quant = self._output_details[0].get('quantization', (1.0, 0))
                if quant and len(quant) >= 2:
                    scale, zero_point = quant
                    output = (output.astype(np.float32) - zero_point) * scale

        if np.any(output < 0) or not (0.9 <= np.sum(output) <= 1.1):
            exp_out = np.exp(output - np.max(output))
            output  = exp_out / np.sum(exp_out)

        if len(output) >= 2:
            pothole_prob = float(output[1])
            log.debug(f"Pothole — clear={output[0]:.3f}  pothole={pothole_prob:.3f}")
        else:
            pothole_prob = float(output[0])
            log.debug(f"Pothole — pothole={pothole_prob:.3f}")

        if pothole_prob >= self.conf_threshold:
            result.pothole_detected = True
            result.confidence       = pothole_prob
            result.position         = self._estimate_position(road_roi)

            if result.position == "left":
                result.avoidance_steer =  self.steer_magnitude
            elif result.position == "right":
                result.avoidance_steer = -self.steer_magnitude
            else:
                result.avoidance_steer =  self.steer_magnitude

            log.info(f"POTHOLE detected: pos={result.position}, conf={pothole_prob:.2f}")

        return result

    def _detect_ssd(self, road_roi, result, frame_width):
        """SSD object detection: precise bounding-box localisation."""
        input_img   = cv2.resize(road_roi, (self.input_size, self.input_size))
        input_img   = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_dtype = self._input_details[0]['dtype']

        if input_dtype == np.uint8:
            input_data = np.expand_dims(input_img.astype(np.uint8), axis=0)
        else:
            input_data = np.expand_dims(input_img.astype(np.float32) / 255.0, axis=0)

        self._interpreter.set_tensor(self._input_details[0]['index'], input_data)
        self._interpreter.invoke()

        boxes   = self._interpreter.get_tensor(self._output_details[0]['index'])[0]
        scores  = self._interpreter.get_tensor(self._output_details[2]['index'])[0]

        best_idx   = np.argmax(scores)
        best_score = float(scores[best_idx])

        if best_score >= self.conf_threshold:
            result.pothole_detected = True
            result.confidence       = best_score

            ymin, xmin, ymax, xmax = boxes[best_idx]
            roi_h, roi_w = road_roi.shape[:2]
            result.bounding_box = (
                int(xmin * roi_w), int(ymin * roi_h),
                int(xmax * roi_w), int(ymax * roi_h),
            )

            box_center_x = (xmin + xmax) / 2
            if box_center_x < 0.33:
                result.position        = "left"
                result.avoidance_steer =  self.steer_magnitude
            elif box_center_x > 0.66:
                result.position        = "right"
                result.avoidance_steer = -self.steer_magnitude
            else:
                result.position        = "center"
                result.avoidance_steer =  self.steer_magnitude

            log.info(f"POTHOLE (SSD): pos={result.position}, conf={best_score:.2f}")

        return result

    def _estimate_position(self, road_roi):
        """
        Estimate pothole position from edge density in left / center / right thirds.
        Edge density is invariant to absolute brightness — robust on dark floors.
        """
        gray  = cv2.cvtColor(road_roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)

        h, w  = edges.shape
        third = w // 3

        left_edges   = np.sum(edges[:, :third]       > 0)
        center_edges = np.sum(edges[:, third:2*third] > 0)
        right_edges  = np.sum(edges[:, 2*third:]     > 0)

        total_edges = left_edges + center_edges + right_edges
        if total_edges < 100:
            return "center"

        max_edges = max(left_edges, center_edges, right_edges)
        if max_edges == left_edges:
            return "left"
        elif max_edges == right_edges:
            return "right"
        return "center"

    @property
    def is_loaded(self):
        """Whether the model is loaded and ready."""
        return self._loaded
