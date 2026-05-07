"""
TARA ADAS — Traffic Sign Recognition (TSR)
MobileNetV2 (α=0.35, 96×96) INT8 TFLite inference on GTSRB (43 classes).

Pipeline: Crop ROI → Reject dark frames → Resize → Preprocess
          → Inference → Confidence filter → Majority vote → Stable output

Key optimisations:
  - ROI crop before resize (avoids wasting pixels on floor/sky)
  - MobileNetV2 preprocessing: scale to [-1, 1]
  - Temporal majority voting over 5 frames (eliminates flickering)
  - Dark frame early rejection (~15 ms saved per skip)
  - INT8 dequantisation with fallback to legacy quantisation tuple

Performance: ~15–25 ms per inference on Raspberry Pi 4B.
"""
import time
import cv2
import numpy as np
from collections import deque, Counter
from utils.logger import get_logger

log = get_logger("TSR")


class TSRResult:
    """Container for traffic sign recognition outputs."""

    def __init__(self):
        self.sign_detected = False
        self.class_id      = -1
        self.class_name    = ""
        self.confidence    = 0.0
        self.speed_limit   = None
        self.is_stop_sign  = False
        self.inference_ms  = 0.0


class TrafficSignRecognizer:
    """
    Traffic sign recognition using a lightweight MobileNetV2 TFLite model.
    Designed for real-time inference on Raspberry Pi 4B.
    """

    def __init__(self, config):
        self.cfg             = config
        self.input_size      = config.TSR_INPUT_SIZE
        self.conf_threshold  = config.TSR_CONFIDENCE_THRESHOLD
        self.sign_names      = config.TSR_SIGN_NAMES
        self.speed_limits    = config.TSR_SPEED_LIMITS

        self._interpreter    = None
        self._input_details  = None
        self._output_details = None
        self._loaded         = False

        self._roi_y_start = 0.2
        self._roi_y_end   = 0.95
        self._roi_x_start = 0.2
        self._roi_x_end   = 0.8

        self._prediction_buffer = deque(maxlen=5)
        self._frame_skip        = getattr(config, 'TSR_FRAME_SKIP', 1)
        self._call_count        = 0
        self._min_brightness    = 30
        self._last_stable_result = TSRResult()

        log.info(f"TSR initialized: input {self.input_size}x{self.input_size}, "
                 f"ROI [{self._roi_y_start:.0%}-{self._roi_y_end:.0%}] × "
                 f"[{self._roi_x_start:.0%}-{self._roi_x_end:.0%}], skip={self._frame_skip}")

    def load_model(self):
        """Load TFLite model via ai-edge-litert. Install: pip install ai-edge-litert"""
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            log.error("ai-edge-litert not installed. Run: pip install ai-edge-litert")
            return False

        try:
            self._interpreter = Interpreter(model_path=self.cfg.TSR_MODEL_PATH, num_threads=4)
            self._interpreter.allocate_tensors()

            self._input_details  = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()

            in_shape  = self._input_details[0]['shape']
            in_dtype  = self._input_details[0]['dtype']
            out_shape = self._output_details[0]['shape']

            expected = (1, self.input_size, self.input_size, 3)
            if tuple(in_shape) != expected:
                log.warning(f"TSR model input shape {in_shape} != expected {expected}. Adjusting.")
                self.input_size = in_shape[1]

            log.info(f"TSR model loaded: input={in_shape} ({in_dtype}), output={out_shape}")
            self._loaded = True
            return True

        except Exception as e:
            log.error(f"Failed to load TSR model: {e}")
            return False

    def detect(self, frame):
        """
        Run traffic sign recognition on a frame.

        Args:
            frame: BGR image (any size)

        Returns:
            TSRResult
        """
        result = TSRResult()
        if not self._loaded:
            return result

        self._call_count += 1
        if self._frame_skip > 1 and (self._call_count % self._frame_skip != 0):
            return self._last_stable_result

        t_start = time.monotonic()

        h, w = frame.shape[:2]
        y1 = int(h * self._roi_y_start); y2 = int(h * self._roi_y_end)
        x1 = int(w * self._roi_x_start); x2 = int(w * self._roi_x_end)
        roi = frame[y1:y2, x1:x2]

        if roi.mean() < self._min_brightness:
            result.inference_ms = (time.monotonic() - t_start) * 1000
            return result

        input_img  = cv2.resize(roi, (self.input_size, self.input_size))
        input_img  = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_dtype = self._input_details[0]['dtype']

        if input_dtype == np.uint8:
            input_data = np.expand_dims(input_img.astype(np.uint8), axis=0)
        else:
            preprocessed = (input_img.astype(np.float32) / 127.5) - 1.0
            input_data   = np.expand_dims(preprocessed, axis=0)

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

        if np.any(output < 0) or np.sum(output) < 0.9 or np.sum(output) > 1.1:
            exp_output = np.exp(output - np.max(output))
            output     = exp_output / np.sum(exp_output)

        class_id   = int(np.argmax(output))
        confidence = float(output[class_id])

        t_end = time.monotonic()
        result.inference_ms = (t_end - t_start) * 1000
        log.debug(f"TSR inference: {result.inference_ms:.1f}ms")

        if confidence < self.conf_threshold:
            return result

        self._prediction_buffer.append(class_id)
        if len(self._prediction_buffer) < 2:
            return result

        vote_counts                     = Counter(self._prediction_buffer)
        voted_class_id, voted_count     = vote_counts.most_common(1)[0]

        if voted_count < 2:
            return result

        result.sign_detected = True
        result.class_id      = voted_class_id
        result.class_name    = self.sign_names.get(voted_class_id, f"Unknown ({voted_class_id})")
        result.confidence    = confidence

        if voted_class_id in self.speed_limits:
            result.speed_limit = self.speed_limits[voted_class_id]

        if voted_class_id == 14:
            result.is_stop_sign = True

        log.debug(f"TSR: {result.class_name} ({confidence:.2f}), votes={voted_count}/5")

        self._last_stable_result = result
        return result

    @property
    def is_loaded(self):
        """Whether the model is loaded and ready."""
        return self._loaded
