"""
TARA ADAS — Traffic Sign Recognition (TSR)
MobileNetV2 (α=0.35, 96x96) via TFLite INT8 inference.
Trained on GTSRB dataset (43 classes).

Pipeline order (strict): Crop → Resize → Preprocess → Predict → Smooth → Output

Optimizations for RPi 4B:
  [1]  ROI cropping — only process center region where signs appear
  [2]  MobileNetV2 preprocessing — correct [-1, 1] scaling
  [3]  Strict pipeline order — crop before resize, always
  [4]  Confidence threshold — ignore weak predictions
  [5]  Temporal smoothing — majority vote over last 5 predictions
  [6]  Frame skipping — process every Nth frame (configurable)
  [7]  Direct ROI resize — no intermediate sizes
  [8]  Inference timing — measure and log FPS
  [9]  Early frame rejection — skip dark / empty frames
  [10] Stable output gating — only output if confident AND stable
  [11] TFLite compatible — input shape (1, 96, 96, 3)

Performance: ~15-25ms per inference on RPi 4B.
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
        self.class_id = -1
        self.class_name = ""
        self.confidence = 0.0
        self.speed_limit = None   # PWM value if it's a speed limit sign
        self.is_stop_sign = False
        self.inference_ms = 0.0   # [8] Inference time for monitoring


class TrafficSignRecognizer:
    """
    Traffic sign recognition using a lightweight MobileNetV2 TFLite model.
    Designed for real-time inference on Raspberry Pi 4B.

    Key design decisions:
      - ROI crop before resize (avoids wasting pixels on road/sky)
      - MobileNetV2 preprocessing (scale to [-1, 1] not [0, 1])
      - Majority voting over 5 frames eliminates flickering
      - Dark frame rejection saves ~15ms per skipped frame
    """

    def __init__(self, config):
        """
        Args:
            config: Config module with TSR parameters
        """
        self.cfg = config
        self.input_size = config.TSR_INPUT_SIZE    # 96
        self.conf_threshold = config.TSR_CONFIDENCE_THRESHOLD  # 0.6
        self.sign_names = config.TSR_SIGN_NAMES
        self.speed_limits = config.TSR_SPEED_LIMITS

        # TFLite interpreter
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._loaded = False

        # ── [1] ROI cropping ratios ───────────────────────────────────
        # On an indoor track, signs are roughly at center height.
        # Crop to the center 40%×40% of the frame to avoid processing
        # floor, ceiling, and side walls.
        self._roi_y_start = 0.3   # 30% from top
        self._roi_y_end = 0.7     # 70% from top
        self._roi_x_start = 0.3   # 30% from left
        self._roi_x_end = 0.7     # 70% from left

        # ── [5] Temporal smoothing — majority vote buffer ─────────────
        # Keeps last 5 predictions. Final output = most frequent class.
        # This kills flickering: e.g. [Stop, Stop, 30, Stop, Stop] → Stop
        self._prediction_buffer = deque(maxlen=5)

        # ── [6] Frame skipping counter ────────────────────────────────
        # Only run inference every Nth call. The scheduler in main.py
        # already staggers TSR to every 4th frame; this adds a second
        # layer if needed. Set to 1 = process every call (no skip).
        self._frame_skip = getattr(config, 'TSR_FRAME_SKIP', 1)
        self._call_count = 0

        # ── [9] Early rejection brightness threshold ──────────────────
        # If the ROI mean pixel value is below this, the frame is too
        # dark to contain a readable sign — skip inference entirely.
        self._min_brightness = 30  # 0-255 scale

        # ── [10] Last stable output (for gating) ─────────────────────
        self._last_stable_result = TSRResult()

        log.info(f"TSR initialized: input {self.input_size}x{self.input_size}, "
                 f"ROI [{self._roi_y_start:.0%}-{self._roi_y_end:.0%}] × "
                 f"[{self._roi_x_start:.0%}-{self._roi_x_end:.0%}], "
                 f"vote_buffer=5, skip={self._frame_skip}")

    def load_model(self):
        """
        Load the TFLite model. Call this once during startup.
        Separated from __init__ to allow graceful fallback if model not found.
        """
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            try:
                # Fallback: use TFLite from full TensorFlow
                import tensorflow.lite as tflite
                log.warning("Using full TensorFlow for TFLite — "
                            "install tflite-runtime for production")
            except ImportError:
                log.error("Neither tflite-runtime nor tensorflow found! "
                          "TSR will be disabled.")
                return False

        model_path = self.cfg.TSR_MODEL_PATH
        try:
            self._interpreter = tflite.Interpreter(
                model_path=model_path,
                num_threads=4,  # RPi 4B has 4 cores — use them all
            )
            self._interpreter.allocate_tensors()

            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()

            # [11] Verify model input shape is (1, 96, 96, 3)
            in_shape = self._input_details[0]['shape']
            in_dtype = self._input_details[0]['dtype']
            out_shape = self._output_details[0]['shape']

            expected = (1, self.input_size, self.input_size, 3)
            if tuple(in_shape) != expected:
                log.warning(f"TSR model input shape {in_shape} != expected "
                            f"{expected}. Adjusting input_size.")
                self.input_size = in_shape[1]

            log.info(f"TSR model loaded: input={in_shape} ({in_dtype}), "
                     f"output={out_shape}")

            self._loaded = True
            return True

        except Exception as e:
            log.error(f"Failed to load TSR model from {model_path}: {e}")
            return False

    def detect(self, frame):
        """
        Run traffic sign recognition on a frame.

        Pipeline: Crop → Reject dark → Resize → Preprocess → Predict
                  → Confidence filter → Majority vote → Stable output

        Args:
            frame: BGR image from camera (any size)

        Returns:
            TSRResult with detection info
        """
        result = TSRResult()

        if not self._loaded:
            return result

        # ── [6] Frame skipping ────────────────────────────────────────
        # If frame_skip > 1, return the last stable result on skipped frames.
        self._call_count += 1
        if self._frame_skip > 1 and (self._call_count % self._frame_skip != 0):
            return self._last_stable_result

        # ── [8] Start inference timer ─────────────────────────────────
        t_start = time.monotonic()

        # ── [1] ROI crop — center region where signs are expected ─────
        # Crop BEFORE resize so we don't waste pixels on floor/ceiling.
        h, w = frame.shape[:2]
        y1, y2 = int(h * self._roi_y_start), int(h * self._roi_y_end)
        x1, x2 = int(w * self._roi_x_start), int(w * self._roi_x_end)
        roi = frame[y1:y2, x1:x2]

        # ── [9] Early rejection — skip dark / empty frames ────────────
        # Computing mean brightness is ~0.1ms, saves ~15ms if we skip.
        if roi.mean() < self._min_brightness:
            result.inference_ms = (time.monotonic() - t_start) * 1000
            return result

        # ── [3][7] Resize ROI directly to model input size ────────────
        # No intermediate resize — ROI goes straight to 96x96.
        input_img = cv2.resize(roi, (self.input_size, self.input_size))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        # ── [2] Correct MobileNetV2 preprocessing ─────────────────────
        # MobileNetV2 expects input scaled to [-1.0, 1.0]:
        #   preprocess_input(x) = (x / 127.5) - 1.0
        # We implement this manually to avoid importing full Keras on RPi.
        input_dtype = self._input_details[0]['dtype']
        if input_dtype == np.uint8:
            # INT8 quantized model — feed raw uint8 (0-255).
            # The quantization parameters handle the scaling internally.
            input_data = np.expand_dims(input_img.astype(np.uint8), axis=0)
        else:
            # Float model — apply MobileNetV2 preprocessing: [-1, 1]
            preprocessed = (input_img.astype(np.float32) / 127.5) - 1.0
            input_data = np.expand_dims(preprocessed, axis=0)

        # ── Run TFLite inference ──────────────────────────────────────
        self._interpreter.set_tensor(
            self._input_details[0]['index'], input_data)
        self._interpreter.invoke()

        # Get output — softmax probabilities (or logits)
        output = self._interpreter.get_tensor(
            self._output_details[0]['index'])[0]

        # Handle INT8 output (dequantize to float)
        output_dtype = self._output_details[0]['dtype']
        if output_dtype in (np.uint8, np.int8):
            quant_params = self._output_details[0].get(
                'quantization_parameters', {})
            scales = quant_params.get('scales', None)
            zero_points = quant_params.get('zero_points', None)
            if scales is not None and len(scales) > 0:
                output = (output.astype(np.float32) - zero_points[0]) * scales[0]
            else:
                # Fallback to legacy quantization tuple
                quant = self._output_details[0].get('quantization', (1.0, 0))
                if quant and len(quant) >= 2:
                    scale, zero_point = quant
                    output = (output.astype(np.float32) - zero_point) * scale

        # Apply softmax if output isn't already probabilities
        if np.any(output < 0) or np.sum(output) < 0.9 or np.sum(output) > 1.1:
            exp_output = np.exp(output - np.max(output))
            output = exp_output / np.sum(exp_output)

        # ── [4] Confidence threshold filtering ────────────────────────
        class_id = int(np.argmax(output))
        confidence = float(output[class_id])

        # ── [8] Stop inference timer ──────────────────────────────────
        t_end = time.monotonic()
        result.inference_ms = (t_end - t_start) * 1000
        fps = 1.0 / (t_end - t_start) if (t_end - t_start) > 0 else 0
        log.debug(f"TSR inference: {result.inference_ms:.1f}ms "
                  f"({fps:.0f} FPS potential)")

        if confidence < self.conf_threshold:
            # [4] Weak prediction — don't trust it, don't add to buffer
            return result

        # ── [5] Temporal smoothing — majority voting ──────────────────
        # Add this prediction to the buffer, then pick the most common.
        self._prediction_buffer.append(class_id)

        # Need at least 2 predictions in buffer to vote
        if len(self._prediction_buffer) < 2:
            return result

        # Count occurrences of each class in the buffer
        vote_counts = Counter(self._prediction_buffer)
        voted_class_id, voted_count = vote_counts.most_common(1)[0]

        # ── [10] Stable output gating ─────────────────────────────────
        # Only accept if the majority winner has ≥2 votes out of 5.
        # This means the sign must appear in at least 2 recent frames.
        if voted_count < 2:
            return result

        # All gates passed — this is a stable, confident detection
        result.sign_detected = True
        result.class_id = voted_class_id
        result.class_name = self.sign_names.get(
            voted_class_id, f"Unknown ({voted_class_id})")
        result.confidence = confidence

        # Check if it's a speed limit sign
        if voted_class_id in self.speed_limits:
            result.speed_limit = self.speed_limits[voted_class_id]

        # Check for stop sign (class 14 in GTSRB)
        if voted_class_id == 14:
            result.is_stop_sign = True

        log.debug(f"TSR: {result.class_name} ({confidence:.2f}), "
                  f"votes={voted_count}/5, {result.inference_ms:.1f}ms")

        # Cache as last stable result for frame-skip returns
        self._last_stable_result = result

        return result

    @property
    def is_loaded(self):
        """Whether the model is loaded and ready."""
        return self._loaded
