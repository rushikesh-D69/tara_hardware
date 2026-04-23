"""
TARA ADAS — Main Pipeline  (Raspberry Pi 4B Edition)
Time-multiplexed ADAS scheduler — all ML runs locally, no cloud required.

Schedule (per frame):
  Frame 0: Lane + ACC
  Frame 1: Lane + TSR
  Frame 2: Lane + ACC + TLR
  Frame 3: Lane + Pothole

Usage:
  python3 main.py                 # Normal mode
  python3 main.py --debug         # With OpenCV window overlay
  python3 main.py --no-serial     # Without ESP32 (vision-only testing)
  python3 main.py --no-cloud      # Offline mode
  python3 main.py --video x.mp4   # Use video file instead of camera
"""
# ── Suppress TensorFlow / oneDNN noise BEFORE any TF import ───────────────────
# Must be set before 'import tensorflow' or 'import tflite_runtime' is called.
import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")   # kill oneDNN warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")    # suppress C++ TF logs
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")      # no GPU on RPi
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")  # skip MSMF on RPi

import sys
import time
import argparse
import signal
import cv2

# ── Ensure the rpi/ directory is on the import path ───────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from camera.capture import CameraCapture
from adas.lane_detection import LaneDetector
from adas.traffic_sign import TrafficSignRecognizer
from adas.pothole_detection import PotholeDetector
from adas.adaptive_cruise import AdaptiveCruiseControl
from adas.traffic_light import TrafficLightDetector
from adas.decision_manager import DecisionManager
from comms.serial_bridge import SerialBridge
from cloud.firebase_logger import FirebaseLogger, LocalSessionRecorder
from utils.fps_counter import FPSCounter
from utils.logger import setup_logger, get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="TARA ADAS — RPi Pipeline")
    parser.add_argument("--debug", action="store_true",
                        help="Show OpenCV debug window")
    parser.add_argument("--no-serial", action="store_true",
                        help="Run without ESP32 serial connection")
    parser.add_argument("--no-cloud", action="store_true",
                        help="Disable cloud logging (fully offline)")
    parser.add_argument("--video", type=str, default=None,
                        help="Use video file instead of live camera")
    parser.add_argument("--log-level", type=str,
                        default=config.LOG_LEVEL,          # reads from config.py
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    return parser.parse_args()



class TARAAdas:
    """
    Main ADAS pipeline coordinator.
    Manages all modules and the time-multiplexed scheduling.
    """

    def __init__(self, args):
        self.args = args
        self.running = False

        # Setup logging
        setup_logger("TARA", level=args.log_level, log_file=config.LOG_FILE)
        self.log = get_logger("Main")

        # Frame counter for scheduling
        self.frame_num = 0

        # Initialize modules
        self.log.info("=" * 50)
        self.log.info("  TARA ADAS — Initializing")
        self.log.info("=" * 50)

        # Camera
        if args.video:
            self.camera = CameraCapture(
                index=args.video,
                width=config.FRAME_WIDTH,
                height=config.FRAME_HEIGHT,
            )
        else:
            self.camera = CameraCapture(
                index=config.CAMERA_INDEX,
                width=config.FRAME_WIDTH,
                height=config.FRAME_HEIGHT,
                fps=config.CAMERA_FPS,
            )

        # ADAS modules
        self.lane_detector = LaneDetector(config)
        self.tsr = TrafficSignRecognizer(config)
        self.pothole_detector = PotholeDetector(config)
        self.acc = AdaptiveCruiseControl(config)
        self.tl_detector = TrafficLightDetector(config)
        self.decision_manager = DecisionManager(config)

        # Communication
        self.serial = None
        if not args.no_serial:
            self.serial = SerialBridge(
                port=config.SERIAL_PORT,
                baud=config.BAUD_RATE,
                timeout=config.SERIAL_TIMEOUT,
            )

        # Cloud logger (async, non-blocking)
        self.cloud = None
        if not args.no_cloud and getattr(config, 'CLOUD_ENABLED', False):
            self.cloud = FirebaseLogger(config)

        # Local session recorder (always-on fallback)
        self.local_recorder = None
        if getattr(config, 'LOCAL_RECORDING_ENABLED', True):
            self.local_recorder = LocalSessionRecorder(config)

        # Performance tracking
        self.fps = FPSCounter(window_size=30)

        # Latest results (persist between frames for non-scheduled modules)
        self._last_lane = None
        self._last_tsr = None
        self._last_pothole = None
        self._last_acc = None
        self._last_tl = None
        self._last_frame = None  # Keep reference for event snapshots

    def start(self):
        """Initialize all hardware and start the ADAS pipeline."""
        self.log.info("Starting TARA ADAS pipeline...")

        # Start camera
        try:
            self.camera.start()
        except RuntimeError as e:
            self.log.error(f"Camera failed: {e}")
            return False

        # Load ML models
        tsr_ok = self.tsr.load_model()
        if not tsr_ok:
            self.log.warning("TSR model not loaded — traffic sign recognition disabled")

        pothole_ok = self.pothole_detector.load_model()
        if not pothole_ok:
            self.log.warning("Pothole model not loaded — pothole detection disabled")

        # Connect to ESP32
        if self.serial:
            if not self.serial.connect():
                self.log.warning("ESP32 not connected — running in vision-only mode")
                self.serial = None

        # Start cloud logger
        if self.cloud:
            if not self.cloud.connect():
                self.log.info("Cloud logging unavailable — running offline")
                self.cloud = None

        # Start local recorder
        if self.local_recorder:
            self.local_recorder.start()

        self.running = True
        self.log.info("=" * 50)
        self.log.info("  TARA ADAS — RUNNING")
        mode_parts = ["Edge ML"]
        if self.cloud and self.cloud.is_enabled:
            mode_parts.append("Cloud logging")
        if self.local_recorder and self.local_recorder.is_enabled:
            mode_parts.append("Local recording")
        if self.serial:
            mode_parts.append("ESP32 connected")
        self.log.info(f"  Mode: {' + '.join(mode_parts)}")
        self.log.info("=" * 50)

        return True

    def run(self):
        """Main processing loop."""
        if not self.start():
            self.log.error("Startup failed. Exiting.")
            return

        try:
            while self.running:
                self._process_frame()
        except KeyboardInterrupt:
            self.log.info("Interrupted by user")
        finally:
            self.stop()

    def _process_frame(self):
        """Process a single frame through the scheduled pipeline."""
        # Grab latest frame
        frame = self.camera.read()
        if frame is None:
            time.sleep(0.01)
            return

        self.fps.tick()
        cycle_pos = self.frame_num % 4  # 4-frame cycle

        # ── Always run: Lane Detection ─────────────────────────────────
        t = self.fps.start_module("Lane")
        self._last_lane = self.lane_detector.detect(frame, debug=self.args.debug)
        self.fps.stop_module(t)

        # ── Scheduled: TSR (frame 1, 5, 9, ...) ───────────────────────
        if cycle_pos == config.SCHEDULE_TSR_OFFSET % 4:
            t = self.fps.start_module("TSR")
            self._last_tsr = self.tsr.detect(frame)
            self.fps.stop_module(t)

            # Update ACC with TSR speed limit
            if self._last_tsr and self._last_tsr.sign_detected and self._last_tsr.speed_limit is not None:
                self.acc.set_speed_limit(self._last_tsr.speed_limit)

        # ── Scheduled: Pothole (frame 3, 7, 11, ...) ──────────────────
        if cycle_pos == config.SCHEDULE_POTHOLE_OFFSET % 4:
            t = self.fps.start_module("Pothole")
            self._last_pothole = self.pothole_detector.detect(frame)
            self.fps.stop_module(t)

        # ── Scheduled: ACC Sensor Read (frames 0, 2) ──────────────────
        if cycle_pos % config.SCHEDULE_ACC_EVERY == 0:
            t = self.fps.start_module("ACC")
            sensor_data = None
            if self.serial:
                sensor_data = self.serial.get_sensor_data()

            if sensor_data:
                self._last_acc = self.acc.update(sensor_data)
            else:
                # No sensor data — use safe defaults
                self._last_acc = self.acc.update({
                    'distance_cm': 999, 'left_enc': 0, 'right_enc': 0,
                    'accel_x': 0, 'accel_y': 0, 'accel_z': 9.81,
                })
            self.fps.stop_module(t)

        # ── Scheduled: Traffic Light (frames 0, 2) ─────────────────────
        if cycle_pos % 2 == 0:
            t = self.fps.start_module("TLR")
            self._last_tl = self.tl_detector.detect(frame)
            self.fps.stop_module(t)

        # ── Always run: Decision Manager ───────────────────────────────
        t = self.fps.start_module("Decision")
        sensor_data = self.serial.get_sensor_data() if self.serial else None
        
        command = self.decision_manager.update(
            lane_result=self._last_lane,
            tsr_result=self._last_tsr,
            pothole_result=self._last_pothole,
            acc_result=self._last_acc,
            tl_result=self._last_tl,
            sensor_data=sensor_data,
        )
        self.fps.stop_module(t)

        # ── Send command to ESP32 ──────────────────────────────────────
        if self.serial:
            self.serial.send_command(command)

        # ── Cloud + local data logging (async, non-blocking) ──────────
        self._log_data(frame, command)

        # ── Debug visualization ────────────────────────────────────────
        if self.args.debug:
            self._show_debug(frame, command)

        # ── Console output ─────────────────────────────────────────────
        if config.LOG_FPS and self.frame_num % 30 == 0:
            self.log.info(self.fps.summary())

        self._last_frame = frame
        self.frame_num += 1

    def _show_debug(self, frame, command):
        """Show debug visualization window."""
        display = frame.copy()

        # Overlay lane detection debug if available
        if self._last_lane and self._last_lane.debug_frame is not None:
            debug_small = self._last_lane.debug_frame
            # Place lane debug in top-left corner
            dh, dw = debug_small.shape[:2]
            display[0:dh, 0:dw] = debug_small

        # Draw status panel on the right
        panel_x = display.shape[1] - 220
        panel_y = 10

        # Background panel
        overlay = display.copy()
        cv2.rectangle(overlay, (panel_x - 10, panel_y - 5),
                      (display.shape[1] - 5, panel_y + 200),
                      (0, 0, 0), -1)
        display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)

        # Status lines
        texts = [
            f"FPS: {self.fps.fps():.1f}",
            f"Frame: {self.frame_num}",
            "---",
            f"CMD: steer={command.steering} spd={command.speed}",
            "---",
        ]

        # Lane status
        if self._last_lane:
            lane_status = "DETECTED" if self._last_lane.lane_detected else "LOST"
            texts.append(f"Lane: {lane_status}")
            texts.append(f"  Offset: {self._last_lane.lane_center_offset:.1f}px")
            if self._last_lane.departure_warning:
                texts.append("  !! LDW WARNING !!")

        # TSR status
        if self._last_tsr and self._last_tsr.sign_detected:
            texts.append(f"TSR: {self._last_tsr.class_name}")
            texts.append(f"  Conf: {self._last_tsr.confidence:.2f}")
        else:
            texts.append("TSR: --")

        # Pothole status
        if self._last_pothole and self._last_pothole.pothole_detected:
            texts.append(f"POTHOLE: {self._last_pothole.position}")
        else:
            texts.append("Pothole: clear")

        # ACC status
        if self._last_acc:
            texts.append(f"ACC: {self._last_acc.mode}")
            texts.append(f"  Dist: {self._last_acc.distance_cm:.0f}cm")

        # TL status
        if self._last_tl and self._last_tl.detected:
            texts.append(f"TL: {self._last_tl.state}")
            texts.append(f"  Conf: {self._last_tl.confidence:.3f}")
        else:
            texts.append("TL: --")

        # Render text
        for i, text in enumerate(texts):
            color = (0, 255, 0)
            if "WARNING" in text or "POTHOLE" in text:
                color = (0, 0, 255)
            elif "LOST" in text or "E_STOP" in text:
                color = (0, 100, 255)
            cv2.putText(display, text, (panel_x, panel_y + 15 + i * 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        cv2.imshow("TARA ADAS", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.running = False

    def _log_data(self, frame, command):
        """
        Push telemetry + detection events to cloud and local recorder.
        All calls here are non-blocking — they either queue data
        for a background thread (cloud) or do a quick file write (local).
        """
        current_fps = self.fps.fps()

        # Telemetry (throttled internally to every N seconds)
        if self.cloud:
            self.cloud.log_telemetry(
                current_fps, command,
                acc_result=self._last_acc,
                lane_result=self._last_lane,
            )
        if self.local_recorder:
            self.local_recorder.log_telemetry(
                current_fps, command,
                acc_result=self._last_acc,
                lane_result=self._last_lane,
            )

        # Detection events — snapshot frame only on actual detections
        # (not every frame, so upload volume stays tiny)
        if self._last_tsr and self._last_tsr.sign_detected:
            meta = {
                'class': self._last_tsr.class_name,
                'confidence': round(self._last_tsr.confidence, 3),
            }
            if self.cloud:
                self.cloud.log_event(frame, 'sign', meta)
            if self.local_recorder:
                self.local_recorder.log_event(frame, 'sign', meta)

        if self._last_pothole and self._last_pothole.pothole_detected:
            meta = {
                'position': self._last_pothole.position,
                'confidence': round(self._last_pothole.confidence, 3),
            }
            if self.cloud:
                self.cloud.log_event(frame, 'pothole', meta)
            if self.local_recorder:
                self.local_recorder.log_event(frame, 'pothole', meta)

        if (self._last_lane and self._last_lane.departure_warning
                and self.frame_num % 15 == 0):  # Throttle departure events
            if self.cloud:
                self.cloud.log_event(frame, 'departure', {
                    'offset': round(self._last_lane.lane_center_offset, 1),
                })
            if self.local_recorder:
                self.local_recorder.log_event(frame, 'departure')

        if self._last_acc and self._last_acc.emergency_stop:
            if self.cloud:
                self.cloud.log_event(frame, 'emergency_stop', {
                    'distance_cm': round(self._last_acc.distance_cm, 1),
                })
            if self.local_recorder:
                self.local_recorder.log_event(frame, 'emergency_stop')

    def stop(self):
        """Gracefully shut down all modules."""
        self.log.info("Stopping TARA ADAS...")
        self.running = False

        # Send stop command to ESP32
        if self.serial:
            self.serial.send_stop()
            self.serial.disconnect()

        # Stop cloud logger (flushes remaining uploads)
        if self.cloud:
            self.cloud.stop()

        # Stop local recorder
        if self.local_recorder:
            self.local_recorder.stop()

        # Stop camera
        self.camera.stop()

        # Close debug window
        if self.args.debug:
            cv2.destroyAllWindows()

        self.log.info("=" * 50)
        self.log.info("  TARA ADAS — STOPPED")
        self.log.info(f"  Total frames processed: {self.frame_num}")
        self.log.info("=" * 50)


def main():
    args = parse_args()
    adas = TARAAdas(args)

    def signal_handler(sig, _frame):
        # Set flag so the run loop exits naturally, then raise to unblock any sleep
        adas.running = False

    signal.signal(signal.SIGINT,  signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    adas.run()
    sys.exit(0)   # explicit clean exit — prevents exit code 1 from signal


if __name__ == "__main__":
    main()
