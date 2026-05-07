"""
TARA ADAS — Main Pipeline (Raspberry Pi 4B)
Time-multiplexed ADAS scheduler running all ML inference locally.

Frame schedule (4-frame cycle):
  Frame 0: Lane + ACC
  Frame 1: Lane + TSR
  Frame 2: Lane + ACC + TLR
  Frame 3: Lane + Pothole

Usage:
  python3 main.py                 # Normal mode
  python3 main.py --debug         # MJPEG debug stream at http://<RPI_IP>:5000/
  python3 main.py --no-wifi       # Vision-only (no ESP32 connection)
  python3 main.py --no-cloud      # Offline mode
  python3 main.py --video x.mp4   # Use video file instead of camera
"""
import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

import sys
import time
import argparse
import signal
import threading
import cv2
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from camera.capture import CameraCapture
from adas.lane_detection import LaneDetector
from adas.sign_detector_cv import SignDetectorCV
from adas.traffic_sign import TrafficSignRecognizer
from adas.pothole_detection import PotholeDetector
from adas.adaptive_cruise import AdaptiveCruiseControl
from adas.traffic_light import TrafficLightDetector
from adas.decision_manager import DecisionManager
from comms.ws_bridge import WsBridge
from utils.fps_counter import FPSCounter
from utils.logger import setup_logger, get_logger

# ── MJPEG Stream Server ───────────────────────────────────────────────────────
_mjpeg_frame = None
_mjpeg_lock  = threading.Lock()
_MJPEG_PORT  = 5000


class _MjpegHandler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(
                b'<html><body style="margin:0;background:#000">'
                b'<img src="/stream" style="width:100%;max-width:800px">'
                b'</body></html>'
            )
        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type',
                             'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    with _mjpeg_lock:
                        jpg = _mjpeg_frame
                    if jpg:
                        self.wfile.write(
                            b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                            + jpg + b'\r\n')
                    time.sleep(0.04)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


class _ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True


def _start_mjpeg_server():
    srv = _ReusableHTTPServer(('0.0.0.0', _MJPEG_PORT), _MjpegHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True, name='MJPEGServer')
    t.start()
    return srv


def _push_mjpeg_frame(bgr_frame):
    global _mjpeg_frame
    ok, buf = cv2.imencode('.jpg', bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if ok:
        with _mjpeg_lock:
            _mjpeg_frame = buf.tobytes()


# ── Argument Parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="TARA ADAS — RPi Pipeline")
    parser.add_argument("--debug", action="store_true",
                        help="Stream annotated video at http://RPI_IP:5000/")
    parser.add_argument("--no-wifi", action="store_true",
                        help="Run without ESP32 WiFi connection (vision-only)")
    parser.add_argument("--video", type=str, default=None,
                        help="Use video file instead of live camera")
    parser.add_argument("--log-level", type=str,
                        default=config.LOG_LEVEL,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    return parser.parse_args()


# ── Main ADAS Coordinator ─────────────────────────────────────────────────────

class TARAAdas:
    """
    Main ADAS pipeline coordinator.
    Owns all module instances and runs the time-multiplexed scheduling loop.
    """

    def __init__(self, args):
        self.args    = args
        self.running = False

        setup_logger("TARA", level=args.log_level, log_file=config.LOG_FILE)
        self.log = get_logger("Main")

        self.frame_num = 0

        self.log.info("=" * 50)
        self.log.info("  TARA ADAS — Initializing")
        self.log.info("=" * 50)

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

        self.lane_detector    = LaneDetector(config)
        self.sign_detector    = SignDetectorCV(config)
        self.tsr              = TrafficSignRecognizer(config)
        self.pothole_detector = PotholeDetector(config)
        self.acc              = AdaptiveCruiseControl(config)
        self.tl_detector      = TrafficLightDetector(config)
        self.decision_manager = DecisionManager(config)

        self.ws = None
        if not args.no_wifi:
            self.ws = WsBridge(
                host=config.ESP32_HOST,
                port=config.ESP32_WS_PORT,
                path=config.ESP32_WS_PATH,
            )

        self._mjpeg_server = None
        if args.debug:
            self._mjpeg_server = _start_mjpeg_server()
            self.log.info(f"Debug stream: http://<RPI_IP>:{_MJPEG_PORT}/")

        self.fps = FPSCounter(window_size=30)

        self._last_lane      = None
        self._last_sign_hint = None
        self._last_tsr       = None
        self._last_pothole   = None
        self._last_acc       = None
        self._last_tl        = None
        self._last_frame     = None

    def start(self):
        """Open hardware, load models, connect to ESP32."""
        self.log.info("Starting TARA ADAS pipeline...")

        try:
            self.camera.start()
        except RuntimeError as e:
            self.log.error(f"Camera failed: {e}")
            return False

        tsr_ok = self.tsr.load_model()
        if not tsr_ok:
            self.log.warning("TSR model not loaded — traffic sign recognition disabled")

        pothole_ok = self.pothole_detector.load_model()
        if not pothole_ok:
            self.log.warning("Pothole model not loaded — pothole detection disabled")

        if self.ws:
            if not self.ws.connect():
                self.log.warning("ESP32 WiFi not connected — running in vision-only mode")
                self.ws = None

        self.running = True
        self.log.info("=" * 50)
        self.log.info("  TARA ADAS — RUNNING")
        mode_parts = ["Edge ML"]
        if self.ws:
            mode_parts.append(f"ESP32 WiFi ({config.ESP32_HOST})")
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
        frame = self.camera.read()
        if frame is None:
            time.sleep(0.01)
            return

        self.fps.tick()
        cycle_pos = self.frame_num % 4

        t = self.fps.start_module("Lane")
        self._last_lane = self.lane_detector.detect(frame, debug=self.args.debug)
        self.fps.stop_module(t)

        if cycle_pos == config.SCHEDULE_TSR_OFFSET % 4:
            t = self.fps.start_module("TSR")
            self._last_tsr = self.tsr.detect(frame)
            self.fps.stop_module(t)

            if (self._last_tsr and self._last_tsr.sign_detected
                    and self._last_tsr.speed_limit is not None):
                self.acc.set_speed_limit(self._last_tsr.speed_limit)

        if cycle_pos == config.SCHEDULE_POTHOLE_OFFSET % 4:
            t = self.fps.start_module("Pothole")
            self._last_pothole = self.pothole_detector.detect(frame)
            self.fps.stop_module(t)

        if cycle_pos % config.SCHEDULE_ACC_EVERY == 0:
            t = self.fps.start_module("ACC")
            self._last_acc = self.acc.update()
            self.fps.stop_module(t)

        if cycle_pos % 2 == 0:
            t = self.fps.start_module("TLR")
            self._last_tl = self.tl_detector.detect(frame)
            self.fps.stop_module(t)

        if self.frame_num % 2 == 0:
            t = self.fps.start_module("SignCV")
            bev_frame = self.lane_detector.warp_bev(frame)
            self._last_sign_hint = self.sign_detector.detect(bev_frame)
            self.fps.stop_module(t)

        t = self.fps.start_module("Decision")
        command = self.decision_manager.update(
            lane_result=self._last_lane,
            sign_hint=self._last_sign_hint,
            tsr_result=self._last_tsr,
            pothole_result=self._last_pothole,
            acc_result=self._last_acc,
            tl_result=self._last_tl,
        )
        self.fps.stop_module(t)

        if self.args.debug:
            self._show_debug(frame, command)

        if config.LOG_FPS and self.frame_num % 30 == 0:
            self.log.info(self.fps.summary())

        self._last_frame = frame
        self.frame_num += 1

    def _show_debug(self, frame, command):
        """Compose annotated debug frame and push to the MJPEG server."""
        display = frame.copy()

        if self._last_lane and self._last_lane.debug_frame is not None:
            debug_small = self._last_lane.debug_frame
            dh, dw = debug_small.shape[:2]
            display[0:dh, 0:dw] = debug_small

        panel_x = display.shape[1] - 220
        panel_y = 10

        overlay = display.copy()
        cv2.rectangle(overlay, (panel_x - 10, panel_y - 5),
                      (display.shape[1] - 5, panel_y + 200),
                      (0, 0, 0), -1)
        display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)

        texts = [
            f"FPS: {self.fps.fps():.1f}",
            f"Frame: {self.frame_num}",
            "---",
            f"CMD: steer={command.steering} spd={command.speed}",
            "---",
        ]

        if self._last_lane:
            lane_status = "DETECTED" if self._last_lane.lane_detected else "LOST"
            texts.append(f"Lane: {lane_status}")
            texts.append(f"  Offset: {self._last_lane.lane_center_offset:.1f}px")
            if self._last_lane.departure_warning:
                texts.append("  !! LDW WARNING !!")

        if self._last_tsr and self._last_tsr.sign_detected:
            texts.append(f"TSR: {self._last_tsr.class_name}")
            texts.append(f"  Conf: {self._last_tsr.confidence:.2f}")
        else:
            texts.append("TSR: --")

        if self._last_pothole and self._last_pothole.pothole_detected:
            texts.append(f"POTHOLE: {self._last_pothole.position}")
        else:
            texts.append("Pothole: clear")

        if self._last_acc:
            texts.append(f"ACC: {self._last_acc.mode}")
            texts.append(f"  Speed: {self._last_acc.speed_norm:.2f}")

        if self._last_tl and self._last_tl.detected:
            texts.append(f"TL: {self._last_tl.state}")
            texts.append(f"  Conf: {self._last_tl.confidence:.3f}")
        else:
            texts.append("TL: --")

        for i, text in enumerate(texts):
            color = (0, 255, 0)
            if "WARNING" in text or "POTHOLE" in text:
                color = (0, 0, 255)
            elif "LOST" in text:
                color = (0, 100, 255)
            cv2.putText(display, text, (panel_x, panel_y + 15 + i * 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        _push_mjpeg_frame(display)

    def stop(self):
        """Gracefully shut down all modules."""
        self.log.info("Stopping TARA ADAS...")
        self.running = False

        if self.ws:
            self.ws.send_stop()
            self.ws.disconnect()

        self.camera.stop()

        self.log.info("=" * 50)
        self.log.info("  TARA ADAS — STOPPED")
        self.log.info(f"  Total frames processed: {self.frame_num}")
        self.log.info("=" * 50)


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    adas = TARAAdas(args)

    def signal_handler(sig, _frame):
        adas.running = False

    signal.signal(signal.SIGINT,  signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    adas.run()
    sys.exit(0)


if __name__ == "__main__":
    main()
