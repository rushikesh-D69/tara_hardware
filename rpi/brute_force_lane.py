#!/usr/bin/env python3
"""
TARA — Brute-Force Lane Follower (No ML)
=========================================
Replaces the full ADAS pipeline with simple pixel-counting logic for reliable
track completion on a black-surface / white-tape indoor course.

Algorithm:
  1. Capture frame → grayscale → adaptive threshold → crop bottom half
  2. Divide into LEFT / CENTER / RIGHT regions
  3. Count white pixels per region → decide STRAIGHT / LEFT / RIGHT
  4. If total white pixels < threshold → lane lost → TURN using last_direction

Usage:
  python3 brute_force_lane.py               # Normal
  python3 brute_force_lane.py --debug       # MJPEG debug stream at :5000
  python3 brute_force_lane.py --video x.mp4 # Test with video file
"""

import os
import sys
import time
import argparse
import signal
import threading
import cv2
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from camera.capture import CameraCapture

# ── Tunable Parameters ────────────────────────────────────────────────────────
BINARY_THRESHOLD    = 175
CROP_RATIO          = 0.6
LANE_LOST_THRESHOLD = 1200

STEER_STRAIGHT   =  0.0
STEER_LEFT       = -0.4
STEER_RIGHT      =  0.4
STEER_HARD_LEFT  = -0.9
STEER_HARD_RIGHT =  0.9

CRUISE_SPEED     = 0.45
TURN_SPEED       = 0.35
TURN_HOLD_TIME   = 1.5

CENTER_DOMINANCE  = 0.35
BALANCE_TOLERANCE = 0.15

LEFT_END    = 0.33
RIGHT_START = 0.66

STARTUP_SKIP_FRAMES = 10

# ── MJPEG Debug Stream ────────────────────────────────────────────────────────
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
    t = threading.Thread(target=srv.serve_forever, daemon=True, name='MJPEGDbg')
    t.start()
    return srv


def _push_mjpeg(bgr):
    global _mjpeg_frame
    ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if ok:
        with _mjpeg_lock:
            _mjpeg_frame = buf.tobytes()


# ── Core Logic ────────────────────────────────────────────────────────────────

class BruteForceLaneFollower:
    """
    Dead-simple lane follower using binary thresholding + region pixel counts.
    No ML, no PID, no polynomial fitting — just count white pixels and steer.
    """

    def __init__(self, args):
        self.args    = args
        self.running = False

        self.last_direction  = "LEFT"
        self.turn_active     = False
        self.turn_start_time = 0.0
        self.frame_num       = 0

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

        self._mjpeg = None
        if args.debug:
            self._mjpeg = _start_mjpeg_server()
            print(f"[DEBUG] Stream at http://<RPI_IP>:{_MJPEG_PORT}/")

    def start(self):
        """Open camera and begin processing."""
        print("=" * 55)
        print("  TARA — Brute-Force Lane Follower")
        print("  Track: Black surface + white lanes")
        print("  Logic: Threshold → Region count → Steer")
        print("=" * 55)

        try:
            self.camera.start()
        except RuntimeError as e:
            print(f"[ERROR] Camera failed: {e}")
            return False

        self.running = True
        print("[OK] Pipeline running. Press Ctrl+C to stop.\n")
        return True

    def run(self):
        """Main loop."""
        if not self.start():
            return

        try:
            while self.running:
                self._process_frame()
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        finally:
            self.stop()

    def _process_frame(self):
        """Process one frame through the brute-force pipeline."""
        frame = self.camera.read()
        if frame is None:
            time.sleep(0.01)
            return

        self.frame_num += 1

        if self.frame_num <= STARTUP_SKIP_FRAMES:
            if self.frame_num == STARTUP_SKIP_FRAMES:
                print(f"[STARTUP] Skipped {STARTUP_SKIP_FRAMES} frames for camera warmup")
            return

        t0 = time.monotonic()

        # Step 1: Preprocess
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray   = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            51, -20,
        )

        h, w = binary.shape
        crop_start = int(h * (1.0 - CROP_RATIO))
        roi = binary[crop_start:h, :]

        # Step 2: Region pixel counts
        roi_h, roi_w   = roi.shape
        left_boundary  = int(roi_w * LEFT_END)
        right_boundary = int(roi_w * RIGHT_START)

        left_region   = roi[:, 0:left_boundary]
        center_region = roi[:, left_boundary:right_boundary]
        right_region  = roi[:, right_boundary:roi_w]

        left_count   = cv2.countNonZero(left_region)
        center_count = cv2.countNonZero(center_region)
        right_count  = cv2.countNonZero(right_region)
        total_count  = left_count + center_count + right_count

        # Step 3: Decision logic
        command = "STRAIGHT"
        steer   = STEER_STRAIGHT
        speed   = CRUISE_SPEED
        now     = time.monotonic()

        if self.turn_active:
            elapsed = now - self.turn_start_time
            if elapsed < TURN_HOLD_TIME:
                command = f"TURN_{self.last_direction}"
                steer   = STEER_HARD_LEFT if self.last_direction == "LEFT" else STEER_HARD_RIGHT
                speed   = TURN_SPEED
            else:
                self.turn_active = False

        if not self.turn_active:
            if total_count < LANE_LOST_THRESHOLD:
                command              = f"TURN_{self.last_direction}"
                steer                = STEER_HARD_LEFT if self.last_direction == "LEFT" else STEER_HARD_RIGHT
                speed                = TURN_SPEED
                self.turn_active     = True
                self.turn_start_time = now

            elif total_count > 0:
                balance = (left_count - right_count) / total_count

                if abs(balance) < BALANCE_TOLERANCE:
                    command = "STRAIGHT"
                    steer   = STEER_STRAIGHT
                    speed   = CRUISE_SPEED
                elif balance > 0:
                    command              = "CENTER_RIGHT"
                    steer                = STEER_RIGHT
                    speed                = CRUISE_SPEED
                    self.last_direction  = "RIGHT"
                else:
                    command              = "CENTER_LEFT"
                    steer                = STEER_LEFT
                    speed                = CRUISE_SPEED
                    self.last_direction  = "LEFT"

            else:
                command = "SEARCH"
                steer   = STEER_LEFT * 0.5 if self.last_direction == "LEFT" else STEER_RIGHT * 0.5
                speed   = CRUISE_SPEED

        elapsed_ms = (time.monotonic() - t0) * 1000
        print(
            f"[F{self.frame_num:05d}] {command:12s} | "
            f"L:{left_count:5d}  C:{center_count:5d}  R:{right_count:5d}  "
            f"Total:{total_count:6d} | "
            f"Steer:{steer:+.2f}  Speed:{speed:.2f} | "
            f"{elapsed_ms:.1f}ms"
        )

        if self.args.debug:
            self._draw_debug(frame, roi, left_boundary, right_boundary,
                             crop_start, left_count, center_count, right_count,
                             total_count, command, steer, speed)

    def _draw_debug(self, frame, roi, left_b, right_b, crop_start,
                    left_c, center_c, right_c, total_c, command, steer, speed):
        """Compose debug overlay and push to MJPEG stream."""
        display = frame.copy()
        h, w    = display.shape[:2]

        cv2.line(display, (left_b,  crop_start), (left_b,  h), (0, 255, 255), 2)
        cv2.line(display, (right_b, crop_start), (right_b, h), (0, 255, 255), 2)
        cv2.line(display, (0, crop_start), (w, crop_start), (255, 0, 255), 2)

        overlay = display.copy()
        cv2.rectangle(overlay, (0,       crop_start), (left_b,  h), (255, 100, 0), -1)
        cv2.rectangle(overlay, (left_b,  crop_start), (right_b, h), (0, 255,   0), -1)
        cv2.rectangle(overlay, (right_b, crop_start), (w,       h), (0,   0, 255), -1)
        display = cv2.addWeighted(overlay, 0.25, display, 0.75, 0)

        panel_lines = [
            f"CMD: {command}",
            f"Steer: {steer:+.2f}  Speed: {speed:.2f}",
            f"L:{left_c:5d} C:{center_c:5d} R:{right_c:5d}",
            f"Total: {total_c}  Threshold: {LANE_LOST_THRESHOLD}",
            f"Last Dir: {self.last_direction}",
            f"Turn Active: {self.turn_active}",
            f"Frame: {self.frame_num}",
        ]

        cv2.rectangle(display, (5, 5), (320, 15 + len(panel_lines) * 22), (0, 0, 0), -1)

        for i, line in enumerate(panel_lines):
            color = (0, 255, 0)
            if "TURN" in line:
                color = (0, 100, 255)
            elif "LOST" in command:
                color = (0, 0, 255)
            cv2.putText(display, line, (10, 25 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        roi_color = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        roi_small = cv2.resize(roi_color, (w // 3, roi.shape[0] // 2))
        rh, rw    = roi_small.shape[:2]
        display[h - rh:h, 0:rw] = roi_small

        _push_mjpeg(display)

    def stop(self):
        """Graceful shutdown."""
        print("\n[INFO] Stopping brute-force lane follower...")
        self.running = False
        self.camera.stop()
        print("=" * 55)
        print("  TARA — Brute-Force Lane Follower STOPPED")
        print(f"  Total frames processed: {self.frame_num}")
        print("=" * 55)


# ── Entry Point ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="TARA — Brute-Force Lane Follower (No ML)")
    parser.add_argument("--debug", action="store_true",
                        help="Stream debug view at http://RPI_IP:5000/")
    parser.add_argument("--video", type=str, default=None,
                        help="Use video file instead of live camera")
    return parser.parse_args()


def main():
    args     = parse_args()
    follower = BruteForceLaneFollower(args)

    def signal_handler(sig, _frame):
        follower.running = False

    signal.signal(signal.SIGINT,  signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    follower.run()
    sys.exit(0)


if __name__ == "__main__":
    main()
