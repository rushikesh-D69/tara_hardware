#!/usr/bin/env python3
"""
TARA — Brute-Force Lane Follower (No ML)
=========================================
Replaces the entire ADAS ML pipeline with simple pixel-counting logic.

Track: Black surface, white lane markings, sharp 90° turns.
Goal:  Complete the track RELIABLY, not intelligently.

Algorithm:
  1. Capture frame → grayscale → binary threshold → crop bottom half
  2. Divide into LEFT / CENTER / RIGHT regions
  3. Count white pixels in each → decide STRAIGHT / LEFT / RIGHT
  4. If total white pixels < threshold → lane lost → TURN using last_direction
  5. Send normalized (x, y) command to ESP32 via WebSocket

Usage:
  python3 brute_force_lane.py                 # Normal
  python3 brute_force_lane.py --debug         # MJPEG debug stream at :5000
  python3 brute_force_lane.py --no-wifi       # Vision-only (no ESP32)
  python3 brute_force_lane.py --video x.mp4   # Test with video file
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

# ── Ensure rpi/ is on the import path ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from camera.capture import CameraCapture
from comms.ws_bridge import WsBridge
from adas.decision_manager import Command

# ═══════════════════════════════════════════════════════════════════════════════
# TUNABLE PARAMETERS — Adjust these for your specific track & lighting
# ═══════════════════════════════════════════════════════════════════════════════

# Binary threshold: pixels brighter than this → white (lane), rest → black
# For white tape on black chart paper, 150–200 works well.
# Lower = more sensitive (picks up noise), Higher = stricter (may miss faint tape)
BINARY_THRESHOLD = 160

# How much of the frame to use (bottom portion only)
# 0.5 = bottom half, 0.6 = bottom 60%, etc.
# Higher = see further ahead (but more noise from surroundings)
CROP_RATIO = 0.5

# Minimum total white pixels to consider "lane is visible"
# Below this → lane is LOST → trigger turn logic
# Depends on resolution and tape width. Start with 500, tune up/down.
LANE_LOST_THRESHOLD = 500

# Steering values sent to ESP32 (normalized -1.0 to 1.0)
STEER_STRAIGHT = 0.0
STEER_LEFT     = -0.4    # gentle left correction
STEER_RIGHT    =  0.4    # gentle right correction
STEER_HARD_LEFT  = -0.9  # full turn at corner
STEER_HARD_RIGHT =  0.9  # full turn at corner

# Constant speed (normalized 0.0 to 1.0)
# Lower = safer but may stall on carpet. Higher = faster but riskier on turns.
CRUISE_SPEED = 0.25

# Turn speed (slower during hard turns for stability)
TURN_SPEED = 0.18

# How long to hold a hard turn when lane is lost (seconds)
# Too short = doesn't complete the 90° turn. Too long = overshoots.
TURN_HOLD_TIME = 0.8

# Minimum "dominance" ratio for center to count as STRAIGHT
# If center has > this fraction of total white pixels → go straight
CENTER_DOMINANCE = 0.35

# Region split ratios (divide frame width into 3 regions)
# Default: equal thirds. Adjust if camera is off-center.
LEFT_END   = 0.33    # left region: 0% to 33% of width
RIGHT_START = 0.66   # right region: 66% to 100% of width

# Startup frames to skip (let camera auto-expose)
STARTUP_SKIP_FRAMES = 10

# ═══════════════════════════════════════════════════════════════════════════════
# MJPEG Debug Stream (same as main.py — view at http://RPI_IP:5000/)
# ═══════════════════════════════════════════════════════════════════════════════

_mjpeg_frame = None
_mjpeg_lock = threading.Lock()
_MJPEG_PORT = 5000


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


# ═══════════════════════════════════════════════════════════════════════════════
# CORE LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

class BruteForceLaneFollower:
    """
    Dead-simple lane follower using binary thresholding + region pixel counts.
    No ML, no PID, no polynomial fitting — just count white pixels and steer.
    """

    def __init__(self, args):
        self.args = args
        self.running = False

        # State
        self.last_direction = "LEFT"      # default turn when lane lost
        self.turn_active = False           # currently executing a hard turn?
        self.turn_start_time = 0.0         # when the turn started
        self.frame_num = 0

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

        # ESP32 connection
        self.ws = None
        if not args.no_wifi:
            self.ws = WsBridge(
                host=config.ESP32_HOST,
                port=config.ESP32_WS_PORT,
                path=config.ESP32_WS_PATH,
            )

        # MJPEG debug server
        self._mjpeg = None
        if args.debug:
            self._mjpeg = _start_mjpeg_server()
            print(f"[DEBUG] Stream at http://<RPI_IP>:{_MJPEG_PORT}/")

    def start(self):
        """Initialize camera and ESP32 connection."""
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

        if self.ws:
            if not self.ws.connect():
                print("[WARN] ESP32 not connected — vision-only mode")
                self.ws = None

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

        # Skip initial frames (camera auto-exposure settling)
        if self.frame_num <= STARTUP_SKIP_FRAMES:
            if self.frame_num == STARTUP_SKIP_FRAMES:
                print(f"[STARTUP] Skipped {STARTUP_SKIP_FRAMES} frames for camera warmup")
            return

        t0 = time.monotonic()

        # ── Step 1: Preprocess ────────────────────────────────────────────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Binary threshold: white tape → 255, everything else → 0
        _, binary = cv2.threshold(gray, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)

        # Crop bottom portion only (ignore sky / top of room)
        h, w = binary.shape
        crop_start = int(h * (1.0 - CROP_RATIO))
        roi = binary[crop_start:h, :]

        # ── Step 2: Region-based pixel counting ──────────────────────────
        roi_h, roi_w = roi.shape
        left_boundary  = int(roi_w * LEFT_END)
        right_boundary = int(roi_w * RIGHT_START)

        left_region   = roi[:, 0:left_boundary]
        center_region = roi[:, left_boundary:right_boundary]
        right_region  = roi[:, right_boundary:roi_w]

        left_count   = cv2.countNonZero(left_region)
        center_count = cv2.countNonZero(center_region)
        right_count  = cv2.countNonZero(right_region)
        total_count  = left_count + center_count + right_count

        # ── Step 3: Decision logic ───────────────────────────────────────
        command = "STRAIGHT"
        steer = STEER_STRAIGHT
        speed = CRUISE_SPEED
        now = time.monotonic()

        # Check if we're in the middle of a hard turn
        if self.turn_active:
            elapsed = now - self.turn_start_time
            if elapsed < TURN_HOLD_TIME:
                # Still turning — hold the turn
                command = f"TURN_{self.last_direction}"
                steer = STEER_HARD_LEFT if self.last_direction == "LEFT" else STEER_HARD_RIGHT
                speed = TURN_SPEED
            else:
                # Turn complete — re-evaluate
                self.turn_active = False

        if not self.turn_active:
            if total_count < LANE_LOST_THRESHOLD:
                # ── LANE LOST → Hard turn using last known direction ────
                command = f"TURN_{self.last_direction}"
                steer = STEER_HARD_LEFT if self.last_direction == "LEFT" else STEER_HARD_RIGHT
                speed = TURN_SPEED
                self.turn_active = True
                self.turn_start_time = now

            elif total_count > 0 and (center_count / total_count) > CENTER_DOMINANCE:
                # Center has the most white → go straight
                command = "STRAIGHT"
                steer = STEER_STRAIGHT
                speed = CRUISE_SPEED

            elif left_count > right_count:
                # Lane is more to the left → steer left
                command = "LEFT"
                steer = STEER_LEFT
                speed = CRUISE_SPEED
                self.last_direction = "LEFT"

            elif right_count > left_count:
                # Lane is more to the right → steer right
                command = "RIGHT"
                steer = STEER_RIGHT
                speed = CRUISE_SPEED
                self.last_direction = "RIGHT"

            else:
                # Equal or ambiguous → SEARCH (use last direction gently)
                command = "SEARCH"
                steer = STEER_LEFT * 0.5 if self.last_direction == "LEFT" else STEER_RIGHT * 0.5
                speed = CRUISE_SPEED

        # ── Step 4: Send command to ESP32 ────────────────────────────────
        cmd = Command()
        cmd.steer_x = steer
        cmd.speed_y = speed

        if self.ws:
            self.ws.send_command(cmd)

        # ── Step 5: Debug output ─────────────────────────────────────────
        elapsed_ms = (time.monotonic() - t0) * 1000

        print(f"[F{self.frame_num:05d}] Command: {command:12s} | "
              f"L:{left_count:5d}  C:{center_count:5d}  R:{right_count:5d}  "
              f"Total:{total_count:6d} | "
              f"Steer:{steer:+.2f}  Speed:{speed:.2f} | "
              f"{elapsed_ms:.1f}ms")

        # ── Step 6: Debug visualization ──────────────────────────────────
        if self.args.debug:
            self._draw_debug(frame, roi, left_boundary, right_boundary,
                             crop_start, left_count, center_count, right_count,
                             total_count, command, steer, speed)

    def _draw_debug(self, frame, roi, left_b, right_b, crop_start,
                    left_c, center_c, right_c, total_c, command, steer, speed):
        """Draw debug visualization and push to MJPEG stream."""
        display = frame.copy()
        h, w = display.shape[:2]

        # Draw region boundaries on the frame
        cv2.line(display, (left_b, crop_start), (left_b, h), (0, 255, 255), 2)
        cv2.line(display, (right_b, crop_start), (right_b, h), (0, 255, 255), 2)
        cv2.line(display, (0, crop_start), (w, crop_start), (255, 0, 255), 2)

        # Color-code regions based on white pixel density
        # Overlay transparent colors on each region
        overlay = display.copy()
        alpha = 0.25

        # Left region - blue tint
        cv2.rectangle(overlay, (0, crop_start), (left_b, h), (255, 100, 0), -1)
        # Center region - green tint
        cv2.rectangle(overlay, (left_b, crop_start), (right_b, h), (0, 255, 0), -1)
        # Right region - red tint
        cv2.rectangle(overlay, (right_b, crop_start), (w, h), (0, 0, 255), -1)

        display = cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0)

        # Draw text info panel
        panel_lines = [
            f"CMD: {command}",
            f"Steer: {steer:+.2f}  Speed: {speed:.2f}",
            f"L:{left_c:5d} C:{center_c:5d} R:{right_c:5d}",
            f"Total: {total_c}  Threshold: {LANE_LOST_THRESHOLD}",
            f"Last Dir: {self.last_direction}",
            f"Turn Active: {self.turn_active}",
            f"Frame: {self.frame_num}",
        ]

        # Background panel
        cv2.rectangle(display, (5, 5), (320, 15 + len(panel_lines) * 22), (0, 0, 0), -1)

        for i, line in enumerate(panel_lines):
            color = (0, 255, 0)
            if "TURN" in line:
                color = (0, 100, 255)
            elif "LOST" in command:
                color = (0, 0, 255)
            cv2.putText(display, line, (10, 25 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                        cv2.LINE_AA)

        # Show the binary ROI as a small inset (bottom-left corner)
        roi_color = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        roi_small = cv2.resize(roi_color, (w // 3, roi.shape[0] // 2))
        rh, rw = roi_small.shape[:2]
        display[h - rh:h, 0:rw] = roi_small

        # Push to MJPEG stream
        _push_mjpeg(display)

    def stop(self):
        """Graceful shutdown."""
        print("\n[INFO] Stopping brute-force lane follower...")
        self.running = False

        if self.ws:
            self.ws.send_stop()
            self.ws.disconnect()

        self.camera.stop()

        print("=" * 55)
        print("  TARA — Brute-Force Lane Follower STOPPED")
        print(f"  Total frames processed: {self.frame_num}")
        print("=" * 55)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="TARA — Brute-Force Lane Follower (No ML)")
    parser.add_argument("--debug", action="store_true",
                        help="Stream debug view at http://RPI_IP:5000/")
    parser.add_argument("--no-wifi", action="store_true",
                        help="Run without ESP32 (vision-only)")
    parser.add_argument("--video", type=str, default=None,
                        help="Use video file instead of live camera")
    return parser.parse_args()


def main():
    args = parse_args()
    follower = BruteForceLaneFollower(args)

    def signal_handler(sig, _frame):
        follower.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    follower.run()
    sys.exit(0)


if __name__ == "__main__":
    main()
