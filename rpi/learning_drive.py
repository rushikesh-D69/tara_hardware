#!/usr/bin/env python3
"""
TARA - Learning Mode (Record & Playback)
========================================
1. Manual Phase: Press keys to drive. The script records the duration of each move.
2. Auto Phase: Press 'q' to stop recording and start repeating the lap based on time.

Controls:
  'w' : Move Forward
  'a' : Turn Left
  'd' : Turn Right
  's' : Stop
  'q' : Finish Recording and Start Playback
"""

import os
import sys
import time
import tty
import termios
import threading

# Ensure rpi/ is on the import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from adas.decision_manager import Command
try:
    from comms.ws_bridge import WsBridge
except ImportError:
    print("[ERROR] comms.ws_bridge not found.")
    sys.exit(1)

# --- CONFIG ---
SPEED = 0.45
TURN_SPEED = 0.35

def get_key():
    """Captures a single keypress or arrow key escape sequence."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
        if ch == '\x1b':  # Escape sequence (e.g. arrows)
            ch += sys.stdin.read(2)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def run_learning_mode():
    ws = WsBridge(host=config.ESP32_HOST, port=config.ESP32_WS_PORT)
    if not ws.connect():
        return

    recorded_sequence = []
    current_move = None  # (steer, speed)
    start_time = None
    recording = True

    print("\n" + "="*50)
    print("  TARA LEARNING MODE - RECORDING")
    print("="*50)
    print("  Controls: [Arrow Keys] to drive, [q] to Playback")
    print("="*50)

    try:
        # --- PHASE 1: RECORDING ---
        while recording:
            key = get_key()
            now = time.time()

            # If we were already moving, save the duration of that move
            if current_move is not None and start_time is not None:
                duration = now - start_time
                recorded_sequence.append((current_move, duration))
                # print(f"  [SAVED] Move {current_move} for {duration:.2f}s")

            # Detect Keys / Arrow Sequences
            if key == '\x1b[A': # UP
                current_move = (0.0, SPEED)
                print("  [INPUT] Forward (Up Arrow)")
            elif key == '\x1b[D': # LEFT
                current_move = (-0.9, TURN_SPEED)
                print("  [INPUT] Left (Left Arrow)")
            elif key == '\x1b[C': # RIGHT
                current_move = (0.9, TURN_SPEED)
                print("  [INPUT] Right (Right Arrow)")
            elif key == '\x1b[B': # DOWN
                current_move = (0.0, 0.0)
                print("  [INPUT] Stop (Down Arrow)")
            elif key == 'q' or key == 'Q':
                print("\n[INFO] Recording finished. Starting Playback...")
                recording = False
                break
            else:
                continue

            # Send immediate command and start timer
            start_time = time.time()
            cmd = Command()
            cmd.steer_x, cmd.speed_y = current_move
            ws.send_command(cmd)

        # --- PHASE 2: PLAYBACK ---
        print("\n" + "="*50)
        print("  TARA LEARNING MODE - PLAYBACK")
        print("  The car is repeating your lap. Press Ctrl+C to stop.")
        print("="*50)

        while True:
            for (steer, speed), duration in recorded_sequence:
                print(f"  [PLAY] Steering {steer}, Speed {speed} for {duration:.2f}s")
                seg_start = time.time()
                while time.time() - seg_start < duration:
                    cmd = Command()
                    cmd.steer_x, cmd.speed_y = steer, speed
                    ws.send_command(cmd)
                    time.sleep(0.05)
                
                # Tiny pause between steps
                ws.send_command({"type": "auto_cmd", "x": 0, "y": 0})
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        ws.send_stop()
        ws.disconnect()

if __name__ == "__main__":
    run_learning_mode()
