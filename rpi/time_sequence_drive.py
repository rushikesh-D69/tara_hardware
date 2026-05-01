#!/usr/bin/env python3
"""
TARA - Time-Based Sequence Drive
================================
Follows a hardcoded sequence of timed movements.
Bypasses the camera to ensure reliable movement on a fixed track.
"""

import os
import sys
import time
import signal

# Ensure rpi/ is on the import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from adas.decision_manager import Command
try:
    from comms.ws_bridge import WsBridge
except ImportError:
    print("[ERROR] comms.ws_bridge not found. Ensure you are in the rpi directory.")
    sys.exit(1)

# --- TUNABLE PARAMETERS ---
STRAIGHT_SPEED = 0.45
TURN_SPEED     = 0.35
TURN_STEER     = -0.9   # -0.9 for LEFT, +0.9 for RIGHT

# Adjust this value until the car does exactly a 90-degree turn
# Start with 1.2 and increase if it turns too little, decrease if it turns too much.
TURN_DURATION  = 1.2 

def run_sequence():
    print("=" * 50)
    print("  TARA - Starting Time-Based Sequence")
    print(f"  Target: ESP32 at {config.ESP32_HOST}")
    print("=" * 50)

    ws = WsBridge(host=config.ESP32_HOST, port=config.ESP32_WS_PORT)
    if not ws.connect():
        print("[ERROR] Could not connect to ESP32. Is it powered and on WiFi?")
        return

    # Define the sequence: (Label, Duration, Steer, Speed)
    sequence = [
        ("STRAIGHT (5s)",  5.0,  0.0,        STRAIGHT_SPEED),
        ("TURN LEFT",      TURN_DURATION, TURN_STEER, TURN_SPEED),
        ("STRAIGHT (4s)",  4.0,  0.0,        STRAIGHT_SPEED),
        ("TURN LEFT",      TURN_DURATION, TURN_STEER, TURN_SPEED),
        ("STRAIGHT (END)", 10.0, 0.0,        STRAIGHT_SPEED),
    ]

    try:
        for label, duration, steer, speed in sequence:
            print(f"[EXEC] {label} ...")
            start_time = time.time()
            
            while time.time() - start_time < duration:
                cmd = Command()
                cmd.steer_x = steer
                cmd.speed_y = speed
                ws.send_command(cmd)
                time.sleep(0.05)  # 20 Hz send rate

            # Small pause between actions to stabilize
            ws.send_command({"type": "auto_cmd", "x": 0, "y": 0})
            time.sleep(0.2)

        print("\n[DONE] Sequence completed successfully.")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        print("[INFO] Stopping motors and disconnecting...")
        ws.send_stop()
        ws.disconnect()

if __name__ == "__main__":
    run_sequence()
