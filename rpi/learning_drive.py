#!/usr/bin/env python3
"""
TARA - Advanced Learning Mode (Record & Playback)
=================================================
1. Teach Phase: Drive with Arrow Keys. The script records every move and duration.
2. Auto Phase: The car repeats the lap perfectly based on saved timings.

Features:
  - Persistent storage in lap_sequence.json
  - Replay existing lap or record a new one
  - Captures the final segment before 'q'
  - Smoothing between transitions
"""

import os
import sys
import time
import tty
import termios
import json

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
LAP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lap_sequence.json")

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

def print_summary(sequence):
    print("\n" + "="*50)
    print(f"  LAP SUMMARY ({len(sequence)} steps)")
    print("="*50)
    for i, step in enumerate(sequence, 1):
        m = step["move"]
        label = "Straight" if m[0] == 0 else ("Left" if m[0] < 0 else "Right")
        if m[1] == 0: label = "Stop"
        print(f"  {i}. {label:10s} | Time: {step['duration']:5.2f}s")
    print("="*50)

def run_learning_mode():
    ws = WsBridge(host=config.ESP32_HOST, port=config.ESP32_WS_PORT)
    if not ws.connect():
        return

    recorded_sequence = []
    
    # --- CHECK FOR EXISTING LAP ---
    if os.path.exists(LAP_FILE):
        print(f"\n[FOUND] An existing lap was found in {os.path.basename(LAP_FILE)}.")
        choice = input("Do you want to [R]eplay the saved lap or [T]each a new one? (R/T): ").lower()
        if choice == 'r':
            with open(LAP_FILE, 'r') as f:
                recorded_sequence = json.load(f)
            print_summary(recorded_sequence)
            print("Starting Playback in 3 seconds...")
            time.sleep(3)
            start_playback(ws, recorded_sequence)
            return

    # --- PHASE 1: RECORDING ---
    current_move = None 
    start_time = None
    recording = True

    print("\n" + "="*50)
    print("  TARA LEARNING MODE - RECORDING")
    print("="*50)
    print("  ↑ : Forward")
    print("  ← : Left        → : Right")
    print("  ↓ : Stop")
    print("  q : Finish and Save")
    print("="*50)

    try:
        while recording:
            key = get_key()
            now = time.time()

            # Define new intended move
            new_move = None
            if key == '\x1b[A': new_move = (0.0, SPEED)        # UP
            elif key == '\x1b[D': new_move = (-0.9, TURN_SPEED) # LEFT
            elif key == '\x1b[C': new_move = (0.9, TURN_SPEED)  # RIGHT
            elif key == '\x1b[B': new_move = (0.0, 0.0)         # DOWN
            elif key.lower() == 'q':
                # Final segment capture
                if current_move and start_time:
                    recorded_sequence.append({"move": current_move, "duration": now - start_time})
                recording = False
                break
            
            if new_move is None or new_move == current_move:
                continue

            # Save the completed segment
            if current_move is not None and start_time is not None:
                duration = now - start_time
                if duration > 0.1: # Filter out accidental double-taps
                    recorded_sequence.append({"move": current_move, "duration": duration})
                    print(f"  [SAVED] {duration:.2f}s segment")

            # Start the new move
            current_move = new_move
            start_time = time.time()
            cmd = Command()
            cmd.steer_x, cmd.speed_y = current_move
            ws.send_command(cmd)
            print(f"  [INPUT] Driving...")

        # Save to file
        if recorded_sequence:
            with open(LAP_FILE, "w") as f:
                json.dump(recorded_sequence, f, indent=2)
            print_summary(recorded_sequence)
            start_playback(ws, recorded_sequence)

    except KeyboardInterrupt:
        ws.send_stop()
    finally:
        ws.disconnect()

def start_playback(ws, sequence):
    print("\n[AUTO] Starting Playback Loop. Press Ctrl+C to stop.")
    try:
        while True:
            for step in sequence:
                steer, speed = step["move"]
                duration = step["duration"]
                
                seg_start = time.time()
                while time.time() - seg_start < duration:
                    cmd = Command()
                    cmd.steer_x, cmd.speed_y = steer, speed
                    ws.send_command(cmd)
                    time.sleep(0.05)
                
                # Smooth transition: brief neutral command
                ws.send_command({"type": "auto_cmd", "x": 0, "y": 0})
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[INFO] Playback stopped.")

if __name__ == "__main__":
    run_learning_mode()
