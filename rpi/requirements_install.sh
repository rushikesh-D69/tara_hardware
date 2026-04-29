# ─────────────────────────────────────────────────────────────────────────────
#  TARA RPi — All pip install commands
#  Run these on the Raspberry Pi before starting main.py
#
#  Tested on: Raspberry Pi 4B, Raspberry Pi OS (Bullseye/Bookworm), Python 3.9+
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. Core vision & ML ──────────────────────────────────────────────────────

# OpenCV (headless = no GUI — saves ~100 MB on RPi)
pip install opencv-contrib-python-headless

# NumPy (required by OpenCV + TFLite)
pip install numpy

# ── 2. TensorFlow Lite runtime ───────────────────────────────────────────────
# Use the lightweight tflite-runtime — do NOT install full tensorflow on RPi
# (too large, too slow to build)

pip install tflite-runtime

# If tflite-runtime is not available for your Python version, use:
# pip install tflite-runtime --extra-index-url https://google-coral.github.io/py-repo/

# NOTE: training/export_tflite.py uses full tensorflow (tf.keras).
#       Only needed if re-training on a PC — NOT required to run on RPi.
# pip install tensorflow      ← PC only, skip on RPi

# ── 3. WebSocket client ──────────────────────────────────────────────────────
# Used by comms/ws_bridge.py to connect RPi → ESP32 over WiFi

pip install websocket-client

# ── 4. Serial (legacy / debug only) ─────────────────────────────────────────
# comms/serial_bridge.py still exists but is no longer used at runtime.
# Install anyway in case you want to debug over USB:

pip install pyserial

# ── 5. Firebase cloud logger ─────────────────────────────────────────────────
# cloud/firebase_logger.py — only needed if CLOUD_ENABLED = True in config.py

pip install firebase-admin

# ── 6. Matplotlib (training/visualisation only) ──────────────────────────────
# Only needed for training/export_tflite.py plots — skip on RPi if not training

pip install matplotlib

# ─────────────────────────────────────────────────────────────────────────────
#  ONE-LINER (copy-paste onto RPi terminal)
# ─────────────────────────────────────────────────────────────────────────────

pip install \
  opencv-contrib-python-headless \
  numpy \
  tflite-runtime \
  websocket-client \
  pyserial \
  firebase-admin  

# ─────────────────────────────────────────────────────────────────────────────
#  Standard library modules (no install needed — built into Python)
# ─────────────────────────────────────────────────────────────────────────────
#  argparse, collections, datetime, json, logging,
#  os, queue, signal, sys, threading, time
