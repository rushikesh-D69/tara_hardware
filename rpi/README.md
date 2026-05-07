# rpi/ — Raspberry Pi ADAS Software

This directory contains all Python code that runs on the **Raspberry Pi 4B**.

---

## Entry Points

| Script | Purpose |
|--------|---------|
| `main.py` | Full ADAS pipeline — ML inference, lane keeping, sign/pothole/light detection |
| `brute_force_lane.py` | No-ML fallback — pixel-counting lane follower, reliable for track completion |

## Directory Structure

```
rpi/
├── main.py                 # Full ML ADAS pipeline
├── brute_force_lane.py     # Brute-force lane follower (no ML)
├── config.py               # All tunable parameters
├── requirements.txt        # Python package dependencies
│
├── adas/                   # Perception & decision modules
│   ├── lane_detection.py   # LDW + LKA (adaptive threshold + polynomial fit)
│   ├── traffic_sign.py     # TSR (MobileNetV2 TFLite, GTSRB 43-class)
│   ├── pothole_detection.py# Pothole binary classifier / SSD
│   ├── adaptive_cruise.py  # ACC speed policy
│   ├── traffic_light.py    # TLR (HSV + circularity)
│   ├── sign_detector_cv.py # Directional sign (OpenCV, no ML)
│   └── decision_manager.py # Priority arbitrator → ESP32 command
│
├── camera/
│   └── capture.py          # Threaded camera (live + video file)
│
├── comms/
│   └── ws_bridge.py        # WebSocket client → ESP32
│
├── utils/
│   ├── fps_counter.py      # FPS + per-module timing
│   └── logger.py           # Structured logging setup
│
├── models/                 # TFLite models (not tracked in git)
│   ├── tsr_mobilenetv2_int8.tflite
│   └── pothole_mobilenetv2_int8.tflite
│
└── training/               # Model training notebooks / scripts
```

---

## Configuration

All parameters live in **`config.py`**. The most important ones to set before running:

```python
ESP32_HOST = "192.168.43.40"   # ESP32's IP address on your WiFi
CAMERA_INDEX = 0               # /dev/video0 — use 1 or 2 if needed
```

---

## CLI Flags

```
main.py:
  --debug        Start MJPEG stream at http://<RPI_IP>:5000/
  --no-wifi      Vision-only mode (no ESP32 connection)
  --video FILE   Use a video file instead of camera
  --log-level    DEBUG | INFO | WARNING | ERROR  (default: INFO)

brute_force_lane.py:
  --debug        MJPEG debug stream
  --video FILE   Test with video file
```

---

## Dependencies

Install on the Raspberry Pi:

```bash
pip install -r requirements.txt
```

Key packages: `opencv-python`, `numpy`, `websocket-client`, `ai-edge-litert`.
