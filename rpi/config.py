"""
TARA ADAS — Central Configuration
All tunable parameters for the ADAS pipeline.
Adjust these for your specific track/environment.
"""
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ─── Camera ───────────────────────────────────────────────────────────────────
# USB webcam (confirmed) — index 0 is the first /dev/video* device.
# If RPi shows multiple /dev/video* entries, try index 1 or 2.
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_FPS = 30

# Processing resolution (downscaled for speed)
PROC_WIDTH = 320
PROC_HEIGHT = 240

# ─── Serial Communication (RPi ↔ ESP32) ──────────────────────────────────────
# ESP32 plugged into RPi USB — appears as /dev/ttyUSB0 (CP2102/CH340) or
# /dev/ttyACM0 (native USB-CDC). Check with: ls /dev/tty*
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200
SERIAL_TIMEOUT = 0.01         # Non-blocking reads (10ms)
COMMAND_INTERVAL = 0.05       # Send commands every 50ms (20Hz)
WATCHDOG_TIMEOUT_MS = 500     # ESP32 stops motors if no command for 500ms

# ─── Lane Detection (LDW + LKA) ──────────────────────────────────────────────
# Tuned for INDOOR track with controlled/consistent lighting.

# Canny edge detection — tighter thresholds for clean indoor contrast
CANNY_LOW = 40
CANNY_HIGH = 120

# Gaussian blur kernel
BLUR_KERNEL = (5, 5)

# HSV color range for white lane markings
# Lowered value threshold 170→150 to catch highway lane markings at varying brightness
LANE_WHITE_HSV_LOW  = (0,   0, 150)
LANE_WHITE_HSV_HIGH = (180, 55, 255)

# HSV color range for yellow lane markings
LANE_YELLOW_HSV_LOW  = (15, 80, 120)
LANE_YELLOW_HSV_HIGH = (35, 255, 255)

# Hough transform parameters (tuned for highway dashed lane lines)
HOUGH_RHO = 1
HOUGH_THETA_DIVISOR = 180
HOUGH_THRESHOLD    = 20    # was 30 — lower to pick up dashed lines
HOUGH_MIN_LINE_LEN = 25    # was 40 — dashes are shorter
HOUGH_MAX_LINE_GAP = 200   # was 100 — large gaps between dashes

# Lane departure
LANE_DEPARTURE_THRESHOLD = 30  # pixels from center (was 40)

# LKA steering is pure proportional on the RPi (normalized offset -1..1).
# Motor-level PID (pidL / pidR) lives in the ESP32 Tasks.cpp / Navigation.cpp.

# Bird's-eye view perspective points (ratio of frame dimensions)
# Widened trapezoid for highway/dashcam video.
# The source quad must tightly bracket the visible lane markings.
# If lanes appear crossed (X pattern), widen the bottom or narrow the top.
BEV_SRC_RATIOS = [
    (0.05, 1.00),   # bottom-left  — full width at ground level
    (0.40, 0.62),   # top-left     — just below horizon on the left
    (0.60, 0.62),   # top-right    — just below horizon on the right
    (0.95, 1.00),   # bottom-right
]
BEV_DST_RATIOS = [
    (0.15, 1.0),    # bottom-left
    (0.15, 0.0),    # top-left
    (0.85, 0.0),    # top-right
    (0.85, 1.0),    # bottom-right
]

# ─── Traffic Sign Recognition ─────────────────────────────────────────────────
TSR_MODEL_PATH = os.path.join(MODELS_DIR, "tsr_mobilenetv2_int8.tflite")
TSR_CONFIDENCE_THRESHOLD = 0.6    # Per-frame threshold (stability comes from voting)
TSR_INPUT_SIZE = 96
TSR_NUM_CLASSES = 43
TSR_FRAME_SKIP = 1                # 1 = process every scheduled call, 2 = skip every other

# GTSRB class names (subset of important ones for prototype)
TSR_SIGN_NAMES = {
    0: "Speed limit 20",
    1: "Speed limit 30",
    2: "Speed limit 50",
    3: "Speed limit 60",
    4: "Speed limit 70",
    5: "Speed limit 80",
    6: "End speed 80",
    7: "Speed limit 100",
    8: "Speed limit 120",
    9: "No passing",
    10: "No passing >3.5t",
    11: "Right-of-way next",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "No >3.5t vehicles",
    17: "No entry",
    18: "General caution",
    19: "Dangerous left curve",
    20: "Dangerous right curve",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware ice/snow",
    31: "Wild animals",
    32: "End restrictions",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End no passing",
    42: "End no passing >3.5t",
}

# Speed limit mappings (sign class → max speed PWM value)
TSR_SPEED_LIMITS = {
    0: 50,    # 20 km/h
    1: 80,    # 30 km/h
    2: 120,   # 50 km/h
    3: 150,   # 60 km/h
    4: 180,   # 70 km/h
    5: 200,   # 80 km/h
    7: 230,   # 100 km/h
    8: 255,   # 120 km/h
    14: 0,    # Stop sign → stop
}

# ─── Pothole Detection ────────────────────────────────────────────────────────
POTHOLE_MODEL_PATH = os.path.join(MODELS_DIR, "pothole_mobilenetv2_int8.tflite")
POTHOLE_CONFIDENCE_THRESHOLD = 0.6    # output[1] = pothole prob; tune after real-road tests
POTHOLE_INPUT_SIZE = 128      # Using classifier approach (faster)
POTHOLE_USE_SSD = False       # Set True for SSD detector (slower but precise)
POTHOLE_SSD_MODEL_PATH = os.path.join(MODELS_DIR, "pothole_ssd_mobilenetv2_int8.tflite")
POTHOLE_SSD_INPUT_SIZE = 300

# Avoidance steering magnitude
POTHOLE_STEER_MAGNITUDE = 60  # PWM offset for avoidance

# ─── Adaptive Cruise Control ─────────────────────────────────────────────────
ACC_EMERGENCY_STOP_DIST = 10   # cm — slam brakes
ACC_MIN_FOLLOW_DIST = 25       # cm — slow down significantly
ACC_CRUISE_DIST = 50           # cm — maintain speed
ACC_DEFAULT_SPEED = 180        # PWM — default cruise speed (ESP32 floor is 150)
ACC_MAX_SPEED = 220            # PWM — absolute max speed (limit to 230 for stability)

# ACC PID removed — speed control is handled by the ESP32 motor driver.
# ACC now outputs a normalized speed setpoint (0.0–1.0) → ESP32 jd.y.
ACC_PID_KP = 2.0   # kept for reference only — not used by Python
ACC_PID_KI = 0.05
ACC_PID_KD = 0.5

# Encoder constants
ENCODER_TICKS_PER_REV = 20    # Ticks per wheel revolution
WHEEL_DIAMETER_CM = 6.5       # Wheel diameter in cm
WHEEL_CIRCUMFERENCE_CM = WHEEL_DIAMETER_CM * 3.14159

# ─── Decision Manager ────────────────────────────────────────────────────────
# Priority levels (lower = higher priority)
PRIORITY_EMERGENCY_STOP = 1
PRIORITY_POTHOLE_AVOID = 2
PRIORITY_LKA_STEER = 3
PRIORITY_ACC_SPEED = 4
PRIORITY_TSR_SPEED_LIMIT = 5
PRIORITY_LDW_WARNING = 6

# Steering range
STEER_MIN = -100
STEER_MAX = 100
SPEED_MIN = 0
SPEED_MAX = 255

# ─── Pipeline Scheduler ──────────────────────────────────────────────────────
# Which frame index modulo to run each module
SCHEDULE_LANE_EVERY = 1       # Every frame
SCHEDULE_ACC_EVERY = 2        # Every 2nd frame
SCHEDULE_TSR_OFFSET = 1       # Runs on frame 1, 5, 9, ...
SCHEDULE_TSR_EVERY = 4        # Every 4th frame
SCHEDULE_POTHOLE_OFFSET = 3   # Runs on frame 3, 7, 11, ...
SCHEDULE_POTHOLE_EVERY = 4    # Every 4th frame

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = "DEBUG"           # Changed to DEBUG to see raw pothole/TSR scores — set back to INFO in production
LOG_FILE = os.path.join(BASE_DIR, "tara_adas.log")
LOG_FPS = True                # Print FPS to console

# ─── Cloud (Firebase) ────────────────────────────────────────────────────────
# All cloud uploads are async and non-blocking.
# If Firebase is not configured, the prototype runs fully offline.
CLOUD_ENABLED = True          # Set False to skip cloud entirely

# Firebase project credentials — download from Firebase Console:
# Project Settings → Service Accounts → Generate new private key
FIREBASE_CREDENTIALS_PATH = os.path.join(BASE_DIR, "firebase_credentials.json")

# Firebase Realtime Database URL (from Firebase Console → Realtime Database)
FIREBASE_DB_URL = ""          # e.g. "https://tara-adas-default-rtdb.firebaseio.com"

# Firebase Storage bucket (from Firebase Console → Storage)
FIREBASE_STORAGE_BUCKET = ""  # e.g. "tara-adas.appspot.com"

# How often to push telemetry (seconds) — lower = more data, more bandwidth
CLOUD_TELEMETRY_INTERVAL = 2.0

# ─── Local Recording ─────────────────────────────────────────────────────────
# Always-on local backup — writes sensor CSV + event snapshots to SD card.
LOCAL_RECORDING_ENABLED = True
LOCAL_RECORDING_DIR = os.path.join(BASE_DIR, "recordings")

# ─── Traffic Light Recognition ───────────────────────────────────────────────
TL_ENABLED = True
TL_MIN_PIXELS = 500  # Number of glowing pixels to confirm detection

# HSV Ranges for typical traffic lights (tuned for indoor/bright LED lights)
# Red: (0, 100, 100) to (10, 255, 255)
TL_RED_LOW = (0, 100, 100)
TL_RED_HIGH = (10, 255, 255)

# Yellow: (15, 100, 100) to (35, 255, 255)
TL_YELLOW_LOW = (15, 100, 100)
TL_YELLOW_HIGH = (35, 255, 255)

# Green: (40, 50, 50) to (90, 255, 255)
TL_GREEN_LOW = (40, 50, 50)
TL_GREEN_HIGH = (90, 255, 255)
