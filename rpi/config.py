"""
TARA ADAS — Central Configuration
All tunable parameters for the ADAS pipeline.
"""
import os

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_INDEX  = 0
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480
CAMERA_FPS    = 30

PROC_WIDTH  = 320
PROC_HEIGHT = 240

# ── ESP32 WiFi / WebSocket ────────────────────────────────────────────────────
ESP32_HOST       = "192.168.43.40"
ESP32_WS_PORT    = 80
ESP32_WS_PATH    = "/ws"
COMMAND_INTERVAL = 0.05

# ── Lane Detection (LDW + LKA) ────────────────────────────────────────────────
LANE_USE_ADAPTIVE_THRESH = True
LANE_ADAPTIVE_BLOCK_SIZE = 51
LANE_ADAPTIVE_C          = -15
LANE_MORPH_KERNEL        = 5
LANE_STEERING_DEADBAND   = 0

CANNY_LOW  = 40
CANNY_HIGH = 120
BLUR_KERNEL = (5, 5)

LANE_WHITE_HSV_LOW   = (0,   0, 130)
LANE_WHITE_HSV_HIGH  = (180, 50, 255)
LANE_YELLOW_HSV_LOW  = (15,  80, 120)
LANE_YELLOW_HSV_HIGH = (35, 255, 255)

HOUGH_RHO           = 1
HOUGH_THETA_DIVISOR = 180
HOUGH_THRESHOLD     = 20
HOUGH_MIN_LINE_LEN  = 15
HOUGH_MAX_LINE_GAP  = 50

LANE_DEPARTURE_THRESHOLD = 30

# Bird's-eye view perspective ratios (fraction of frame dimensions)
BEV_SRC_RATIOS = [
    (0.00, 1.00),
    (0.20, 0.65),
    (0.80, 0.65),
    (1.00, 1.00),
]
BEV_DST_RATIOS = [
    (0.10, 1.0),
    (0.10, 0.0),
    (0.90, 0.0),
    (0.90, 1.0),
]

# ── Traffic Sign Recognition (TSR) ────────────────────────────────────────────
TSR_MODEL_PATH           = os.path.join(MODELS_DIR, "tsr_mobilenetv2_int8.tflite")
TSR_CONFIDENCE_THRESHOLD = 0.6
TSR_INPUT_SIZE           = 96
TSR_NUM_CLASSES          = 43
TSR_FRAME_SKIP           = 1

TSR_SIGN_NAMES = {
    0:  "Speed limit 20",
    1:  "Speed limit 30",
    2:  "Speed limit 50",
    3:  "Speed limit 60",
    4:  "Speed limit 70",
    5:  "Speed limit 80",
    6:  "End speed 80",
    7:  "Speed limit 100",
    8:  "Speed limit 120",
    9:  "No passing",
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

# Normalized speed setpoints (0.0–1.0) for each speed-limit sign class
TSR_SPEED_LIMITS = {
    0:  0.20,
    1:  0.31,
    2:  0.47,
    3:  0.59,
    4:  0.71,
    5:  0.78,
    7:  0.90,
    8:  1.00,
    14: 0.0,
}

# ── Pothole Detection ─────────────────────────────────────────────────────────
POTHOLE_MODEL_PATH           = os.path.join(MODELS_DIR, "pothole_mobilenetv2_int8.tflite")
POTHOLE_CONFIDENCE_THRESHOLD = 0.6
POTHOLE_INPUT_SIZE           = 128
POTHOLE_USE_SSD              = False
POTHOLE_SSD_MODEL_PATH       = os.path.join(MODELS_DIR, "pothole_ssd_mobilenetv2_int8.tflite")
POTHOLE_SSD_INPUT_SIZE       = 300
POTHOLE_STEER_MAGNITUDE      = 60

# ── Adaptive Cruise Control ───────────────────────────────────────────────────
ACC_EMERGENCY_STOP_DIST = 10
ACC_MIN_FOLLOW_DIST     = 25
ACC_CRUISE_DIST         = 50
ACC_DEFAULT_SPEED       = 50
ACC_MAX_SPEED           = 160

ACC_PID_KP = 2.0
ACC_PID_KI = 0.05
ACC_PID_KD = 0.5

ENCODER_TICKS_PER_REV   = 20
WHEEL_DIAMETER_CM       = 6.5
WHEEL_CIRCUMFERENCE_CM  = WHEEL_DIAMETER_CM * 3.14159

# ── Decision Manager ──────────────────────────────────────────────────────────
PRIORITY_EMERGENCY_STOP  = 1
PRIORITY_TRAFFIC_LIGHT   = 2
PRIORITY_POTHOLE_AVOID   = 3
PRIORITY_TSR_SPEED_LIMIT = 4
PRIORITY_ACC_SPEED       = 5
PRIORITY_LKA_STEER       = 6
PRIORITY_LDW_WARNING     = 7

STEER_MIN = -100
STEER_MAX  = 100
SPEED_MIN  = 0
SPEED_MAX  = 255

# ── Pipeline Scheduler ────────────────────────────────────────────────────────
SCHEDULE_LANE_EVERY      = 1
SCHEDULE_ACC_EVERY       = 2
SCHEDULE_TSR_OFFSET      = 1
SCHEDULE_TSR_EVERY       = 4
SCHEDULE_POTHOLE_OFFSET  = 3
SCHEDULE_POTHOLE_EVERY   = 4

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FILE  = os.path.join(BASE_DIR, "tara_adas.log")
LOG_FPS   = True

# ── Cloud (Firebase) ──────────────────────────────────────────────────────────
CLOUD_ENABLED              = True
FIREBASE_CREDENTIALS_PATH  = os.path.join(BASE_DIR, "firebase_credentials.json")
FIREBASE_DB_URL            = ""
FIREBASE_STORAGE_BUCKET    = ""
CLOUD_TELEMETRY_INTERVAL   = 2.0

# ── Local Recording ───────────────────────────────────────────────────────────
LOCAL_RECORDING_ENABLED = True
LOCAL_RECORDING_DIR     = os.path.join(BASE_DIR, "recordings")

# ── Traffic Light Recognition ─────────────────────────────────────────────────
TL_ENABLED         = True
TL_MIN_PIXELS      = 800

TL_RED_LOW    = (0,   100, 100)
TL_RED_HIGH   = (10,  255, 255)
TL_RED_LOW_2  = (170, 100, 100)
TL_RED_HIGH_2 = (180, 255, 255)

TL_YELLOW_LOW  = (15,  100, 100)
TL_YELLOW_HIGH = (35,  255, 255)

TL_GREEN_LOW  = (40, 50,  50)
TL_GREEN_HIGH = (90, 255, 255)

TL_MIN_CIRCULARITY  = 0.5
TL_MIN_CONTOUR_AREA = 200

# ── OpenCV Sign Detection ─────────────────────────────────────────────────────
SIGN_BLUE_LOW             = (100, 100,  50)
SIGN_BLUE_HIGH            = (140, 255, 255)
SIGN_WHITE_LOW            = (0,   0,   180)
SIGN_WHITE_HIGH           = (180,  50, 255)
SIGN_MIN_AREA             = 150
SIGN_CIRCULARITY_THRESHOLD = 0.5
SIGN_TURN_DELAY_SEC       = 1.0
SIGN_TURN_HOLD_SEC        = 1.2
