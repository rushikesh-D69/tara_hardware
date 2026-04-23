#include "Globals.h"

// Credentials
const char *ssid = "taralab";
const char *password = "tara@579";

// Network
AsyncWebServer server(80);
AsyncWebSocket ws("/ws");
unsigned long lastWsMessage = 0;

// Encoders
volatile long pulseLeft = 0;
volatile long pulseRight = 0;
long prevPulseL = 0;
long prevPulseR = 0;

// Kinematics
float PPR_L = 18.0f;
float PPR_R = 18.0f;
float M_PER_PULSE_L = WHEEL_CIRC / PPR_L;
float M_PER_PULSE_R = WHEEL_CIRC / PPR_R;
volatile bool pprUpdatePending = false;
volatile float new_PPR_L = 18.0f;
volatile float new_PPR_R = 18.0f;

// Actuation
int8_t motorDirL = 1;
int8_t motorDirR = 1;
int currentL_PWM = 0;
int currentR_PWM = 0;

float vL_actual = 0.0f;
float vR_actual = 0.0f;
float v_linear = 0.0f;
float v_angular = 0.0f;

// Dead-reckoning
float posX = 0.0f;
float posY = 0.0f;
float theta = 0.0f;
float distTraveled = 0.0f;

// EKF State
float ekf_x = 0.0f, ekf_y = 0.0f, ekf_theta = 0.0f;
float P[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
extern const float Q_pos = 0.001f;
extern const float Q_theta = 0.0005f;
extern const float R_theta = 0.5f;

// Sequence Tracking
float expectedX = 0.0f;
float expectedY = 0.0f;
float expectedTheta = 0.0f;

// Navigation
NavCmd navSequence[MAX_SEQ_STEPS] = {};
int seqLen = 0;
int seqIdx = 0;
NavMode navMode = NAV_IDLE;

// GOTO state
float navGoalDist = 0.0f;
float navTraveled = 0.0f;
int settleCount = 0;
long navStartPulseL = 0;
long navStartPulseR = 0;
float navSpeed = 0.15f;
float gotoPrevSpeed = 0.15f;
float smoothHeadingCorrection = 0.0f;

// TURN state
float navTurnInitErr = 0.0f;
float navTurnSpeed = 0.20f;
unsigned long navTurnStartMs = 0;
float turnStartYaw = 0.0f;
bool turnFirstTick = true;
float navTargetAngle = 0.0f;
extern const float TURN_TOL = 4.0f;

float headingKp = 3.0f;
uint8_t navStatus = 0;
float navProgress = 0.0f;

// IMU
volatile float yaw = 0.0f;
volatile float filteredRate = 0.0f;
float gyroZ_offset = 0.0f;
volatile bool resetYawPending = false;

// Sensors
float distanceCm = 999.0f;
bool eStopActive = false;
extern const float AUTO_STOP_DIST = 5.0f; // effectively disabled
float batVoltage = 12.0f;

// Control
float baseSpeed = 0.30f;
QueueHandle_t controlQueue;
