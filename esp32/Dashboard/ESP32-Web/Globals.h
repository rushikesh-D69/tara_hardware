#ifndef GLOBALS_H
#define GLOBALS_H

#include <Arduino.h>
#include <WiFi.h>
#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>
#include "Config.h"

// WiFi Credentials
extern const char *ssid;
extern const char *password;

extern AsyncWebServer server;
extern AsyncWebSocket ws;
extern unsigned long lastWsMessage;

// Encoders
extern volatile long pulseLeft;
extern volatile long pulseRight;
extern long prevPulseL;
extern long prevPulseR;

// Kinematics Configuration
extern float PPR_L;
extern float PPR_R;
extern float M_PER_PULSE_L;
extern float M_PER_PULSE_R;
extern volatile bool pprUpdatePending;
extern volatile float new_PPR_L;
extern volatile float new_PPR_R;

// Actuation & States
extern int8_t motorDirL;
extern int8_t motorDirR;
extern int currentL_PWM;
extern int currentR_PWM;

extern float vL_actual;
extern float vR_actual;
extern float v_linear;
extern float v_angular;

extern float posX;
extern float posY;
extern float theta;
extern float distTraveled;

// EKF State
extern float ekf_x;
extern float ekf_y;
extern float ekf_theta;
extern float P[9];
extern const float Q_pos;
extern const float Q_theta;
extern const float R_theta;

extern float expectedX;
extern float expectedY;
extern float expectedTheta;

// Navigation Enums and Structs
enum NavStep { STEP_GOTO, STEP_TURN };
struct NavCmd {
  NavStep type;
  float param;
  float speed;
};
enum NavMode { NAV_IDLE, NAV_GOTO, NAV_TURN };

extern NavCmd navSequence[MAX_SEQ_STEPS];
extern int seqLen;
extern int seqIdx;
extern NavMode navMode;

// GOTO state
extern float navGoalDist;
extern float navTraveled;
extern int settleCount;
extern long navStartPulseL;
extern long navStartPulseR;
extern float navSpeed;
extern float gotoPrevSpeed;
extern float smoothHeadingCorrection;

// TURN state
extern float navTurnInitErr;
extern float navTurnSpeed;
extern unsigned long navTurnStartMs;
extern float turnStartYaw;
extern bool turnFirstTick;
extern float navTargetAngle;
extern const float TURN_TOL;

extern float headingKp;
extern uint8_t navStatus;
extern float navProgress;

// IMU
extern volatile float yaw;
extern volatile float filteredRate;
extern float gyroZ_offset;
extern volatile bool resetYawPending;

// Sensors
extern float distanceCm;
extern bool eStopActive;
extern const float AUTO_STOP_DIST;
extern float batVoltage;

// Control
extern float baseSpeed;
struct JoyData {
  float x;
  float y;
};
extern QueueHandle_t controlQueue;

#endif // GLOBALS_H
