#ifndef CONFIG_H
#define CONFIG_H

#include <math.h>

/* -------- HARDWARE PINS -------- */
// Motors — TB6612FNG
#define AIN1 26 // Motor A → RIGHT wheel
#define AIN2 27
#define PWMA 14
#define BIN1 25 // Motor B → LEFT wheel
#define BIN2 33
#define PWMB 32
#define STBY 13

// Sensors
#define ENC_L_PIN 16
#define ENC_R_PIN 17
#define TRIG_PIN 18
#define ECHO_PIN 19
#define BAT_PIN 36 // VP — analog battery voltage divider

constexpr int PWM_FREQ = 5000;
constexpr int PWM_RES = 8; // 8-bit → 0-255

// IMU
#define MPU_ADDR 0x68

/* ===================================================
   ROBOT KINEMATICS
   =================================================== */
constexpr float WHEEL_RADIUS = 0.030f; // m
constexpr float WHEEL_BASE = 0.140f;   // m (4WD scrub wider effective base)
constexpr float MAX_SPEED_MS = 1.0f; // m/s at PWM=255 — calibrate after testing!

// Note: WHEEL_CIRC is computed here.
constexpr float WHEEL_CIRC = 2.0f * (float)M_PI * WHEEL_RADIUS; // ≈ 0.1885 m

/* ===================================================
   AUTONOMOUS NAVIGATION CONTROL
   =================================================== */
#define MAX_SEQ_STEPS 10

#endif // CONFIG_H
