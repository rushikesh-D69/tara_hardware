// ─────────────────────────────────────────────────────────────────────────────
//  TARA Firmware  |  ESP32-Web.ino
//  Boot order: IMU calibration → Wi-Fi → WebSocket → FreeRTOS tasks → loop
//
//  Communication — WiFi only (no UART to RPi)
//  ──────────────────────────────────────────
//  All three nodes on the same WiFi network.
//  ESP32 runs a single WebSocket server at ws://IP/ws
//  Both the Dashboard and the RPi connect as WS clients.
//
//  The RPi sends {"type":"register","role":"rpi"} immediately on connect
//  so the ESP32 can tell which client is which.
//
//  Drive Modes
//  ───────────
//  MODE_MANUAL : Dashboard joystick → controlQueue.
//                RPi movement commands ignored.
//                Watchdog: if Dashboard goes silent >2 s → motors stop.
//  MODE_AUTO   : RPi auto_cmd → controlQueue (up to 50 Hz).
//                Dashboard reads telemetry; movement ignored.
//                Watchdog: if RPi goes silent >1.5 s → fallback to MANUAL.
//
//  Telemetry
//  ─────────
//  SEN:… broadcast to ALL WebSocket clients at 10 Hz from loop().
// ─────────────────────────────────────────────────────────────────────────────

#include <Arduino.h>
#include <WiFi.h>
#include <Wire.h>
#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>
#include <ArduinoJson.h>

#include "Config.h"
#include "Globals.h"
#include "Encoders.h"
#include "IMU.h"
#include "Tasks.h"
#include "WsHandler.h"
#include "Navigation.h"

// Auto-mode watchdog REMOVED per user request. 
// Mode will persist until changed manually via Dashboard or RPi command.

// ─── setup() ─────────────────────────────────────────────────────────────────
void setup() {
  // Serial is for debug monitor only — NOT used to communicate with RPi
  Serial.begin(115200);
  delay(500);
  Serial.println("\n\n=== TARA SYSTEM BOOT ===");
  Serial.println("[INFO] Communication: WiFi WebSocket only (no UART to RPi)");

  // ── 1. Pin setup ──
  pinMode(AIN1, OUTPUT); pinMode(AIN2, OUTPUT);
  pinMode(BIN1, OUTPUT); pinMode(BIN2, OUTPUT);
  pinMode(STBY, OUTPUT);
  digitalWrite(STBY, HIGH);
  // Note: TRIG_PIN / ECHO_PIN (ultrasonic) removed — sensor not used
  pinMode(ENC_L_PIN, INPUT_PULLUP);
  pinMode(ENC_R_PIN, INPUT_PULLUP);

  ledcAttach(PWMA, PWM_FREQ, PWM_RES);
  ledcAttach(PWMB, PWM_FREQ, PWM_RES);

  attachInterrupt(digitalPinToInterrupt(ENC_L_PIN), leftEncoderISR,  RISING);
  attachInterrupt(digitalPinToInterrupt(ENC_R_PIN), rightEncoderISR, RISING);

  // ── 2. IMU calibration (BLOCKING — robot must be still) ──
  Wire.begin(21, 22);
  Wire.setClock(400000);
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B); Wire.write(0x00);  // wake up
  if (Wire.endTransmission(true) == 0) {
    Serial.println("[IMU] MPU6050 found");

    Wire.beginTransmission(MPU_ADDR);  // DLPF 42 Hz
    Wire.write(0x1A); Wire.write(0x03);
    Wire.endTransmission();

    Wire.beginTransmission(MPU_ADDR);  // Gyro ±250 °/s
    Wire.write(0x1B); Wire.write(0x00);
    Wire.endTransmission();

    calibrateGyro();
    Serial.println("[IMU] Calibration complete");
  } else {
    Serial.println("[IMU] MPU6050 not found — yaw from encoders only");
  }

  // ── 3. FreeRTOS control queue ──
  controlQueue = xQueueCreate(1, sizeof(JoyData));

  // ── 4. Wi-Fi ──
  Serial.printf("[WIFI] Connecting to %s", ssid);
  WiFi.begin(ssid, password);
  for (int i = 0; i < 20 && WiFi.status() != WL_CONNECTED; i++) {
    delay(500); Serial.print('.');
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n[WIFI] Connected — IP: %s\n",
                  WiFi.localIP().toString().c_str());
    Serial.println("[WIFI] Share this IP with RPi (config.py ESP32_HOST) "
                   "and Dashboard (script.js ESP32_IP)");
  } else {
    Serial.println("\n[WIFI] Timeout — no network found, continuing offline");
  }

  // ── 5. WebSocket server ──
  ws.onEvent(onWsEvent);
  server.addHandler(&ws);
  server.begin();
  Serial.println("[WS] Server active at ws://" +
                 WiFi.localIP().toString() + "/ws");
  Serial.println("[WS] Expecting connections from:");
  Serial.println("       Dashboard : browser → ws://IP/ws");
  Serial.println("       RPi ADAS  : ws_bridge.py → ws://IP/ws");
  Serial.println("       RPi registers with {type:register,role:rpi}");

  // Seed watchdogs (prevents false timeout at boot)
  lastWsMessage = millis();
  lastRpiCmd    = millis();

  // ── 6. FreeRTOS tasks ──
  //   Core 1 (real-time): motorTask, imuTask
  //   Core 0 (networking): nothing extra — WS handled by AsyncTCP ISR
  xTaskCreatePinnedToCore(motorTask, "Motors", 8192, NULL, 2, NULL, 1);
  xTaskCreatePinnedToCore(imuTask,   "IMU",    4096, NULL, 2, NULL, 1);
  // serialParserTask REMOVED — RPi communicates via WebSocket now
  // ultrasonicTask   REMOVED — no ultrasonic sensor

  Serial.println("=== SYSTEM READY — Mode: MANUAL ===");
  Serial.printf( "    ESP32 IP : %s\n", WiFi.localIP().toString().c_str());
}

// ─── loop() ──────────────────────────────────────────────────────────────────
static unsigned long lastTelemetry = 0;

void loop() {
  ws.cleanupClients();

  // Auto-mode watchdog REMOVED

  // ── Telemetry broadcast at 10 Hz to ALL WebSocket clients ────────────────
  if (millis() - lastTelemetry > 100) {
    lastTelemetry = millis();

    float heading_deg = ekf_theta * 180.0f / (float)M_PI;
    // Ultrasonic fields kept at 0 — sensor removed but SEN format preserved
    // so existing dashboard parser doesn't break field indexing.
    const float lead_dist = 0.0f;
    const float ttc       = 0.0f;
    const int   aeb_status = eStopActive ? 1 : 0;

    char packet[400];
    snprintf(packet, sizeof(packet),
      "SEN:%.3f,%.3f,%.2f,%.2f,%.3f,%d,%d,%ld,%ld,%.3f,%.3f,%.3f,%.2f,%.1f,%.2f,%d,%d,%.2f,%d,%.2f,%d",
      v_linear,           // [1]  forward velocity m/s
      baseSpeed,          // [2]  speed scale 0–1
      yaw,                // [3]  IMU yaw °
      filteredRate,       // [4]  IMU yaw rate °/s
      v_angular,          // [5]  angular velocity rad/s
      currentL_PWM,       // [6]  left motor PWM
      currentR_PWM,       // [7]  right motor PWM
      pulseLeft,          // [8]  left encoder pulses
      pulseRight,         // [9]  right encoder pulses
      distTraveled,       // [10] odometry distance m
      posX,               // [11] EKF X m
      posY,               // [12] EKF Y m
      heading_deg,        // [13] EKF heading °
      lead_dist,          // [14] ultrasonic cm (always 0 — sensor removed)
      ttc,                // [15] TTC s        (always 0 — sensor removed)
      0,                  // [16] acc_status   (reserved)
      aeb_status,         // [17] 0=ok 1=estop
      batVoltage,         // [18] battery V
      (int)navStatus,     // [19] 0=idle 1=goto 2=turn
      navProgress,        // [20] nav progress 0–1
      (int)driveMode      // [21] 0=MANUAL 1=AUTO
    );

    if (WiFi.status() == WL_CONNECTED && ws.count() > 0 &&
        ws.availableForWriteAll()) {
      ws.textAll(packet);
    }
  }
}