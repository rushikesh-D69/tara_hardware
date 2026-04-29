// ─────────────────────────────────────────────────────────────────────────────
//  TARA Firmware  |  ESP32-Web.ino
//  Boot order: IMU calibration → Wi-Fi → FreeRTOS tasks → loop
//
//  Drive Modes
//  ───────────
//  MODE_MANUAL : Dashboard joystick controls the robot.
//                RPi CMD: packets are accepted ONLY for e-stop flag (bit 0x04).
//  MODE_AUTO   : RPi UART sends CMD:x,y,flags at up to 50 Hz.
//                Dashboard can read telemetry; movement input is ignored.
//                If no CMD: arrives for >1 s the mode auto-falls back to MANUAL.
//
//  Mode Toggle
//  ───────────
//  Dashboard sends:  {"type":"set_mode","mode":"auto"}  or  "manual"
//  RPi can also set: MODE flag in CMD bits (bit 0x01 = request AUTO,
//                    bit 0x02 = request MANUAL; processed in serialParserTask)
//
//  Telemetry
//  ─────────
//  SEN:… is broadcast over WebSocket only.
//  Serial (RPi) receives only ACK messages, so it never confuses its own
//  telemetry for an incoming command.
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

// ─── Serial Parser Task ───────────────────────────────────────────────────────
// Runs on Core 1 at priority 3. Parses RPi UART stream.
// Format: CMD:x,y,flags\n
//   x     = steering  −1.0 … +1.0
//   y     = throttle  −1.0 … +1.0
//   flags = bitmask
//             bit 0x01 → request MODE_AUTO
//             bit 0x02 → request MODE_MANUAL
//             bit 0x04 → emergency stop
void serialParserTask(void *pvParameters) {
  while (true) {
    if (Serial.available()) {
      String line = Serial.readStringUntil('\n');
      line.trim();

      if (line.startsWith("CMD:")) {
        float x = 0.0f, y = 0.0f;
        int   flags = 0;

        if (sscanf(line.c_str(), "CMD:%f,%f,%d", &x, &y, &flags) == 3) {
          lastSerialCmd = millis();  // RPi watchdog feed

          // ── Mode requests from RPi ──
          if (flags & 0x01) {
            driveMode = MODE_AUTO;
            // Notify dashboard of mode change
            if (ws.count() > 0) ws.textAll("{\"type\":\"mode\",\"mode\":\"auto\"}");
            Serial.println("[MODE] AUTO (RPi request)");
          } else if (flags & 0x02) {
            driveMode = MODE_MANUAL;
            if (ws.count() > 0) ws.textAll("{\"type\":\"mode\",\"mode\":\"manual\"}");
            Serial.println("[MODE] MANUAL (RPi request)");
          }

          // ── Emergency stop — always honoured regardless of mode ──
          eStopActive = (flags & 0x04) != 0;

          // ── In AUTO mode, push movement command to queue ──
          if (driveMode == MODE_AUTO) {
            lastWsMessage = millis(); // keep generic watchdog alive
            JoyData jd = {x, y};
            xQueueOverwrite(controlQueue, &jd);
          }
          // In MANUAL mode: RPi CMD is intentionally ignored for movement.
        }
      }
      else if (line.startsWith("nav_stop")) {
        navMode  = NAV_IDLE;
        seqLen   = seqIdx = 0;
        navStatus = 0; navProgress = 0.0f;
        pidL.reset(); pidR.reset();
        Serial.println("[NAV] RPi stop");
      }
      else if (line.startsWith("reset_odom")) {
        resetOdometry();
        yaw = 0.0f;
        Serial.println("[ODOM] Reset");
      }
    }
    vTaskDelay(5 / portTICK_PERIOD_MS); // 200 Hz parse rate
  }
}

// ─── Auto-mode watchdog ───────────────────────────────────────────────────────
// If in AUTO mode and no CMD: has been received for >1.5 s, fall back to MANUAL
// so the robot doesn't keep driving blind.
static void checkAutoWatchdog() {
  if (driveMode == MODE_AUTO && (millis() - lastSerialCmd > 1500)) {
    driveMode = MODE_MANUAL;
    JoyData stop = {0.0f, 0.0f};
    xQueueOverwrite(controlQueue, &stop);
    if (ws.count() > 0) ws.textAll("{\"type\":\"mode\",\"mode\":\"manual\",\"reason\":\"rpi_timeout\"}");
    Serial.println("[MODE] MANUAL (RPi timeout watchdog)");
  }
}

// ─── setup() ─────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n\n=== TARA SYSTEM BOOT ===");

  // ── 1. Pin setup ──
  pinMode(AIN1, OUTPUT); pinMode(AIN2, OUTPUT);
  pinMode(BIN1, OUTPUT); pinMode(BIN2, OUTPUT);
  pinMode(STBY, OUTPUT);
  digitalWrite(STBY, HIGH);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(ENC_L_PIN, INPUT_PULLUP);
  pinMode(ENC_R_PIN, INPUT_PULLUP);

  ledcAttach(PWMA, PWM_FREQ, PWM_RES);
  ledcAttach(PWMB, PWM_FREQ, PWM_RES);

  attachInterrupt(digitalPinToInterrupt(ENC_L_PIN), leftEncoderISR,  RISING);
  attachInterrupt(digitalPinToInterrupt(ENC_R_PIN), rightEncoderISR, RISING);

  // ── 2. IMU calibration (BLOCKING — robot must be still) ──
  Wire.begin(21, 22);
  Wire.setClock(400000); // fast mode
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

    // calibrateGyro prints status itself; takes ~6 s
    calibrateGyro();
    Serial.println("[IMU] Calibration complete");
  } else {
    Serial.println("[IMU] MPU6050 not found — yaw from encoders only");
  }

  // ── 3. Create FreeRTOS control queue ──
  controlQueue = xQueueCreate(1, sizeof(JoyData));

  // ── 4. Wi-Fi ──
  Serial.printf("[WIFI] Connecting to %s", ssid);
  WiFi.begin(ssid, password);
  for (int i = 0; i < 20 && WiFi.status() != WL_CONNECTED; i++) {
    delay(500); Serial.print('.');
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n[WIFI] Connected — IP: %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("\n[WIFI] Timeout — dashboard unavailable, continuing");
  }

  // ── 5. WebSocket server ──
  ws.onEvent(onWsEvent);
  server.addHandler(&ws);
  server.begin();
  Serial.println("[WS] Server active");

  // Seed watchdogs
  lastWsMessage = millis();
  lastSerialCmd = millis();

  // ── 6. FreeRTOS tasks ──
  //   Core 0: networking helpers
  //   Core 1: real-time control
  xTaskCreatePinnedToCore(serialParserTask, "SerialParser", 4096, NULL, 3, NULL, 1);
  xTaskCreatePinnedToCore(ultrasonicTask,   "Sonic",        4096, NULL, 1, NULL, 1);
  xTaskCreatePinnedToCore(motorTask,        "Motors",       8192, NULL, 2, NULL, 1);
  xTaskCreatePinnedToCore(imuTask,          "IMU",          4096, NULL, 2, NULL, 1);

  Serial.println("=== SYSTEM READY — Mode: MANUAL ===");
}

// ─── loop() ──────────────────────────────────────────────────────────────────
static unsigned long lastTelemetry = 0;

void loop() {
  ws.cleanupClients();

  // Auto-mode watchdog (if RPi goes silent, revert to manual)
  checkAutoWatchdog();

  if (millis() - lastTelemetry > 100) {  // 10 Hz telemetry
    lastTelemetry = millis();

    int   aeb_status = eStopActive ? 1 : (distanceCm < AUTO_STOP_DIST ? 2 : 0);
    float lead_dist  = (distanceCm < 999.0f) ? distanceCm : 0.0f;
    float ttc        = (v_linear > 0.01f && distanceCm < 999.0f)
                         ? (distanceCm / 100.0f) / v_linear : 0.0f;
    float heading_deg = ekf_theta * 180.0f / (float)M_PI;

    char packet[400];
    // SEN fields [1..20] + mode field [21]
    snprintf(packet, sizeof(packet),
      "SEN:%.3f,%.3f,%.2f,%.2f,%.3f,%d,%d,%ld,%ld,%.3f,%.3f,%.3f,%.2f,%.1f,%.2f,%d,%d,%.2f,%d,%.2f,%d",
      v_linear,           // [1]  forward velocity m/s
      baseSpeed,          // [2]  speed scale 0-1
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
      lead_dist,          // [14] ultrasonic distance cm
      ttc,                // [15] time-to-collision s
      0,                  // [16] acc_status (reserved)
      aeb_status,         // [17] 0=ok 1=estop 2=obstacle
      batVoltage,         // [18] battery voltage V
      (int)navStatus,     // [19] 0=idle 1=goto 2=turn
      navProgress,        // [20] nav progress 0-1
      (int)driveMode      // [21] 0=MANUAL 1=AUTO
    );

    // ── Push to WebSocket dashboard only ──
    // Do NOT echo back to Serial — RPi would parse its own telemetry as CMD
    if (WiFi.status() == WL_CONNECTED && ws.count() > 0 && ws.availableForWriteAll()) {
      ws.textAll(packet);
    }
  }
}