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

// ─── Serial Parser ────────────────────────────────────────────────────────────
// Isolate in its own task so readStringUntil never blocks the main loop.
void serialParserTask(void *pvParameters) {
  while (true) {
    if (Serial.available()) {
      String line = Serial.readStringUntil('\n');
      line.trim();
      if (line.length() > 0 && line.charAt(0) == '{') {
        lastWsMessage = millis(); // keep watchdog alive

        StaticJsonDocument<512> doc;
        if (!deserializeJson(doc, line)) {
          const char *cmd = doc["type"] | "";

          if (strcmp(cmd, "drive") == 0) {
            JoyData jd = { (float)doc["x"], (float)doc["y"] };
            xQueueOverwrite(controlQueue, &jd);
            // flags bit 0x04 = emergency stop from RPi
            int flags = doc["flags"] | 0;
            eStopActive = (flags & 0x04) != 0;

          } else if (strcmp(cmd, "estop") == 0) {
            eStopActive = (bool)doc["state"];
            if (eStopActive) {
              JoyData jd = {0.0f, 0.0f};
              xQueueOverwrite(controlQueue, &jd);
            }

          } else if (strcmp(cmd, "nav_stop") == 0) {
            navMode = NAV_IDLE;
            seqLen = seqIdx = 0;
            navStatus = 0; navProgress = 0.0f;
            pidL.reset(); pidR.reset();

          } else if (strcmp(cmd, "reset_odom") == 0) {
            resetOdometry();
            yaw = 0.0f;
            Serial.println("[ODOM] Reset");
          }
        }
      }
    }
    vTaskDelay(5 / portTICK_PERIOD_MS); // 5 ms — 200 Hz max parse rate
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n--- TÁRA SYSTEM BOOT ---");
  Serial.printf("[CAL] WHEEL_CIRC     = %.5f m\n", WHEEL_CIRC);

  Serial.println("[CAL] Drive exactly 30cm, Serial-print navTraveled,");
  Serial.println("[CAL] then adjust PULSES_PER_REV until it reads 0.300");

  pinMode(AIN1, OUTPUT);
  pinMode(AIN2, OUTPUT);
  pinMode(BIN1, OUTPUT);
  pinMode(BIN2, OUTPUT);
  pinMode(STBY, OUTPUT);
  digitalWrite(STBY, HIGH);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(ENC_L_PIN, INPUT_PULLUP);
  pinMode(ENC_R_PIN, INPUT_PULLUP);

  ledcAttach(PWMA, PWM_FREQ, PWM_RES);
  ledcAttach(PWMB, PWM_FREQ, PWM_RES);

  attachInterrupt(digitalPinToInterrupt(ENC_L_PIN), leftEncoderISR, RISING);
  attachInterrupt(digitalPinToInterrupt(ENC_R_PIN), rightEncoderISR, RISING);

  Wire.begin(21, 22);
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0);
  if (Wire.endTransmission(true) == 0) {
    Serial.println("[OK] MPU6050 Detected");

    // Enable Digital Low Pass Filter (DLPF) to 42Hz
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x1A);
    Wire.write(0x03);
    Wire.endTransmission();

    // Set Gyro Range to +/- 250 deg/s
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x1B);
    Wire.write(0x00);
    Wire.endTransmission();

    calibrateGyro();
  } else {
    Serial.println("[ERROR] MPU6050 Not Found — yaw from encoders only");
  }

  controlQueue = xQueueCreate(1, sizeof(JoyData));

  Serial.print("[INFO] Connecting to Wi-Fi: ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[OK] Wi-Fi Connected");
    Serial.print("[OK] IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\n[FAIL] Wi-Fi Timeout — check credentials.");
  }

  ws.onEvent(onWsEvent);
  server.addHandler(&ws);
  server.begin();
  Serial.println("[OK] WebSocket Server Active");

  lastWsMessage = millis();

  // FreeRTOS tasks
  xTaskCreatePinnedToCore(serialParserTask, "SerialParser", 4096, NULL, 3, NULL, 1);
  xTaskCreatePinnedToCore(ultrasonicTask, "Sonic", 4096, NULL, 1, NULL, 1);
  xTaskCreatePinnedToCore(motorTask, "Motors", 8192, NULL, 2, NULL, 1);
  xTaskCreatePinnedToCore(imuTask, "IMU", 4096, NULL, 2, NULL, 1);

  Serial.println("--- SYSTEM READY ---");
}

unsigned long lastTelemetry = 0;

void loop() {
  ws.cleanupClients();

  if (millis() - lastTelemetry > 100) {
    lastTelemetry = millis();

    int aeb_status = eStopActive ? 1 : (distanceCm < AUTO_STOP_DIST ? 2 : 0);
    float lead_dist = (distanceCm < 999.0f) ? distanceCm : 0.0f;
    float ttc = (v_linear > 0.01f && distanceCm < 999.0f)
                    ? (distanceCm / 100.0f) / v_linear
                    : 0.0f;

    char packet[360];
    float heading_deg = ekf_theta * 180.0f / (float)M_PI;
    snprintf(packet, sizeof(packet),
             "$TARA,%.3f,%.3f,%.2f,%.2f,%.3f,%d,%d,%ld,%ld,%.3f,%.3f,%.3f,%.2f,%.1f,%.2f,%d,%d,%.2f,%d,%.2f",
             v_linear,       // [1]
             baseSpeed,      // [2]
             yaw,            // [3] IMU yaw angle °
             filteredRate,   // [4] IMU yaw rate °/s
             v_angular,      // [5]
             currentL_PWM,   // [6]
             currentR_PWM,   // [7]
             pulseLeft,      // [8]
             pulseRight,     // [9]
             distTraveled,   // [10]
             posX,           // [11]
             posY,           // [12]
             heading_deg,    // [13] EKF heading
             lead_dist,      // [14]
             ttc,            // [15]
             0,              // [16] acc_status
             aeb_status,     // [17]
             batVoltage,     // [18]
             (int)navStatus, // [19] 0=idle,1=goto,2=turn
             navProgress     // [20] 0.0-1.0
    );

    // Always send telemetry to RPi via Serial
    Serial.println(packet);

    // Also push to WebSocket dashboard if clients are connected
    if (WiFi.status() == WL_CONNECTED && ws.count() > 0 && ws.availableForWriteAll()) {
      ws.textAll(packet);
    }
  }
}