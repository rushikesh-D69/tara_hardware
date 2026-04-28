#include "IMU.h"
#include <Arduino.h>
#include <Wire.h>
#include "Config.h"
#include "Globals.h"

void calibrateGyro() {
  Serial.println("Calibrating gyro... keep TÁRA still");
  long sum = 0;
  for (int i = 0; i < 3000; i++) {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x47);
    Wire.endTransmission(false);
    Wire.requestFrom((uint16_t)MPU_ADDR, (uint8_t)2, true);
    int16_t gz = Wire.read() << 8 | Wire.read();
    sum += gz;
    delay(2);
  }
  gyroZ_offset = sum / 3000.0f;
  Serial.print("Gyro offset: ");
  Serial.println(gyroZ_offset);
}

void imuTask(void *pvParameters) {
  unsigned long lastTime = millis();
  while (true) {
    if (resetYawPending) {
      yaw = 0.0f;
      filteredRate = 0.0f;
      resetYawPending = false;
    }

    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x47);
    Wire.endTransmission(false);
    Wire.requestFrom((uint16_t)MPU_ADDR, (uint8_t)2, true);

    if (Wire.available() < 2) {
      vTaskDelay(10 / portTICK_PERIOD_MS);
      continue;
    }

    int16_t gz_raw = Wire.read() << 8 | Wire.read();

    float turnRate = (gz_raw - gyroZ_offset) / 131.0f;
    filteredRate = 0.9f * filteredRate + 0.1f * turnRate;

    unsigned long now = millis();
    yaw += filteredRate * ((now - lastTime) / 1000.0f);
    lastTime = now;

    if (yaw > 180.0f)
      yaw -= 360.0f;
    if (yaw < -180.0f)
      yaw += 360.0f;

    vTaskDelay(10 / portTICK_PERIOD_MS);
  }
}
