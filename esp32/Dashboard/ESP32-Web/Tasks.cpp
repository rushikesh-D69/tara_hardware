#include "Tasks.h"
#include <Arduino.h>
#include "Config.h"
#include "Globals.h"
#include "Navigation.h"

void ultrasonicTask(void *pvParameters) {
  while (true) {
    digitalWrite(TRIG_PIN, LOW);
    delayMicroseconds(2);
    digitalWrite(TRIG_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG_PIN, LOW);

    distanceCm = 999.0f;
    batVoltage = (analogRead(BAT_PIN) / 4095.0f) * 3.3f * (10.0f + 3.3f) / 3.3f;

    vTaskDelay(100 / portTICK_PERIOD_MS);
  }
}

void motorTask(void *pvParameters) {
  JoyData jd = {0.0f, 0.0f};
  unsigned long lastTime = millis();

  while (true) {
    if (pprUpdatePending) {
      PPR_L = new_PPR_L;
      PPR_R = new_PPR_R;
      M_PER_PULSE_L = WHEEL_CIRC / PPR_L;
      M_PER_PULSE_R = WHEEL_CIRC / PPR_R;
      pprUpdatePending = false;
    }

    unsigned long now = millis();
    float dt = (now - lastTime) / 1000.0f;
    if (dt < 0.001f)
      dt = 0.001f;
    lastTime = now;

    float taskYaw = yaw;
    float taskRate = filteredRate;

    portDISABLE_INTERRUPTS();
    long snapL = pulseLeft;
    long snapR = pulseRight;
    portENABLE_INTERRUPTS();

    long dL = snapL - prevPulseL;
    long dR = snapR - prevPulseR;
    prevPulseL = snapL;
    prevPulseR = snapR;

    vL_actual = motorDirL * (dL * M_PER_PULSE_L / dt);
    vR_actual = motorDirR * (dR * M_PER_PULSE_R / dt);

    v_linear = (vL_actual + vR_actual) / 2.0f;
    v_angular = taskRate * ((float)M_PI / 180.0f);

    float dl_dist = vL_actual * dt;
    float dr_dist = vR_actual * dt;
    ekfPredict(dl_dist, dr_dist, dt);
    ekfUpdateYaw(taskYaw);

    posX = ekf_x;
    posY = ekf_y;
    distTraveled += fabsf(v_linear * dt);

    // Watchdog: kill motors if no command received in 1s.
    // We check ws.count() only when WiFi is connected — if Wi-Fi is
    // disconnected AND no Serial commands are coming, the RPi link is dead.
    bool serialDead = (millis() - lastWsMessage > 1000);
    bool wifiLost = serialDead && (WiFi.status() == WL_CONNECTED ? true : ws.count() > 0);
    bool obstacle = (distanceCm < AUTO_STOP_DIST);

    if (eStopActive || wifiLost || obstacle) {
      currentL_PWM = 0;
      currentR_PWM = 0;
      pidL.reset();
      pidR.reset();
    } else {
      if (navMode == NAV_GOTO) {
        float distL = (snapL - navStartPulseL) * M_PER_PULSE_L;
        float distR = (snapR - navStartPulseR) * M_PER_PULSE_R;
        navTraveled = (distL + distR) / 2.0f;
        navProgress = (navGoalDist > 0.001f) ? constrain(navTraveled / navGoalDist, 0.0f, 1.0f) : 1.0f;
        navStatus = 1;

        float slipDiff = fabsf(vL_actual - vR_actual);
        if (slipDiff > 0.15f) {
          navSpeed = max(navSpeed * 0.95f, 0.10f);
        } else if (navSpeed < gotoPrevSpeed) {
          navSpeed = min(navSpeed * 1.02f, gotoPrevSpeed);
        }

        float batScale = constrain(batVoltage / 7.4f, 0.7f, 1.0f);
        float effectiveSpeed = navSpeed * batScale;

        float distErr = navGoalDist - navTraveled;

        if (distErr < 0.008f) {
          settleCount++;
          currentL_PWM = currentR_PWM = 0;
          if (settleCount >= 10) {
            settleCount = 0;
            pidL.reset(); pidR.reset(); pidPos.reset();
            smoothHeadingCorrection = 0.0f;
            _navStepComplete();
          }
        } else {
          settleCount = 0;
          float targetSpeed = pidPos.compute(distErr, 0.0f, dt);
          targetSpeed = constrain(targetSpeed, 0.0f, effectiveSpeed);

          float encDiff = distL - distR;
          float rawCorrection = encDiff * 150.0f;
          smoothHeadingCorrection = 0.7f * smoothHeadingCorrection + 0.3f * rawCorrection;
          smoothHeadingCorrection = constrain(smoothHeadingCorrection, -50.0f, 50.0f);

          float outL = pidL.compute(targetSpeed, vL_actual, dt);
          float outR = pidR.compute(targetSpeed, vR_actual, dt);

          int ff = (int)(targetSpeed * 1200.0f);
          ff = constrain(ff, 0, 255);

          currentL_PWM = constrain(ff + (int)outL - (int)smoothHeadingCorrection, 0, 255);
          currentR_PWM = constrain(ff + (int)outR + (int)smoothHeadingCorrection, 0, 255);
        }

      } else if (navMode == NAV_TURN) {
        navStatus = 2;

        if (turnFirstTick) {
          turnStartYaw = taskYaw;
          currentL_PWM = currentR_PWM = 0;
          if (millis() - navTurnStartMs >= 40) {
            turnFirstTick = false;
          }
        } else {
          float delta = taskYaw - turnStartYaw;
          delta = fmod(delta + 540.0f, 360.0f) - 180.0f;
          float yawErr = navTargetAngle - delta;

          if (fabsf(navTargetAngle) > 0.1f) {
            navProgress = constrain(1.0f - fabsf(yawErr) / fabsf(navTargetAngle), 0.0f, 1.0f);
          } else {
            navProgress = 1.0f;
          }

          bool turnTimeout = (navTurnStartMs > 0) && ((millis() - navTurnStartMs) > 6000UL);

          if (fabsf(yawErr) < TURN_TOL || turnTimeout) {
            if (turnTimeout)
              Serial.printf("[NAV] TURN timeout — got %.1f° of %.1f°\n", delta, navTargetAngle);
            currentL_PWM = currentR_PWM = 0;
            navTurnStartMs = 0;
            _navStepComplete();
          } else {
            float abserr = fabsf(yawErr);
            float proportion = constrain(abserr / 45.0f, 0.0f, 1.0f);
            float angVelAbs = fabsf(taskRate);

            float damping = constrain(angVelAbs / 60.0f, 0.0f, 0.6f);
            float blend = constrain(proportion - damping, 0.0f, 1.0f);

            int spinPwm = (int)(blend * 255.0f);

            if (spinPwm < 5 && abserr > TURN_TOL && angVelAbs < 1.0f) {
              spinPwm = 160;
            }

            float dir = (yawErr > 0) ? 1.0f : -1.0f;
            currentL_PWM = (int)(-dir * spinPwm);
            currentR_PWM = (int)(dir * spinPwm);
          }
        }
      } else {
        navStatus = 0;
        navProgress = 0.0f;
        xQueueReceive(controlQueue, &jd, 0);

        float targetVL = (jd.y + jd.x);
        float targetVR = (jd.y - jd.x);

        targetVL = constrain(targetVL, -1.0f, 1.0f);
        targetVR = constrain(targetVR, -1.0f, 1.0f);

        auto applyOpenLoop = [](float val, float maxPwm) -> int {
          if (fabsf(val) < 0.05f) return 0;
          int sign = (val > 0) ? 1 : -1;
          int floorPwm = 150; // Requested MIN PWM
          if (maxPwm < floorPwm) floorPwm = maxPwm;
          return sign * (floorPwm + (int)((fabsf(val) - 0.05f) * (maxPwm - floorPwm) / 0.95f));
        };

        const int MAX_PWM_VAL = 255;
        int pwmCeil = (int)(baseSpeed * MAX_PWM_VAL);
        currentL_PWM = applyOpenLoop(targetVL, pwmCeil);
        currentR_PWM = applyOpenLoop(targetVR, pwmCeil);
      }
    }

    const int FINAL_MAX_PWM = 255;
    currentL_PWM = constrain(currentL_PWM, -FINAL_MAX_PWM, FINAL_MAX_PWM);
    currentR_PWM = constrain(currentR_PWM, -FINAL_MAX_PWM, FINAL_MAX_PWM);

    motorDirL = (currentL_PWM >= 0) ? 1 : -1;
    motorDirR = (currentR_PWM >= 0) ? 1 : -1;

    digitalWrite(AIN1, currentR_PWM >= 0 ? HIGH : LOW);
    digitalWrite(AIN2, currentR_PWM >= 0 ? LOW : HIGH);
    ledcWrite(PWMA, abs(currentR_PWM));

    digitalWrite(BIN1, currentL_PWM >= 0 ? HIGH : LOW);
    digitalWrite(BIN2, currentL_PWM >= 0 ? LOW : HIGH);
    ledcWrite(PWMB, abs(currentL_PWM));

    vTaskDelay(20 / portTICK_PERIOD_MS);
  }
}
