#include "Navigation.h"
#include <Arduino.h>
#include <math.h>
#include "Globals.h"

PIDCtrl pidL = {50.0f, 10.0f, 0.5f};
PIDCtrl pidR = {50.0f, 10.0f, 0.5f};
PIDCtrl pidPos = {1.2f, 0.8f, 0.3f};

void ekfPredict(float dl, float dr, float dt) {
  if (dt <= 0.0f)
    return;
  float ds = (dl + dr) / 2.0f;
  float dtheta = (dr - dl) / WHEEL_BASE;

  ekf_x += ds * cosf(ekf_theta + dtheta / 2.0f);
  ekf_y += ds * sinf(ekf_theta + dtheta / 2.0f);
  ekf_theta += dtheta;

  P[0] += Q_pos;
  P[4] += Q_pos;
  P[8] += Q_theta;
}

void ekfUpdateYaw(float yaw_measured) {
  float yaw_measured_rad = yaw_measured * (float)M_PI / 180.0f;
  float innov = yaw_measured_rad - ekf_theta;
  innov = fmod(innov + (float)M_PI, 2.0f * (float)M_PI);
  if (innov < 0)
    innov += 2.0f * (float)M_PI;
  innov -= (float)M_PI;

  float K = P[8] / (P[8] + R_theta);
  ekf_theta += K * innov;
  P[8] *= (1.0f - K);
}

void resetOdometry() {
  ekf_x = 0.0f; ekf_y = 0.0f; ekf_theta = 0.0f;
  posX = 0.0f; posY = 0.0f;
  P[0] = 1; P[1] = 0; P[2] = 0;
  P[3] = 0; P[4] = 1; P[5] = 0;
  P[6] = 0; P[7] = 0; P[8] = 1;
  expectedX = 0.0f; expectedY = 0.0f; expectedTheta = 0.0f;
  distTraveled = 0.0f; theta = 0.0f;
}

float _yawDiff(float target, float current) {
  return fmod(target - current + 540.0f, 360.0f) - 180.0f;
}

void startTurn(float degrees, float speed) {
  navMode = NAV_TURN;
  navTargetAngle = degrees;
  navTurnInitErr = fabsf(degrees);
  navTurnSpeed = speed;
  turnFirstTick = true;
  turnStartYaw = yaw;
  navTurnStartMs = millis();
  navStatus = 2;
  navProgress = 0.0f;
  Serial.printf("[NAV] TURN %.1f deg\n", degrees);
}

void _navStartStep(int idx) {
  smoothHeadingCorrection = 0.0f;
  settleCount = 0;
  navTurnStartMs = 0;
  NavCmd &c = navSequence[idx];
  if (c.type == STEP_GOTO) {
    navMode = NAV_GOTO;
    navGoalDist = c.param;
    navStartPulseL = pulseLeft;
    navStartPulseR = pulseRight;
    navSpeed = c.speed;
    gotoPrevSpeed = c.speed;
    expectedX += c.param * cosf(expectedTheta);
    expectedY += c.param * sinf(expectedTheta);
  } else {
    startTurn(c.param, c.speed);
    expectedTheta += c.param * ((float)M_PI / 180.0f);
  }
  Serial.printf("[NAV] Step %d: %s %.2f\n", idx, c.type == STEP_GOTO ? "GOTO" : "TURN", c.param);
}

void _navStepComplete() {
  if (seqLen > 0) {
    if (navMode == NAV_GOTO || navMode == NAV_TURN) {
      float posError = sqrtf(powf(posX - expectedX, 2) + powf(posY - expectedY, 2));
      if (posError > 0.15f) {
        Serial.printf("[NAV] DRIFT ABORT — position error %.3fm exceeds 15cm\n", posError);
        navMode = NAV_IDLE;
        seqLen = seqIdx = 0;
        navStatus = 0; navProgress = 0.0f;
        pidL.reset(); pidR.reset();
        return;
      }
    }
    seqIdx++;
    if (seqIdx >= seqLen) {
      navMode = NAV_IDLE;
      seqLen = seqIdx = 0;
      navStatus = 0; navProgress = 0.0f;
      Serial.println("[NAV] Sequence COMPLETE");
    } else {
      _navStartStep(seqIdx);
    }
  } else {
    navMode = NAV_IDLE;
    navStatus = 0; navProgress = 0.0f;
    Serial.println("[NAV] Step COMPLETE");
  }
  pidL.reset(); pidR.reset();
}
