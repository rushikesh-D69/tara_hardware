#ifndef NAVIGATION_H
#define NAVIGATION_H

#include <Arduino.h>

void ekfPredict(float dl, float dr, float dt);
void ekfUpdateYaw(float yaw_measured);
void resetOdometry();
float _yawDiff(float target, float current);
void startTurn(float degrees, float speed);
void _navStartStep(int idx);
void _navStepComplete();

struct PIDCtrl {
  float kp, ki, kd;
  float integral = 0.0f;
  float prevError = 0.0f;
  float lastSetpoint = 0.0f;
  float lastError = 0.0f;
  const float I_MAX = 80.0f;

  float compute(float setpoint, float measured, float dt) {
    if (dt <= 0.0f)
      return 0.0f;
    lastSetpoint = setpoint;
    float err = setpoint - measured;
    lastError = err;
    integral = constrain(integral + err * dt, -I_MAX, I_MAX);
    float deriv = (err - prevError) / dt;
    prevError = err;
    return kp * err + ki * integral + kd * deriv;
  }
  void reset() {
    integral = 0.0f;
    prevError = 0.0f;
  }
};

extern PIDCtrl pidL;
extern PIDCtrl pidR;
extern PIDCtrl pidPos;

#endif // NAVIGATION_H
