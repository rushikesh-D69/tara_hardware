#include "WsHandler.h"
#include <ArduinoJson.h>
#include "Globals.h"
#include "Navigation.h"

void onWsEvent(AsyncWebSocket *srv, AsyncWebSocketClient *client, AwsEventType type, void *arg, uint8_t *data, size_t len) {
  if (type == WS_EVT_DATA) {
    lastWsMessage = millis();

    StaticJsonDocument<1024> doc;
    if (deserializeJson(doc, data, len)) return;

    const char *cmd = doc["type"];
    if (strcmp(cmd, "drive") == 0) {
      JoyData jd = {doc["x"], doc["y"]};
      xQueueOverwrite(controlQueue, &jd);
    } else if (strcmp(cmd, "heartbeat") == 0) {
    } else if (strcmp(cmd, "ppr") == 0) {
      new_PPR_L = (float)doc["value_l"];
      new_PPR_R = (float)doc["value_r"];
      pprUpdatePending = true;
      Serial.printf("[CONFIG] PPR updated L:%.1f R:%.1f\n", new_PPR_L, new_PPR_R);
    } else if (strcmp(cmd, "speed") == 0) {
      baseSpeed = doc["value"];
    } else if (strcmp(cmd, "estop") == 0) {
      eStopActive = (bool)doc["state"];
    } else if (strcmp(cmd, "goto") == 0) {
      seqLen = seqIdx = 0;
      navMode = NAV_GOTO;
      navGoalDist = (float)doc["dist"];
      navSpeed = doc["speed"] | 0.25f;
      gotoPrevSpeed = navSpeed;
      portDISABLE_INTERRUPTS();
      navStartPulseL = pulseLeft;
      navStartPulseR = pulseRight;
      portENABLE_INTERRUPTS();
      smoothHeadingCorrection = 0.0f;
      settleCount = 0;
      pidL.reset(); pidR.reset();
      Serial.printf("[NAV] GOTO %.2f m @ %.2f m/s\n", navGoalDist, navSpeed);
    } else if (strcmp(cmd, "turn") == 0) {
      seqLen = seqIdx = 0;
      float angle = doc["angle"];
      float speed = doc["speed"] | 0.20f;
      pidL.reset(); pidR.reset();
      startTurn(angle, speed);
    } else if (strcmp(cmd, "nav_stop") == 0) {
      navMode = NAV_IDLE;
      seqLen = seqIdx = 0;
      navStatus = 0; navProgress = 0.0f;
      pidL.reset(); pidR.reset();
      Serial.println("[NAV] ABORTED");
    } else if (strcmp(cmd, "sequence") == 0) {
      JsonArray steps = doc["steps"].as<JsonArray>();
      seqLen = min((int)steps.size(), MAX_SEQ_STEPS);
      for (int i = 0; i < seqLen; i++) {
        const char *t = steps[i]["type"] | "goto";
        navSequence[i].speed = steps[i]["speed"] | 0.25f;
        if (strcmp(t, "goto") == 0) {
          navSequence[i].type = STEP_GOTO;
          navSequence[i].param = steps[i]["dist"];
        } else {
          navSequence[i].type = STEP_TURN;
          navSequence[i].param = steps[i]["angle"];
        }
      }
      seqIdx = 0;
      if (seqLen > 0) {
        expectedX = posX;
        expectedY = posY;
        expectedTheta = ekf_theta;
        _navStartStep(0);
        pidL.reset(); pidR.reset();
      }
      Serial.printf("[NAV] SEQUENCE %d steps\n", seqLen);
    }
  }
}
