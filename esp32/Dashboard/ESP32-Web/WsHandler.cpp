// ─────────────────────────────────────────────────────────────────────────────
//  WsHandler.cpp — WebSocket event processing
//
//  Message formats accepted from the dashboard:
//
//  CSV fast-path:  CMD:x,y,flags
//    • Only accepted in MODE_MANUAL (movement ignored in MODE_AUTO)
//    • flags bit 0x04 = estop (always accepted)
//
//  JSON commands:
//    {"type":"set_mode","mode":"auto"|"manual"}   ← mode toggle (always OK)
//    {"type":"drive","x":0.0,"y":0.0}             ← MANUAL only
//    {"type":"speed","value":0.8}                 ← MANUAL only
//    {"type":"estop","state":true}                ← always accepted
//    {"type":"ppr","value_l":18,"value_r":18}     ← always (config)
//    {"type":"heartbeat"}                         ← always (WD feed)
//    {"type":"goto","dist":0.5,"speed":0.25}      ← MANUAL only
//    {"type":"turn","angle":90,"speed":0.20}      ← MANUAL only
//    {"type":"nav_stop"}                          ← always
//    {"type":"sequence","steps":[...]}            ← MANUAL only
// ─────────────────────────────────────────────────────────────────────────────

#include "WsHandler.h"
#include <ArduinoJson.h>
#include "Globals.h"
#include "Navigation.h"

// Helper: send mode status to all WS clients
static void broadcastMode() {
  char buf[64];
  snprintf(buf, sizeof(buf), "{\"type\":\"mode\",\"mode\":\"%s\"}",
           driveMode == MODE_AUTO ? "auto" : "manual");
  if (ws.count() > 0) ws.textAll(buf);
}

void onWsEvent(AsyncWebSocket *srv, AsyncWebSocketClient *client,
               AwsEventType type, void *arg, uint8_t *data, size_t len) {

  if (type == WS_EVT_CONNECT) {
    // On new connection, immediately push current mode so the dashboard syncs
    char buf[64];
    snprintf(buf, sizeof(buf), "{\"type\":\"mode\",\"mode\":\"%s\"}",
             driveMode == MODE_AUTO ? "auto" : "manual");
    client->text(buf);
    return;
  }

  if (type != WS_EVT_DATA) return;

  lastWsMessage = millis();  // keep WS watchdog alive

  // ── Fast-path: CMD:x,y,flags ─────────────────────────────────────────────
  if (len > 4 && data[0]=='C' && data[1]=='M' && data[2]=='D' && data[3]==':') {
    float x, y; int flags;
    if (sscanf((char*)data, "CMD:%f,%f,%d", &x, &y, &flags) == 3) {
      // E-stop always honoured
      eStopActive = (flags & 0x04) != 0;

      // Movement only in MANUAL mode
      if (driveMode == MODE_MANUAL) {
        JoyData jd = {x, y};
        xQueueOverwrite(controlQueue, &jd);
      }
    }
    return;
  }

  // ── JSON commands ─────────────────────────────────────────────────────────
  StaticJsonDocument<1024> doc;
  if (deserializeJson(doc, data, len)) return;

  const char *cmd = doc["type"];
  if (!cmd) return;

  // ── Mode toggle — ALWAYS accepted from dashboard ──────────────────────────
  if (strcmp(cmd, "set_mode") == 0) {
    const char *m = doc["mode"] | "manual";
    if (strcmp(m, "auto") == 0) {
      driveMode = MODE_AUTO;
      // Clear any pending manual joystick
      JoyData stop = {0.0f, 0.0f};
      xQueueOverwrite(controlQueue, &stop);
      Serial.println("[MODE] AUTO (dashboard)");
    } else {
      driveMode = MODE_MANUAL;
      Serial.println("[MODE] MANUAL (dashboard)");
    }
    broadcastMode();
    return;
  }

  // ── Always-accepted commands ──────────────────────────────────────────────
  if (strcmp(cmd, "heartbeat") == 0) {
    return;  // lastWsMessage already updated above
  }
  if (strcmp(cmd, "estop") == 0) {
    eStopActive = (bool)doc["state"];
    if (eStopActive) {
      JoyData stop = {0.0f, 0.0f};
      xQueueOverwrite(controlQueue, &stop);
    }
    return;
  }
  if (strcmp(cmd, "ppr") == 0) {
    new_PPR_L = (float)doc["value_l"];
    new_PPR_R = (float)doc["value_r"];
    pprUpdatePending = true;
    Serial.printf("[CFG] PPR L:%.1f R:%.1f\n", new_PPR_L, new_PPR_R);
    return;
  }
  if (strcmp(cmd, "nav_stop") == 0) {
    navMode = NAV_IDLE;
    seqLen  = seqIdx = 0;
    navStatus = 0; navProgress = 0.0f;
    pidL.reset(); pidR.reset();
    Serial.println("[NAV] ABORTED (dashboard)");
    return;
  }

  // ── MANUAL-only commands — silently ignore in AUTO mode ──────────────────
  if (driveMode == MODE_AUTO) {
    // In AUTO mode the dashboard cannot issue movement or nav commands.
    // (It can still see telemetry and toggle mode.)
    Serial.printf("[WS] Ignoring '%s' — in AUTO mode\n", cmd);
    return;
  }

  if (strcmp(cmd, "drive") == 0) {
    JoyData jd = {doc["x"], doc["y"]};
    xQueueOverwrite(controlQueue, &jd);
  }
  else if (strcmp(cmd, "speed") == 0) {
    baseSpeed = constrain((float)doc["value"], 0.0f, 1.0f);
  }
  else if (strcmp(cmd, "goto") == 0) {
    seqLen = seqIdx = 0;
    navMode      = NAV_GOTO;
    navGoalDist  = (float)doc["dist"];
    navSpeed     = doc["speed"] | 0.25f;
    gotoPrevSpeed = navSpeed;
    portDISABLE_INTERRUPTS();
    navStartPulseL = pulseLeft;
    navStartPulseR = pulseRight;
    portENABLE_INTERRUPTS();
    smoothHeadingCorrection = 0.0f;
    settleCount = 0;
    pidL.reset(); pidR.reset();
    Serial.printf("[NAV] GOTO %.2f m @ %.2f m/s\n", navGoalDist, navSpeed);
  }
  else if (strcmp(cmd, "turn") == 0) {
    seqLen = seqIdx = 0;
    float angle = doc["angle"];
    float speed = doc["speed"] | 0.20f;
    pidL.reset(); pidR.reset();
    startTurn(angle, speed);
  }
  else if (strcmp(cmd, "sequence") == 0) {
    JsonArray steps = doc["steps"].as<JsonArray>();
    seqLen = min((int)steps.size(), MAX_SEQ_STEPS);
    for (int i = 0; i < seqLen; i++) {
      const char *t = steps[i]["type"] | "goto";
      navSequence[i].speed = steps[i]["speed"] | 0.25f;
      if (strcmp(t, "goto") == 0) {
        navSequence[i].type  = STEP_GOTO;
        navSequence[i].param = steps[i]["dist"];
      } else {
        navSequence[i].type  = STEP_TURN;
        navSequence[i].param = steps[i]["angle"];
      }
    }
    seqIdx = 0;
    if (seqLen > 0) {
      expectedX     = posX;
      expectedY     = posY;
      expectedTheta = ekf_theta;
      _navStartStep(0);
      pidL.reset(); pidR.reset();
    }
    Serial.printf("[NAV] SEQUENCE %d steps\n", seqLen);
  }
}
