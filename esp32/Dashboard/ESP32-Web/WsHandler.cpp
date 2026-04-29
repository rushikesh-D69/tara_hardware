// ─────────────────────────────────────────────────────────────────────────────
//  WsHandler.cpp — WebSocket event processing
//
//  ARCHITECTURE: Single /ws endpoint, two logical clients.
//
//  All three nodes connect to ws://ESP32_IP/ws :
//    1. Dashboard PC  — browser, sends drive/nav/mode JSON
//    2. Raspberry Pi  — Python ws_bridge, sends auto_cmd JSON
//
//  CLIENT IDENTIFICATION (registration handshake):
//    RPi sends immediately on connect:
//      {"type":"register","role":"rpi"}
//    ESP32 stores that client's ID in wsRpiClientId.
//    Any other client (or unregistered) → treated as Dashboard.
//
//  ROUTING RULES:
//    Source\Mode │  MANUAL          │  AUTO
//    ────────────┼──────────────────┼──────────────────────────────
//    Dashboard   │  drive/nav/speed │  read telemetry only, no move
//    RPi         │  set_mode only   │  auto_cmd → controlQueue
//                │  (ignored move)  │  (heartbeat feeds watchdog)
//
//  COMMANDS (Dashboard → ESP32):
//    {"type":"register","role":"rpi"}     ← registers this client as RPi
//    {"type":"set_mode","mode":"auto"|"manual"}
//    {"type":"drive","x":0.0,"y":0.0}     ← MANUAL only
//    {"type":"speed","value":0.8}         ← MANUAL only
//    {"type":"estop","state":true}        ← always
//    {"type":"ppr","value_l":18,"value_r":18}
//    {"type":"heartbeat"}                 ← always (WD feed)
//    {"type":"goto","dist":0.5,"speed":0.25}   ← MANUAL only
//    {"type":"turn","angle":90,"speed":0.20}   ← MANUAL only
//    {"type":"nav_stop"}                  ← always
//    {"type":"sequence","steps":[...]}   ← MANUAL only
//
//  COMMANDS (RPi → ESP32):
//    {"type":"register","role":"rpi"}     ← sent once on connect
//    {"type":"set_mode","mode":"auto"}    ← switch to AUTO
//    {"type":"auto_cmd","x":-1..1,"y":0..1}  ← AUTO movement (50 Hz)
//    {"type":"heartbeat"}                 ← keepalive
//    {"type":"estop","state":true}        ← always
// ─────────────────────────────────────────────────────────────────────────────

#include "WsHandler.h"
#include <ArduinoJson.h>
#include "Globals.h"
#include "Navigation.h"

// ── Helper: broadcast mode to ALL clients ────────────────────────────────────
static void broadcastMode(const char *reason = nullptr) {
  char buf[128];
  if (reason) {
    snprintf(buf, sizeof(buf),
             "{\"type\":\"mode\",\"mode\":\"%s\",\"reason\":\"%s\"}",
             driveMode == MODE_AUTO ? "auto" : "manual", reason);
  } else {
    snprintf(buf, sizeof(buf),
             "{\"type\":\"mode\",\"mode\":\"%s\"}",
             driveMode == MODE_AUTO ? "auto" : "manual");
  }
  if (ws.count() > 0) ws.textAll(buf);
}

// ── Helper: is this client the registered RPi? ───────────────────────────────
static inline bool isRpi(uint32_t clientId) {
  return (wsRpiClientId != 0 && clientId == wsRpiClientId);
}

// ─────────────────────────────────────────────────────────────────────────────
void onWsEvent(AsyncWebSocket *srv, AsyncWebSocketClient *client,
               AwsEventType type, void *arg, uint8_t *data, size_t len) {

  const uint32_t cid = client->id();

  // ── Client connected ──────────────────────────────────────────────────────
  if (type == WS_EVT_CONNECT) {
    Serial.printf("[WS] Client #%u connected from %s\n",
                  cid, client->remoteIP().toString().c_str());
    // Push current mode immediately so new client syncs
    char buf[64];
    snprintf(buf, sizeof(buf), "{\"type\":\"mode\",\"mode\":\"%s\"}",
             driveMode == MODE_AUTO ? "auto" : "manual");
    client->text(buf);
    return;
  }

  // ── Client disconnected ───────────────────────────────────────────────────
  if (type == WS_EVT_DISCONNECT) {
    Serial.printf("[WS] Client #%u disconnected\n", cid);
    if (cid == wsRpiClientId) {
      wsRpiClientId = 0;
      Serial.println("[WS] RPi client gone — immediate MANUAL fallback");
      // Immediately stop & revert — don't wait for the watchdog timer
      if (driveMode == MODE_AUTO) {
        driveMode = MODE_MANUAL;
        JoyData stop = {0.0f, 0.0f};
        xQueueOverwrite(controlQueue, &stop);
        if (ws.count() > 0)
          ws.textAll("{\"type\":\"mode\",\"mode\":\"manual\",\"reason\":\"rpi_disconnected\"}");
      }
    }
    return;
  }

  if (type != WS_EVT_DATA) return;

  // ── Route by source ───────────────────────────────────────────────────────
  const bool fromRpi = isRpi(cid);

  // Update appropriate watchdog timestamp
  if (fromRpi) {
    lastRpiCmd = millis();          // RPi is alive
  } else {
    lastWsMessage = millis();       // Dashboard is alive
  }

  // ── Parse JSON ────────────────────────────────────────────────────────────
  StaticJsonDocument<1024> doc;
  if (deserializeJson(doc, data, len)) return;

  const char *cmd = doc["type"];
  if (!cmd) return;

  // ─────────────────────────────────────────────────────────────────────────
  //  ALWAYS-ACCEPTED (any source, any mode)
  // ─────────────────────────────────────────────────────────────────────────

  // Registration handshake — RPi identifies itself
  if (strcmp(cmd, "register") == 0) {
    const char *role = doc["role"] | "";
    if (strcmp(role, "rpi") == 0) {
      wsRpiClientId = cid;
      Serial.printf("[WS] RPi registered as client #%u\n", cid);
      // Confirm back to RPi
      client->text("{\"type\":\"registered\",\"role\":\"rpi\"}");
    }
    return;
  }

  // Heartbeat — just a keepalive, already updated watchdog above
  if (strcmp(cmd, "heartbeat") == 0) return;

  // E-stop — always honoured regardless of source or mode
  if (strcmp(cmd, "estop") == 0) {
    eStopActive = (bool)doc["state"];
    if (eStopActive) {
      JoyData stop = {0.0f, 0.0f};
      xQueueOverwrite(controlQueue, &stop);
    }
    return;
  }

  // Mode toggle — both Dashboard and RPi can request a mode change
  if (strcmp(cmd, "set_mode") == 0) {
    const char *m = doc["mode"] | "manual";
    if (strcmp(m, "auto") == 0) {
      driveMode = MODE_AUTO;
      lastRpiCmd = millis();   // reset watchdog — give RPi 5 s to connect & send
      JoyData stop = {0.0f, 0.0f};
      xQueueOverwrite(controlQueue, &stop);
      Serial.printf("[MODE] AUTO (client #%u %s)\n", cid,
                    fromRpi ? "RPi" : "Dashboard");
    } else {
      driveMode = MODE_MANUAL;
      Serial.printf("[MODE] MANUAL (client #%u %s)\n", cid,
                    fromRpi ? "RPi" : "Dashboard");
    }
    broadcastMode();
    return;
  }

  // Nav stop — always accepted
  if (strcmp(cmd, "nav_stop") == 0) {
    navMode = NAV_IDLE;
    seqLen  = seqIdx = 0;
    navStatus = 0; navProgress = 0.0f;
    pidL.reset(); pidR.reset();
    Serial.println("[NAV] ABORTED");
    return;
  }

  // PPR config — always accepted from Dashboard only (safety)
  if (strcmp(cmd, "ppr") == 0 && !fromRpi) {
    new_PPR_L = (float)doc["value_l"];
    new_PPR_R = (float)doc["value_r"];
    pprUpdatePending = true;
    Serial.printf("[CFG] PPR L:%.1f R:%.1f\n", new_PPR_L, new_PPR_R);
    return;
  }

  // ─────────────────────────────────────────────────────────────────────────
  //  RPi-ONLY commands (AUTO mode movement)
  // ─────────────────────────────────────────────────────────────────────────
  if (fromRpi) {
    if (strcmp(cmd, "auto_cmd") == 0) {
      if (driveMode != MODE_AUTO) {
        // RPi is sending movement but we're in MANUAL — silently discard
        return;
      }
      JoyData jd = {
        constrain((float)doc["x"], -1.0f,  1.0f),
        constrain((float)doc["y"],  0.0f,  1.0f),
      };
      xQueueOverwrite(controlQueue, &jd);
      lastRpiCmd = millis();   // explicit update (already done above, belt+suspenders)
      return;
    }
    // Any other RPi message not handled above → ignore
    return;
  }

  // ─────────────────────────────────────────────────────────────────────────
  //  DASHBOARD-ONLY commands
  //  Below this point: fromRpi == false (Dashboard client)
  // ─────────────────────────────────────────────────────────────────────────

  // In AUTO mode, dashboard movement/nav commands are silently ignored
  if (driveMode == MODE_AUTO) {
    // Dashboard can still send heartbeat / estop / set_mode (handled above)
    return;
  }

  // ── MANUAL-only commands ─────────────────────────────────────────────────
  if (strcmp(cmd, "drive") == 0) {
    JoyData jd = {(float)doc["x"], (float)doc["y"]};
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
    Serial.printf("[NAV] GOTO %.2f m @ %.2f\n", navGoalDist, navSpeed);
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
