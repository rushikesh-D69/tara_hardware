# esp32/ — ESP32 Firmware & Dashboard

This directory contains the ESP32-side firmware and the browser dashboard for
the TARA robot.

---

## Contents

```
esp32/
└── Dashboard/      # Browser-based robot control dashboard
```

---

## Dashboard

The dashboard is a single-page web application served by the ESP32's built-in
AsyncWebServer. It provides:

- **Live MJPEG video stream** from the Raspberry Pi (port 5000)
- **Manual keyboard control** (arrow keys → `jd.x` / `jd.y` commands)
- **Mode switching** (AUTO / MANUAL)
- **Telemetry display** (sensor readings, mode state)

### Accessing the Dashboard

1. Connect all devices to the same WiFi network.
2. Flash the ESP32 firmware.
3. Open a browser and navigate to `http://<ESP32_IP>/` (default port 80).

---

## ESP32 WebSocket Protocol

The ESP32 runs a WebSocket server at `/ws` and accepts the following JSON messages:

| Message | Sender | Effect |
|---------|--------|--------|
| `{"type":"register","role":"rpi"}` | RPi | Routes messages correctly |
| `{"type":"set_mode","mode":"auto"}` | RPi / Dashboard | Switch to autonomous mode |
| `{"type":"auto_cmd","x":f,"y":f}` | RPi | Steering + throttle (AUTO only) |
| `{"type":"cmd","x":f,"y":f}` | Dashboard | Manual steering + throttle |
| `{"type":"heartbeat"}` | RPi | Keepalive |
| `{"type":"estop","state":true}` | Any | Immediate motor stop |

### ESP32 Motor Mapping

```
targetVL = jd.y + jd.x   (left wheels)
targetVR = jd.y - jd.x   (right wheels)
```

Both values are normalised: `jd.x ∈ [-1, 1]`, `jd.y ∈ [0, 1]`.
