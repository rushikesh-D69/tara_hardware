# comms/ — ESP32 WebSocket Bridge

Provides `WsBridge`, a thread-safe WebSocket client that sends normalised
`(steer_x, speed_y)` commands to the ESP32 over WiFi.

---

## Protocol

### RPi -> ESP32 (JSON messages)

| Message | Trigger |
|---------|---------|
| `{"type":"register","role":"rpi"}` | On connect — routes future messages correctly |
| `{"type":"set_mode","mode":"auto"}` | Switch ESP32 to autonomous mode |
| `{"type":"auto_cmd","x": f, "y": f}` | Movement command (AUTO mode only) |
| `{"type":"heartbeat"}` | Sent every 1 s — prevents ESP32 timeout |
| `{"type":"estop","state": true}` | Emergency stop — bypasses rate limiter |

### ESP32 -> RPi (receive-only, informational)

| Message | Purpose |
|---------|---------|
| `SEN:dist,encL,encR,ax,ay,az\n` | Sensor telemetry (logged but not used for decisions) |
| `{"type":"mode","mode":"auto"\|"manual"}` | Mode change acknowledgement |

---

## Motor Mapping (on ESP32)

The ESP32 converts the two normalised floats to PWM:

```
jd.x  = steer_x   in [-1.0, +1.0]   (negative = left)
jd.y  = speed_y   in [ 0.0, +1.0]   (0 = stop)

targetVL = jd.y + jd.x              (left-wheel duty cycle)
targetVR = jd.y - jd.x              (right-wheel duty cycle)

PWM_left  = clamp(targetVL, 0, 1) * MAX_PWM
PWM_right = clamp(targetVR, 0, 1) * MAX_PWM
```

Differential steering: positive `jd.x` speeds up the right wheel relative to the left, turning the car left. Negative `jd.x` does the opposite.

---

## Rate Limiting

Commands are rate-limited to prevent WebSocket congestion:

```
if (now - last_send_time) < min_send_interval (= 0.05 s = 50 ms = 20 Hz):
    drop command
else:
    send and update last_send_time
```

Emergency stop (`send_stop()`) bypasses this check entirely.

---

## Reconnection

`ws.run_forever(reconnect=3)` is passed as the reconnect interval. The WS library backs off and attempts reconnection every 3 s if the ESP32 drops the connection (e.g., reboot or WiFi blip). The ADAS pipeline continues running in `--no-wifi` fallback mode during the reconnection window.

---

## Usage

```python
from comms.ws_bridge import WsBridge

ws = WsBridge(host="192.168.43.40", port=80, path="/ws")
if ws.connect():
    ws.send_command(command)   # command.to_ws() called internally
    ws.send_stop()
    ws.disconnect()
```

Context manager:

```python
with WsBridge(host="192.168.43.40") as ws:
    ws.send_command(command)
```

---

## Heartbeat

A daemon thread sends `{"type":"heartbeat"}` every 1 s independently of the command loop. The ESP32 uses this to detect RPi disconnection and apply a safety stop after a configurable timeout.
