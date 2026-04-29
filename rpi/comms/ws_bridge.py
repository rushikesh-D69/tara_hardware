"""
TARA ADAS — WebSocket Bridge (RPi → ESP32, WiFi)
Replaces serial_bridge.py. No UART, no cable.

Architecture:
  All three nodes (ESP32, RPi, Dashboard PC) on the same WiFi network.
  RPi connects to the ESP32 WebSocket server as a client.

Protocol (RPi → ESP32, JSON):
  {"type":"set_mode","mode":"auto"}          — switch ESP32 to AUTO on connect
  {"type":"auto_cmd","x":steer,"y":speed}    — movement command (AUTO mode only)
  {"type":"heartbeat"}                       — keepalive every 1 s
  {"type":"estop","state":true}              — emergency stop (always accepted)

Protocol (ESP32 → RPi, receive-only):
  "SEN:..."                                  — telemetry for logging (not used for decisions)
  {"type":"mode","mode":"auto"|"manual"}     — mode change acknowledgement

Install dependency:
  pip install websocket-client
"""
import json
import threading
import time
from utils.logger import get_logger

log = get_logger("WsBridge")

try:
    import websocket
except ImportError:
    raise ImportError(
        "websocket-client not installed. Run: pip install websocket-client"
    )


class WsBridge:
    """
    WebSocket client that connects RPi ADAS to the ESP32 control node over WiFi.
    Runs a background thread for receiving messages.
    All ADAS movement commands are sent as JSON {"type":"auto_cmd",...}.
    """

    def __init__(self, host: str, port: int = 80, path: str = "/ws",
                 heartbeat_interval: float = 1.0,
                 min_send_interval: float = 0.02):
        """
        Args:
            host:               ESP32 IP address (e.g. "192.168.1.105")
            port:               HTTP port of ESP32 web server (default 80)
            path:               WebSocket path (default "/ws")
            heartbeat_interval: Seconds between heartbeat messages (default 1.0)
            min_send_interval:  Minimum seconds between command sends (default 0.02 = 50 Hz)
        """
        self.url = f"ws://{host}:{port}{path}"
        self._hb_interval = heartbeat_interval
        self._min_send_interval = min_send_interval

        self._ws: websocket.WebSocketApp | None = None
        self._ws_thread: threading.Thread | None = None
        self._hb_thread: threading.Thread | None = None
        self._running = False
        self._connected = False
        self._lock = threading.Lock()

        # Rate limiting
        self._last_send_time = 0.0

        # Parse statistics
        self._good_parses = 0
        self._send_count  = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """
        Open WebSocket connection to ESP32 and start background threads.
        Returns True if the connection was initiated (actual connection is async).
        """
        log.info(f"Connecting to ESP32 at {self.url} …")
        self._running = True

        self._ws = websocket.WebSocketApp(
            self.url,
            on_open    = self._on_open,
            on_message = self._on_message,
            on_error   = self._on_error,
            on_close   = self._on_close,
        )

        # Run WS in background thread
        self._ws_thread = threading.Thread(
            target=self._ws.run_forever,
            kwargs={"reconnect": 3},   # auto-reconnect every 3 s on drop
            daemon=True,
            name="WsBridgeRecv",
        )
        self._ws_thread.start()

        # Wait up to 4 s for the connection to establish
        deadline = time.monotonic() + 4.0
        while not self._connected and time.monotonic() < deadline:
            time.sleep(0.1)

        if self._connected:
            log.info("ESP32 WebSocket connected ✓")
            # Start heartbeat thread
            self._hb_thread = threading.Thread(
                target=self._heartbeat_loop,
                daemon=True,
                name="WsBridgeHB",
            )
            self._hb_thread.start()
            return True
        else:
            log.error(f"Could not connect to ESP32 at {self.url} within 4 s")
            return False

    def send_command(self, command) -> bool:
        """
        Send a movement command to the ESP32 over WebSocket.

        On the first call, injects set_mode=auto before the command so the
        ESP32 switches from MANUAL (its boot default) and accepts auto_cmd.

        Args:
            command: Command object with .steer_x / .speed_y attributes,
                     or a dict, or a raw JSON string.
        Returns:
            True if sent, False if not connected or rate-limited.
        """
        if not self._connected or self._ws is None:
            return False

        now = time.monotonic()
        if now - self._last_send_time < self._min_send_interval:
            return True   # skip — not an error
        self._last_send_time = now

        try:
            if hasattr(command, "steer") and hasattr(command, "speed"):
                x = round(max(-1.0, min(1.0, command.steer)), 4)
                y = round(max( 0.0, min(1.0, command.speed)), 4)
                payload = json.dumps({"type": "auto_cmd", "x": x, "y": y})
            elif isinstance(command, dict):
                payload = json.dumps(command)
            else:
                payload = str(command)

            with self._lock:
                if self._ws is None:
                    return False
                self._ws.send(payload)
            self._send_count += 1
            return True

        except Exception as e:
            log.error(f"WS send failed: {e}")
            self._connected = False  # mark disconnected so loop stops spamming
            return False

    def send_stop(self) -> bool:
        """Send emergency stop — bypasses rate limiting."""
        self._last_send_time = 0.0
        return self.send_command({"type": "estop", "state": True})

    def disconnect(self):
        """Stop heartbeat, close WebSocket connection."""
        log.info("Disconnecting WsBridge …")
        self._running = False
        self.send_stop()
        time.sleep(0.15)
        if self._ws:
            self._ws.close()
        log.info(f"WsBridge disconnected (sent {self._send_count} commands, "
                 f"parsed {self._good_parses} SEN packets)")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── WebSocket callbacks ───────────────────────────────────────────────────

    def _on_open(self, ws):
        with self._lock:
            self._ws = ws            # update to the (re)connected socket
        self._connected = True
        log.info("[WS] Connection opened")
        try:
            # Step 1: Register as RPi so ESP32 routes our messages correctly
            ws.send(json.dumps({"type": "register", "role": "rpi"}))
            log.info("[WS] Registered as RPi client")
            # Step 2: Request AUTO mode
            ws.send(json.dumps({"type": "set_mode", "mode": "auto"}))
            log.info("[WS] AUTO mode requested")
        except Exception as e:
            log.error(f"[WS] Failed to send handshake: {e}")


    def _on_message(self, ws, message: str):
        """Receive-only: log SEN: packets and mode changes for diagnostics."""
        msg = message.strip()
        if msg.startswith("SEN:"):
            self._good_parses += 1
            log.debug(f"[WS] Telemetry: {msg[:60]}…")
        elif msg.startswith("{"):
            try:
                obj = json.loads(msg)
                if obj.get("type") == "mode":
                    log.info(f"[WS] ESP32 mode confirmed: {obj.get('mode')}")
            except json.JSONDecodeError:
                pass

    def _on_error(self, ws, error):
        log.error(f"[WS] Error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        self._connected = False
        with self._lock:
            self._ws = None          # prevent sends on dead socket
        log.warning(f"[WS] Connection closed (code={close_status_code}) — will auto-reconnect")

    # ── Heartbeat thread ──────────────────────────────────────────────────────

    def _heartbeat_loop(self):
        """Send heartbeat every _hb_interval seconds to prevent WS watchdog."""
        while self._running and self._connected:
            try:
                with self._lock:
                    self._ws.send(json.dumps({"type": "heartbeat"}))
            except Exception:
                pass
            time.sleep(self._hb_interval)

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()
