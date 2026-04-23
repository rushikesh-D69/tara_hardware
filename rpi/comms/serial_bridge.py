"""
TARA ADAS — Serial Bridge (RPi ↔ ESP32)
Handles bidirectional communication over USB serial.

Protocol:
  RPi → ESP32:  CMD:<steer>,<speed>,<flags>\n
  ESP32 → RPi:  SEN:<dist>,<left_enc>,<right_enc>,<ax>,<ay>,<az>\n
"""
import serial
import threading
import time
from utils.logger import get_logger

log = get_logger("Serial")


class SerialBridge:
    """
    Manages serial communication between RPi 4B and ESP32.
    Runs sensor reading in a background thread.
    """

    def __init__(self, port="/dev/ttyUSB0", baud=115200, timeout=0.01):
        """
        Args:
            port: Serial port path
            baud: Baud rate (must match ESP32 firmware)
            timeout: Read timeout in seconds
        """
        self.port = port
        self.baud = baud
        self.timeout = timeout

        self._serial = None
        self._running = False
        self._read_thread = None

        # Latest sensor data (thread-safe)
        self._sensor_data = {
            'distance_cm': 999.0,
            'left_enc': 0,
            'right_enc': 0,
            'yaw': 0.0,
        }
        self._sensor_lock = threading.Lock()
        self._last_sensor_time = 0

    def connect(self):
        """Open the serial connection to ESP32."""
        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                timeout=self.timeout,
                write_timeout=0.1,
            )
            time.sleep(1.0)  # Wait for ESP32 to reset after connection

            # Flush any startup garbage
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()

            log.info(f"Serial connected: {self.port} @ {self.baud} baud")

            # Start background sensor reading thread
            self._running = True
            self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
            self._read_thread.start()

            return True

        except serial.SerialException as e:
            log.error(f"Failed to connect to {self.port}: {e}")
            return False

    def _read_loop(self):
        """Background thread: continuously read sensor data from ESP32."""
        buffer = ""
        while self._running:
            try:
                if self._serial and self._serial.in_waiting > 0:
                    raw = self._serial.read(self._serial.in_waiting)
                    buffer += raw.decode('ascii', errors='ignore')

                    # Process complete lines
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        if line.startswith("SEN:"):
                            self._parse_sensor_data(line)
                else:
                    time.sleep(0.005)  # 5ms sleep to avoid busy-waiting

            except Exception as e:
                log.error(f"Serial read error: {e}")
                time.sleep(0.1)

    def _parse_sensor_data(self, line):
        """
        Parse sensor data line from ESP32.
        Format: SEN:<dist>,<left_enc>,<right_enc>,<ax>,<ay>,<az>

        Args:
            line: Raw sensor data string
        """
        try:
            # Updated parsing logic:
            parts = line[4:].split(',')
            if len(parts) >= 4:
                data = {
                    'distance_cm': float(parts[0]),
                    'left_enc': int(parts[1]),
                    'right_enc': int(parts[2]),
                    'yaw': float(parts[3]),  # Rotation angle from IMU
                }
                with self._sensor_lock:
                    self._sensor_data = data
                    self._last_sensor_time = time.monotonic()
        except (ValueError, IndexError) as e:
            log.debug(f"Malformed sensor data: {line} — {e}")

    def get_sensor_data(self):
        """
        Get the latest sensor readings (thread-safe).

        Returns:
            Dict with sensor values, or None if data is stale (>500ms old)
        """
        with self._sensor_lock:
            age = time.monotonic() - self._last_sensor_time
            if age > 0.5:  # Data older than 500ms is stale
                return None
            return self._sensor_data.copy()

    def send_command(self, command):
        """
        Send a motor command to ESP32.

        Args:
            command: Command object with .to_serial() method,
                     or a raw string like "CMD:0,150,0"
        """
        if self._serial is None or not self._serial.is_open:
            log.warning("Serial not connected, cannot send command")
            return False

        try:
            if hasattr(command, 'to_serial'):
                cmd_str = command.to_serial()
            else:
                cmd_str = str(command)

            self._serial.write(f"{cmd_str}\n".encode('ascii'))
            return True

        except serial.SerialException as e:
            log.error(f"Serial write failed: {e}")
            return False

    def send_stop(self):
        """Send emergency stop command."""
        return self.send_command("CMD:0,0,4")

    @property
    def is_connected(self):
        """Whether the serial connection is active."""
        return self._serial is not None and self._serial.is_open

    def disconnect(self):
        """Stop the read thread and close serial connection."""
        log.info("Disconnecting serial...")
        self._running = False

        if self._read_thread is not None:
            self._read_thread.join(timeout=2.0)

        # Send stop command before disconnecting
        if self._serial and self._serial.is_open:
            try:
                self._serial.write(b"CMD:0,0,4\n")
                time.sleep(0.1)
            except Exception:
                pass
            self._serial.close()

        log.info("Serial disconnected")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()
