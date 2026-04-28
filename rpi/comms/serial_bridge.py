"""
TARA ADAS — Serial Bridge (RPi ↔ ESP32)
Handles bidirectional communication over USB serial.

Protocol:
  RPi → ESP32:  CMD:<steer>,<speed>,<flags>\n
  ESP32 → RPi:  SEN:<20 comma-separated fields>\n

ESP32 SEN packet fields (from ESP32-Web.ino loop()):
  [0]  v_linear      (m/s)
  [1]  baseSpeed      (0.0–1.0)
  [2]  yaw            (degrees, IMU)
  [3]  filteredRate   (deg/s, IMU yaw rate)
  [4]  v_angular      (rad/s)
  [5]  currentL_PWM   (int)
  [6]  currentR_PWM   (int)
  [7]  pulseLeft      (long, encoder ticks)
  [8]  pulseRight     (long, encoder ticks)
  [9]  distTraveled   (m)
  [10] posX           (m, EKF)
  [11] posY           (m, EKF)
  [12] heading_deg    (degrees, EKF heading)
  [13] lead_dist      (cm, ultrasonic — 0 if out-of-range)
  [14] ttc            (seconds, time-to-collision)
  [15] acc_status     (int)
  [16] aeb_status     (int, 0=ok, 1=estop, 2=obstacle)
  [17] batVoltage     (V)
  [18] navStatus      (int, 0=idle, 1=goto, 2=turn)
  [19] navProgress    (0.0–1.0)
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
        # All fields initialized to safe defaults.
        self._sensor_data = self._default_sensor_data()
        self._sensor_lock = threading.Lock()
        self._last_sensor_time = 0

        # Send rate limiting — avoid flooding ESP32
        self._last_send_time = 0
        self._min_send_interval = 0.02  # 50 Hz max send rate

        # Parse error tracking
        self._parse_errors = 0
        self._good_parses = 0

    @staticmethod
    def _default_sensor_data():
        """Return a dict with safe default values for all sensor fields."""
        return {
            'v_linear': 0.0,        # m/s
            'base_speed': 0.0,      # 0.0–1.0
            'yaw': 0.0,             # degrees
            'yaw_rate': 0.0,        # deg/s
            'v_angular': 0.0,       # rad/s
            'pwm_left': 0,          # int
            'pwm_right': 0,         # int
            'left_enc': 0,          # encoder ticks (long)
            'right_enc': 0,         # encoder ticks (long)
            'dist_traveled': 0.0,   # meters
            'pos_x': 0.0,           # meters (EKF)
            'pos_y': 0.0,           # meters (EKF)
            'heading_deg': 0.0,     # degrees (EKF)
            'distance_cm': 999.0,   # ultrasonic cm (999 = no reading)
            'ttc': 0.0,             # time-to-collision (s)
            'acc_status': 0,        # int
            'aeb_status': 0,        # int
            'bat_voltage': 0.0,     # volts
            'nav_status': 0,        # 0=idle, 1=goto, 2=turn
            'nav_progress': 0.0,    # 0.0–1.0
        }

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

                    # Prevent buffer from growing unbounded on garbage input
                    if len(buffer) > 2048:
                        buffer = buffer[-512:]
                else:
                    time.sleep(0.005)  # 5ms sleep to avoid busy-waiting

            except serial.SerialException as e:
                log.error(f"Serial read error (connection lost?): {e}")
                time.sleep(1.0)
                # Attempt to reconnect
                self._attempt_reconnect()
            except Exception as e:
                log.error(f"Serial read error: {e}")
                time.sleep(0.1)

    def _attempt_reconnect(self):
        """Try to re-open the serial port after a disconnect."""
        if not self._running:
            return
        try:
            if self._serial:
                self._serial.close()
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                timeout=self.timeout,
                write_timeout=0.1,
            )
            self._serial.reset_input_buffer()
            log.info("Serial reconnected successfully")
        except Exception:
            pass  # Will retry on next loop iteration

    def _parse_sensor_data(self, line):
        """
        Parse the 20-field SEN: packet from ESP32.

        Format: SEN:v_lin,baseSpd,yaw,rate,v_ang,Lpwm,Rpwm,
                    encL,encR,distTrav,posX,posY,heading,
                    leadDist,ttc,accStat,aebStat,batV,navStat,navProg

        Args:
            line: Raw sensor data string starting with "SEN:"
        """
        try:
            parts = line[4:].split(',')
            if len(parts) < 14:
                # Need at least 14 fields for the critical data
                self._parse_errors += 1
                if self._parse_errors % 50 == 1:
                    log.debug(f"Short SEN packet ({len(parts)} fields): {line[:80]}")
                return

            # Parse all fields with safe defaults for missing ones
            data = {
                'v_linear':      float(parts[0]),
                'base_speed':    float(parts[1]),
                'yaw':           float(parts[2]),
                'yaw_rate':      float(parts[3]),
                'v_angular':     float(parts[4]) if len(parts) > 4 else 0.0,
                'pwm_left':      int(parts[5])   if len(parts) > 5 else 0,
                'pwm_right':     int(parts[6])   if len(parts) > 6 else 0,
                'left_enc':      int(parts[7])   if len(parts) > 7 else 0,
                'right_enc':     int(parts[8])   if len(parts) > 8 else 0,
                'dist_traveled': float(parts[9]) if len(parts) > 9 else 0.0,
                'pos_x':         float(parts[10]) if len(parts) > 10 else 0.0,
                'pos_y':         float(parts[11]) if len(parts) > 11 else 0.0,
                'heading_deg':   float(parts[12]) if len(parts) > 12 else 0.0,
                'distance_cm':   float(parts[13]) if len(parts) > 13 else 999.0,
                'ttc':           float(parts[14]) if len(parts) > 14 else 0.0,
                'acc_status':    int(parts[15])   if len(parts) > 15 else 0,
                'aeb_status':    int(parts[16])   if len(parts) > 16 else 0,
                'bat_voltage':   float(parts[17]) if len(parts) > 17 else 0.0,
                'nav_status':    int(parts[18])   if len(parts) > 18 else 0,
                'nav_progress':  float(parts[19]) if len(parts) > 19 else 0.0,
            }

            # ESP32 sends lead_dist=0 when ultrasonic is out-of-range.
            # Normalize to 999 (our "no obstacle" sentinel).
            if data['distance_cm'] <= 0.0:
                data['distance_cm'] = 999.0

            with self._sensor_lock:
                self._sensor_data = data
                self._last_sensor_time = time.monotonic()

            self._good_parses += 1

        except (ValueError, IndexError) as e:
            self._parse_errors += 1
            if self._parse_errors % 50 == 1:
                log.debug(f"Malformed sensor data: {line[:80]} — {e}")

    def get_sensor_data(self):
        """
        Get the latest sensor readings (thread-safe snapshot).

        Returns a cached copy — safe to call multiple times per frame.
        Returns None if data is stale (>500ms old).

        Returns:
            Dict with sensor values, or None if data is stale
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
            return False

        # Rate limiting — don't flood ESP32
        now = time.monotonic()
        if now - self._last_send_time < self._min_send_interval:
            return True  # Skip this send, not an error
        self._last_send_time = now

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
        # Bypass rate limiting for emergency stop
        self._last_send_time = 0
        return self.send_command("CMD:0,0,4")

    @property
    def is_connected(self):
        """Whether the serial connection is active."""
        return self._serial is not None and self._serial.is_open

    @property
    def parse_stats(self):
        """Return parsing statistics for diagnostics."""
        return {
            'good': self._good_parses,
            'errors': self._parse_errors,
        }

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

        log.info(f"Serial disconnected (parsed {self._good_parses} good, "
                 f"{self._parse_errors} errors)")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()
