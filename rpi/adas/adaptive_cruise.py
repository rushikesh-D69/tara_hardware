"""
TARA ADAS — Adaptive Cruise Control (ACC)
Sensor fusion: Ultrasonic distance + Encoder speed from ESP32 $TARA telemetry.

Design (post-refactor):
  The ACC sends NORMALIZED speed setpoints (0.0–1.0) directly to the ESP32.
  The ESP32's open-loop motor driver (applyOpenLoop in Tasks.cpp) maps that
  to PWM with its own battery-voltage scaling and floor clamping.
  No PID is run here — that would duplicate effort and fight the ESP32.
"""
import time
from collections import deque
from utils.logger import get_logger

log = get_logger("ACC")


class ACCResult:
    """Container for ACC outputs."""

    def __init__(self):
        self.speed_norm   = 0.0      # Normalized speed 0.0–1.0 → directly to ESP32 jd.y
        self.emergency_stop = False
        self.distance_cm  = -1.0     # Filtered front distance (cm)
        self.v_linear_ms  = 0.0      # Actual robot speed (m/s) from ESP32 telemetry
        self.mode         = "CRUISE" # CRUISE | FOLLOW | DECEL | E_STOP


class AdaptiveCruiseControl:
    """
    ACC module — computes a normalized speed setpoint from distance telemetry.
    Sends 0.0–1.0 to the decison manager, which forwards it to ESP32 as jd.y.
    """

    def __init__(self, config):
        self.cfg = config

        # Distance thresholds (cm)
        self.e_stop_dist    = config.ACC_EMERGENCY_STOP_DIST   # 10 cm
        self.min_follow_dist = config.ACC_MIN_FOLLOW_DIST      # 25 cm
        self.cruise_dist    = config.ACC_CRUISE_DIST           # 50 cm

        # Normalized speed targets (0.0–1.0)
        # ACC_DEFAULT_SPEED and ACC_MAX_SPEED are still in config as PWM (0-255)
        # so divide by 255 here.
        self.cruise_speed_norm = config.ACC_DEFAULT_SPEED / 255.0
        self.max_speed_norm    = config.ACC_MAX_SPEED    / 255.0

        # TSR can override the cruise speed cap
        self._tsr_speed_limit_norm = None

        # Ultrasonic noise filter — average over last 5 readings
        self._dist_buffer = deque(maxlen=5)

        # Speed from $TARA telemetry (v_linear in m/s, populated by serial bridge)
        self._v_linear = 0.0

        log.info(f"ACC initialized: e_stop={self.e_stop_dist}cm, "
                 f"follow={self.min_follow_dist}cm, cruise={self.cruise_dist}cm")

    # ── Public API ────────────────────────────────────────────────────────────

    def set_speed_limit(self, speed_limit_pwm):
        """
        Apply a TSR-derived speed cap.

        Args:
            speed_limit_pwm: PWM value from config.TSR_SPEED_LIMITS, or None to clear
        """
        if speed_limit_pwm is not None:
            self._tsr_speed_limit_norm = speed_limit_pwm / 255.0
            log.info(f"ACC: TSR speed limit → {self._tsr_speed_limit_norm:.2f} (norm)")
        else:
            self._tsr_speed_limit_norm = None

    def update(self, sensor_data):
        """
        Compute normalized speed setpoint from sensor data.

        Args:
            sensor_data: Dict from serial_bridge.get_sensor_data()
                Mandatory key: 'distance_cm'
                Optional key:  'v_linear'  (m/s, from $TARA[1])

        Returns:
            ACCResult with speed_norm (0.0–1.0) and mode string
        """
        result = ACCResult()

        # ── Distance filtering ─────────────────────────────────────────────
        raw_dist = sensor_data.get('distance_cm', 999.0)
        if 2.0 <= raw_dist <= 400.0:
            self._dist_buffer.append(raw_dist)

        distance = (
            sum(self._dist_buffer) / len(self._dist_buffer)
            if self._dist_buffer else raw_dist
        )
        result.distance_cm = distance

        # ── Use ESP32 v_linear if available ───────────────────────────────
        result.v_linear_ms = sensor_data.get('v_linear', 0.0)

        # ── Effective cruise speed (apply TSR cap if set) ─────────────────
        cruise = self.cruise_speed_norm
        if self._tsr_speed_limit_norm is not None:
            cruise = min(cruise, self._tsr_speed_limit_norm)

        # ── Distance-based speed selection (no PID — just setpoints) ─────
        if distance <= self.e_stop_dist:
            # Hard stop — obstacle too close
            result.mode          = "E_STOP"
            result.emergency_stop = True
            result.speed_norm    = 0.0
            log.warning(f"ACC E-STOP! dist={distance:.1f}cm")

        elif distance <= self.min_follow_dist:
            # Decelerate proportionally to proximity
            result.mode = "DECEL"
            ratio = (distance - self.e_stop_dist) / (
                self.min_follow_dist - self.e_stop_dist)   # 0→1
            result.speed_norm = cruise * ratio * 0.5       # max 50% cruise

        elif distance <= self.cruise_dist:
            # Follow mode — hold 70% cruise speed
            result.mode       = "FOLLOW"
            result.speed_norm = cruise * 0.70

        else:
            # Clear road — full cruise
            result.mode       = "CRUISE"
            result.speed_norm = cruise

        # Hard clamp to valid range
        result.speed_norm = max(0.0, min(self.max_speed_norm, result.speed_norm))
        return result
