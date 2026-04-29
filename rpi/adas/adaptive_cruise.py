"""
TARA ADAS — Adaptive Cruise Control (ACC)
Vision-only speed policy — no ultrasonic sensor.

Design:
  ACC outputs a normalized cruise speed setpoint (0.0–1.0) that is then
  subject to priority overrides in DecisionManager:
    TSR speed cap   → reduces setpoint if a sign is detected
    Pothole         → DecisionManager reduces speed during avoidance
    Traffic Light   → DecisionManager overrides to 0 on RED
    Lane confidence → DecisionManager startup gate holds car until lanes confirmed

  No PID, no distance sensor, no emergency stop here.
  The ESP32's motor PID handles closed-loop velocity tracking.
"""
from utils.logger import get_logger

log = get_logger("ACC")


class ACCResult:
    """Container for ACC outputs."""

    def __init__(self):
        self.speed_norm = 0.0      # Cruise speed setpoint 0.0–1.0 → ESP32 jd.y
        self.mode       = "CRUISE" # Always CRUISE (no distance zones)


class AdaptiveCruiseControl:
    """
    Cruise speed policy module.
    Returns a normalized speed setpoint (0.0–1.0) optionally capped by TSR.
    No ultrasonic sensor, no distance-based logic.
    """

    def __init__(self, config):
        self.cfg = config

        # Normalize cruise and max speed from config (stored as PWM 0-255)
        self.cruise_speed_norm = config.ACC_DEFAULT_SPEED / 255.0
        self.max_speed_norm    = config.ACC_MAX_SPEED     / 255.0

        # TSR-derived speed cap (0.0–1.0), set externally by main.py
        self._tsr_speed_limit_norm = None

        log.info(f"ACC initialized: cruise={self.cruise_speed_norm:.2f} "
                 f"max={self.max_speed_norm:.2f} (vision-only, no ultrasonic)")

    # ── Public API ────────────────────────────────────────────────────────────

    def set_speed_limit(self, speed_limit_norm):
        """
        Apply a TSR-derived speed cap.

        Args:
            speed_limit_norm: Normalized speed 0.0–1.0 from config.TSR_SPEED_LIMITS.
                              Already normalized — do NOT divide by 255 here.
        """
        if speed_limit_norm is not None:
            self._tsr_speed_limit_norm = float(speed_limit_norm)
            log.info(f"ACC: TSR speed limit -> {self._tsr_speed_limit_norm:.2f}")
        else:
            self._tsr_speed_limit_norm = None

    def update(self) -> ACCResult:
        """
        Compute normalized cruise speed setpoint.
        No sensor data needed — purely speed policy.

        Returns:
            ACCResult with speed_norm (0.0–1.0)
        """
        result = ACCResult()

        cruise = self.cruise_speed_norm

        # Apply TSR cap if active
        if self._tsr_speed_limit_norm is not None:
            cruise = min(cruise, self._tsr_speed_limit_norm)

        result.speed_norm = max(0.0, min(self.max_speed_norm, cruise))
        return result
