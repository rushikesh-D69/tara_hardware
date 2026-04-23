"""
TARA ADAS — Decision Manager
Priority-based arbitrator combining all ADAS module outputs
into a single normalized (x, y) command for the ESP32.

ESP32 motorTask maps:
    jd.x  = steering  (-1.0 left … +1.0 right)
    jd.y  = throttle  (0.0 stop … 1.0 full forward)
    targetVL = jd.y + jd.x
    targetVR = jd.y - jd.x   (open-loop → applyOpenLoop → PWM)

All PID loops removed from this side — the ESP32 handles motor control.
The RPi only decides WHAT direction/speed to aim for.

Priority (highest → lowest):
  1. Emergency Stop (ACC ultrasonic)
  2. Pothole Avoidance (override steering + reduce throttle)
  3. Traffic Light (red/yellow → stop or slow)
  4. TSR Speed Cap
  5. ACC Throttle
  6. Lane Keeping Assist (steering)
  7. LDW Warning (flag only)
"""
import json
from utils.logger import get_logger

log = get_logger("Decision")


class Command:
    """
    Normalized command for the ESP32.
    steer_x:  -1.0 (hard left) … +1.0 (hard right)  → jd.x
    speed_y:   0.0 (stop)      … +1.0 (full speed)   → jd.y
    flags:    bit-field (see below)
    """

    def __init__(self):
        self.steer_x = 0.0   # -1.0 … 1.0
        self.speed_y = 0.0   # 0.0 … 1.0
        self.flags   = 0
        # Flag bits:
        #   0x01 = LDW warning
        #   0x02 = Pothole detected
        #   0x04 = Emergency stop
        #   0x08 = TSR sign detected
        #   0x10 = Traffic light RED/YELLOW

    # compat shims so existing log/print code still compiles
    @property
    def steering(self):
        return round(self.steer_x * 100)   # -100 … 100 int view

    @property
    def speed(self):
        return round(self.speed_y * 255)   # 0 … 255 int view

    def to_serial(self):
        """Serialize to JSON for ESP32 serialParserTask."""
        return json.dumps({
            "type":  "drive",
            "x":     round(max(-1.0, min(1.0, self.steer_x)), 4),
            "y":     round(max(0.0,  min(1.0, self.speed_y)), 4),
            "flags": int(self.flags),
        })

    def __repr__(self):
        return (f"Command(steer={self.steer_x:.3f}, "
                f"speed={self.speed_y:.3f}, flags={self.flags:#04x})")


class DecisionManager:
    """
    Combines all ADAS outputs with priority-based arbitration.
    Outputs a normalized Command; no PID runs here.
    """

    def __init__(self, config):
        self.cfg = config

        # Default cruise speed (normalized)
        self._cruise_speed   = config.ACC_DEFAULT_SPEED / 255.0
        self._max_speed      = config.ACC_MAX_SPEED     / 255.0

        # Last known values (persisted between scheduled frames)
        self._last_steer_x   = 0.0
        self._last_speed_y   = self._cruise_speed

        # Smoothing state (exponential moving average)
        # α close to 1 = heavier smoothing (slower response)
        self._smooth_steer   = 0.0
        self._smooth_speed   = self._cruise_speed
        self._steer_alpha    = 0.55   # less smoothing than before — no PID wind-up risk
        self._speed_alpha    = 0.70

        # Lane-loss fail-safe
        self._lane_lost_frames    = 0
        self._lane_lost_threshold = 3

        # Pothole avoidance hold (encoder-distance based)
        self._pothole_active      = False
        self._pothole_steer_x     = 0.0
        self._start_dist_cm       = 0.0
        self._total_dist_cm       = 0.0
        self._last_enc            = None
        self._pothole_hold_cm     = 40.0  # hold dodge for 40 cm

        # Temporal validation — require 2 consecutive detections
        self._prev_tsr_detected     = False
        self._prev_pothole_detected = False

        # Traffic light
        self._last_tl_state = "UNKNOWN"

        # TSR-derived speed cap
        self._tsr_speed_cap = None

        # Perception health
        self._no_perception_frames = 0

        log.info("DecisionManager initialized (normalized setpoints, no PID)")

    # ─────────────────────────────────────────────────────────────────────────

    def update(self, lane_result=None, tsr_result=None,
               pothole_result=None, acc_result=None, tl_result=None,
               sensor_data=None):
        """
        Combine ADAS module outputs into a single Command.

        All inputs are optional — missing modules are skipped gracefully.
        Returns a Command ready to call .to_serial() on.
        """
        cmd = Command()
        has_perception = False

        # ── Odometry from encoder telemetry ───────────────────────────────
        if sensor_data is not None:
            enc = (sensor_data.get('left_enc', 0) + sensor_data.get('right_enc', 0)) / 2.0
            if self._last_enc is not None:
                d_ticks = enc - self._last_enc
                d_cm = (d_ticks / self.cfg.ENCODER_TICKS_PER_REV) * self.cfg.WHEEL_CIRCUMFERENCE_CM
                self._total_dist_cm += d_cm
            self._last_enc = enc

        # ── Priority 7: LDW flag ──────────────────────────────────────────
        if lane_result is not None:
            if lane_result.departure_warning:
                cmd.flags |= 0x01

            if lane_result.lane_detected:
                has_perception = True
                self._lane_lost_frames = 0
                # steering_correction is already normalized: -1.0 … 1.0
                self._last_steer_x = lane_result.steering_correction
            else:
                self._lane_lost_frames += 1

        # ── Priority 6: ACC throttle ──────────────────────────────────────
        if acc_result is not None:
            has_perception = True
            # ACCResult now has speed_norm (0.0–1.0) — use it directly
            spd = acc_result.speed_norm
            if self._tsr_speed_cap is not None:
                spd = min(spd, self._tsr_speed_cap)
            self._last_speed_y = spd

        # ── Priority 5: TSR speed cap ─────────────────────────────────────
        if tsr_result is not None and tsr_result.sign_detected:
            has_perception = True
            if self._prev_tsr_detected:
                cmd.flags |= 0x08
                if tsr_result.speed_limit is not None:
                    self._tsr_speed_cap = tsr_result.speed_limit / 255.0
                if tsr_result.is_stop_sign:
                    self._tsr_speed_cap = 0.0
                    self._last_speed_y  = 0.0
            self._prev_tsr_detected = True
        else:
            self._prev_tsr_detected = False

        # ── Priority 4: Traffic Light ─────────────────────────────────────
        if tl_result is not None and tl_result.detected:
            has_perception = True
            self._last_tl_state = tl_result.state
            if tl_result.state == "RED":
                self._last_speed_y = 0.0
                cmd.flags |= 0x10
            elif tl_result.state == "YELLOW":
                self._last_speed_y = self._last_speed_y * 0.4
                cmd.flags |= 0x10

        # ── Assemble raw command ──────────────────────────────────────────
        raw_steer = self._last_steer_x
        raw_speed = self._last_speed_y

        # ── Lane-loss fail-safe ───────────────────────────────────────────
        if self._lane_lost_frames >= self._lane_lost_threshold:
            raw_speed = min(raw_speed, self._cruise_speed * 0.5)
            raw_steer = raw_steer * 0.3
            if self._lane_lost_frames == self._lane_lost_threshold:
                log.warning("Lane lost ≥3 frames — 50% speed, dampened steering")

        cmd.steer_x = raw_steer
        cmd.speed_y = raw_speed

        # ── Priority 3: Pothole avoidance (override steering) ─────────────
        pothole_confirmed = False
        if pothole_result is not None and pothole_result.pothole_detected:
            has_perception = True
            if self._prev_pothole_detected:
                pothole_confirmed = True
            self._prev_pothole_detected = True
        else:
            self._prev_pothole_detected = False

        if pothole_confirmed and not self._pothole_active:
            # Convert int avoidance_steer (e.g. 60) to normalized
            self._pothole_steer_x   = pothole_result.avoidance_steer / 100.0
            self._start_dist_cm     = self._total_dist_cm
            self._pothole_active    = True
            cmd.flags              |= 0x02

        if self._pothole_active:
            traveled = self._total_dist_cm - self._start_dist_cm
            if traveled < self._pothole_hold_cm:
                cmd.steer_x = self._pothole_steer_x
                cmd.speed_y = cmd.speed_y * 0.6
                cmd.flags  |= 0x02
            else:
                self._pothole_active = False
                log.info(f"Pothole avoidance done (traveled {traveled:.1f}cm)")

        # ── Priority 1: Emergency Stop ────────────────────────────────────
        if acc_result is not None and acc_result.emergency_stop:
            cmd.steer_x = 0.0
            cmd.speed_y = 0.0
            cmd.flags  |= 0x04

        # ── Perception health fallback ────────────────────────────────────
        if has_perception:
            self._no_perception_frames = 0
        else:
            self._no_perception_frames += 1
            if self._no_perception_frames >= 3:
                cmd.speed_y   = cmd.speed_y * 0.85
                cmd.steer_x   = 0.0
                if self._no_perception_frames == 3:
                    log.warning("No perception — gradual safe slowdown")

        # ── Exponential smoothing ─────────────────────────────────────────
        self._smooth_steer = (
            self._steer_alpha * self._smooth_steer +
            (1 - self._steer_alpha) * cmd.steer_x
        )
        self._smooth_speed = (
            self._speed_alpha * self._smooth_speed +
            (1 - self._speed_alpha) * cmd.speed_y
        )
        cmd.steer_x = self._smooth_steer
        cmd.speed_y = self._smooth_speed

        # ── Hard clamp ───────────────────────────────────────────────────
        cmd.steer_x = max(-1.0, min(1.0, cmd.steer_x))
        cmd.speed_y = max(0.0,  min(1.0, cmd.speed_y))

        log.debug(f"Decision: {cmd}")
        return cmd
