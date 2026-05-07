"""
TARA ADAS — Decision Manager
Priority-based arbitrator that combines all ADAS module outputs
into a single normalised (x, y) command for the ESP32.

ESP32 motor mapping:
    jd.x  = steering  (-1.0 left … +1.0 right)
    jd.y  = throttle  (0.0 stop … 1.0 full forward)
    targetVL = jd.y + jd.x
    targetVR = jd.y - jd.x

Priority (highest → lowest):
  1. Traffic Light RED/YELLOW  — bypasses smoothing
  2. Pothole Avoidance         — blended release
  3. TSR Speed Cap             — with auto-expiry
  4. ACC Throttle              — vision-only cruise speed
  5. Lane Keeping Assist       — confidence-weighted steering
  6. LDW Warning               — flag only
"""
import time
from utils.logger import get_logger

log = get_logger("Decision")


class Command:
    """
    Normalised command for the ESP32.
      steer_x: -1.0 (hard left) … +1.0 (hard right)  → jd.x
      speed_y:  0.0 (stop)      … +1.0 (full speed)   → jd.y
      flags:   bit-field (0x01=LDW, 0x02=Pothole, 0x08=TSR, 0x10=TL)
    """

    def __init__(self):
        self.steer_x = 0.0
        self.speed_y = 0.0
        self.flags   = 0

    @property
    def steering(self):
        return round(self.steer_x * 100)

    @property
    def speed(self):
        return round(self.speed_y * 255)

    def to_ws(self):
        """Serialise for ESP32 auto_cmd WebSocket message."""
        return {
            "type": "auto_cmd",
            "x":    round(max(-1.0, min(1.0, self.steer_x)), 4),
            "y":    round(max( 0.0, min(1.0, self.speed_y)), 4),
        }

    def to_serial(self):
        x = round(max(-1.0, min(1.0, self.steer_x)), 4)
        y = round(max( 0.0, min(1.0, self.speed_y)), 4)
        return f"CMD:{x},{y},{int(self.flags)}"

    def __repr__(self):
        return (f"Command(steer={self.steer_x:.3f}, "
                f"speed={self.speed_y:.3f}, flags={self.flags:#04x})")


class DecisionManager:
    """
    Combines all ADAS outputs with priority-based arbitration.
    Outputs a normalised Command; no PID runs here.

    Design notes:
    - Emergency stop and red-light BYPASS exponential smoothing.
    - Steering smoothing is light (α=0.10) since the lane detector
      already smooths via polynomial history.
    - Pothole avoidance has a gradual blend-back on release.
    - TSR speed cap auto-expires after 10 s of no re-detection.
    """

    def __init__(self, config):
        self.cfg = config

        self._cruise_speed = config.ACC_DEFAULT_SPEED / 255.0
        self._max_speed    = config.ACC_MAX_SPEED     / 255.0

        self._startup_complete   = False
        self._startup_lane_count = 0
        self._startup_required   = 5

        self._last_steer_x = 0.0
        self._last_speed_y = 0.0

        self._smooth_steer = 0.0
        self._smooth_speed = 0.0
        self._steer_alpha  = 0.10
        self._speed_alpha  = 0.40

        self._lane_lost_frames    = 0
        self._lane_lost_threshold = 8

        self._pothole_active     = False
        self._pothole_steer_x    = 0.0
        self._pothole_start_time = 0.0
        self._pothole_hold_sec   = 0.8
        self._pothole_blend_sec  = 0.4

        self._prev_pothole_detected = False

        self._last_tl_state = "UNKNOWN"

        self._tsr_speed_cap      = None
        self._tsr_last_seen_time = 0.0
        self._tsr_expiry_sec     = 10.0

        self._sign_turn_active     = False
        self._sign_turn_direction  = None
        self._sign_turn_start_time = 0.0
        self._sign_turn_delay_sec  = getattr(config, 'SIGN_TURN_DELAY_SEC', 1.0)
        self._sign_turn_hold_sec   = getattr(config, 'SIGN_TURN_HOLD_SEC', 1.5)

        self._no_perception_frames = 0

        log.info("DecisionManager initialized — car HELD until lanes detected")

    def update(self, lane_result=None, tsr_result=None,
               pothole_result=None, acc_result=None, tl_result=None,
               sign_hint=None):
        """
        Combine ADAS module outputs into a single Command.

        All inputs are optional; missing modules are skipped gracefully.
        Returns a Command with .to_ws() ready to send.

        Priority order: Traffic Light > Pothole > TSR > ACC > LKA > LDW
        """
        cmd            = Command()
        has_perception = False
        now            = time.monotonic()

        # Startup gate — hold until lanes confirmed
        if not self._startup_complete:
            if lane_result is not None and lane_result.lane_detected:
                self._startup_lane_count += 1
                if self._startup_lane_count >= self._startup_required:
                    self._startup_complete = True
                    self._last_speed_y     = self._cruise_speed
                    log.info("Lanes confirmed — car RELEASED, starting autonomous drive")
                else:
                    log.info(f"Startup: lanes detected ({self._startup_lane_count}/{self._startup_required})...")
            else:
                self._startup_lane_count = 0
                log.debug("Startup: waiting for lane detection...")

            if not self._startup_complete:
                cmd.steer_x = 0.0
                cmd.speed_y = 0.0
                return cmd

        # Expire stale TSR speed cap
        if (self._tsr_speed_cap is not None and
                now - self._tsr_last_seen_time > self._tsr_expiry_sec):
            log.info(f"TSR speed cap expired ({self._tsr_expiry_sec:.0f}s without detection)")
            self._tsr_speed_cap = None

        # Priority 7: LDW flag + lane steering
        if lane_result is not None:
            if lane_result.departure_warning:
                cmd.flags |= 0x01

            if lane_result.lane_detected:
                has_perception         = True
                self._lane_lost_frames = 0
                steer_gain             = 2.0
                confidence_weight      = getattr(lane_result, 'confidence', 1.0)
                self._last_steer_x     = (lane_result.steering_correction
                                          * steer_gain
                                          * min(1.0, confidence_weight + 0.3))
            else:
                self._lane_lost_frames += 1

        # Priority 5: ACC throttle
        if acc_result is not None:
            has_perception = True
            spd = acc_result.speed_norm
            if self._tsr_speed_cap is not None:
                spd = min(spd, self._tsr_speed_cap)
            self._last_speed_y = spd

        # Priority 4: TSR speed cap
        if tsr_result is not None and tsr_result.sign_detected:
            has_perception           = True
            cmd.flags               |= 0x08
            self._tsr_last_seen_time = now

            if tsr_result.speed_limit is not None:
                self._tsr_speed_cap = tsr_result.speed_limit

            if tsr_result.is_stop_sign:
                self._tsr_speed_cap = 0.0
                self._last_speed_y  = 0.0

        raw_steer = self._last_steer_x
        raw_speed = self._last_speed_y

        # OpenCV sign — landmark navigation maneuver
        if sign_hint in ["LEFT", "RIGHT"]:
            if not self._sign_turn_active or self._sign_turn_direction != sign_hint:
                log.info(f"SIGN DETECTED: {sign_hint} — entering {self._sign_turn_delay_sec}s delay")
            self._sign_turn_active     = True
            self._sign_turn_direction  = sign_hint
            self._sign_turn_start_time = now

        if self._sign_turn_active:
            elapsed        = now - self._sign_turn_start_time
            total_duration = self._sign_turn_delay_sec + self._sign_turn_hold_sec

            if elapsed < self._sign_turn_delay_sec:
                raw_speed *= 0.8
            elif elapsed < total_duration:
                if elapsed - self._sign_turn_delay_sec < 0.1:
                    log.info(f"SIGN MANEUVER: executing SHARP {self._sign_turn_direction} turn")
                bias                = -0.95 if self._sign_turn_direction == "LEFT" else 0.95
                self._last_steer_x  = bias
                raw_speed          *= 0.4
                raw_steer           = bias
            else:
                self._sign_turn_active = False
                log.info(f"SIGN MANEUVER: completed {self._sign_turn_direction}")

        # Dynamic speed reduction on tight steering
        steer_abs = abs(raw_steer)
        if steer_abs > 0.3:
            speed_factor = 1.0 - (steer_abs - 0.3) * 0.8
            raw_speed   *= max(0.4, speed_factor)

        # Lane-loss fail-safe
        if self._lane_lost_frames > 0:
            raw_speed *= 0.7

            if self._lane_lost_frames < self._lane_lost_threshold:
                raw_steer = self._last_steer_x
            else:
                raw_steer = 0.0
                if self._lane_lost_frames == 10:
                    log.error("LANE LOST for 10 frames — EMERGENCY STOP")

        # Exponential smoothing (base command)
        self._smooth_steer = (self._steer_alpha * self._smooth_steer +
                              (1 - self._steer_alpha) * raw_steer)
        self._smooth_speed = (self._speed_alpha * self._smooth_speed +
                              (1 - self._speed_alpha) * raw_speed)

        cmd.steer_x = self._smooth_steer
        cmd.speed_y = self._smooth_speed

        # Priority 3: Pothole avoidance (overrides steering post-smoothing)
        pothole_confirmed = False
        if pothole_result is not None and pothole_result.pothole_detected:
            has_perception = True
            if self._prev_pothole_detected:
                pothole_confirmed = True
            self._prev_pothole_detected = True
        else:
            self._prev_pothole_detected = False

        if pothole_confirmed and not self._pothole_active:
            self._pothole_steer_x    = pothole_result.avoidance_steer / 100.0
            self._pothole_start_time = now
            self._pothole_active     = True
            log.info(f"Pothole avoidance started: steer={self._pothole_steer_x:.2f}")

        if self._pothole_active:
            elapsed        = now - self._pothole_start_time
            total_duration = self._pothole_hold_sec + self._pothole_blend_sec

            if elapsed < self._pothole_hold_sec:
                cmd.steer_x  = self._pothole_steer_x
                cmd.speed_y *= 0.6
                cmd.flags   |= 0x02
            elif elapsed < total_duration:
                blend_progress = (elapsed - self._pothole_hold_sec) / self._pothole_blend_sec
                cmd.steer_x   = ((1.0 - blend_progress) * self._pothole_steer_x
                                 + blend_progress * self._smooth_steer)
                cmd.speed_y  *= (0.6 + 0.4 * blend_progress)
                cmd.flags    |= 0x02
            else:
                self._pothole_active = False
                log.info(f"Pothole avoidance done ({elapsed:.1f}s)")

        # Priority 2: Traffic light (bypasses smoothing)
        if tl_result is not None and tl_result.detected:
            has_perception      = True
            self._last_tl_state = tl_result.state

            if tl_result.state == "RED":
                cmd.speed_y         = 0.0
                cmd.flags          |= 0x10
                self._smooth_speed  = 0.0
            elif tl_result.state == "YELLOW":
                cmd.speed_y        = cmd.speed_y * 0.3
                cmd.flags         |= 0x10
                self._smooth_speed = cmd.speed_y

        # Perception health fallback
        if has_perception:
            self._no_perception_frames = 0
        else:
            self._no_perception_frames += 1
            if self._no_perception_frames >= 5:
                cmd.speed_y = cmd.speed_y * 0.85
                cmd.steer_x = 0.0
                if self._no_perception_frames == 5:
                    log.warning("No perception for 5 frames — gradual safe slowdown")

        # Hard clamp
        cmd.steer_x = max(-1.0, min(1.0, cmd.steer_x))
        cmd.speed_y = max(0.0,  min(1.0, cmd.speed_y))

        log.debug(f"Decision: {cmd}")
        return cmd
