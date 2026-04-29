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
  1. Emergency Stop (ACC ultrasonic)         — bypasses smoothing
  2. Traffic Light RED/YELLOW                — bypasses smoothing
  3. Pothole Avoidance (override steering)   — blended release
  4. TSR Speed Cap                           — with auto-expiry
  5. ACC Throttle                            — proportional to distance
  6. Lane Keeping Assist (steering)          — with confidence weighting
  7. LDW Warning (flag only)
"""
import time
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
        """Serialize to CSV for ESP32 serialParserTask."""
        x = round(max(-1.0, min(1.0, self.steer_x)), 4)
        y = round(max(0.0,  min(1.0, self.speed_y)), 4)
        flags = int(self.flags)
        return f"CMD:{x},{y},{flags}"

    def __repr__(self):
        return (f"Command(steer={self.steer_x:.3f}, "
                f"speed={self.speed_y:.3f}, flags={self.flags:#04x})")


class DecisionManager:
    """
    Combines all ADAS outputs with priority-based arbitration.
    Outputs a normalized Command; no PID runs here.

    Key design decisions:
    - Emergency stop and red-light BYPASS exponential smoothing
    - Steering smoothing is light (α=0.35) since lane detector already smooths
    - Pothole avoidance has a gradual blend-back on release
    - TSR speed cap auto-expires after 10 seconds of no re-detection
    """

    def __init__(self, config):
        self.cfg = config

        # Default cruise speed (normalized)
        self._cruise_speed   = config.ACC_DEFAULT_SPEED / 255.0
        self._max_speed      = config.ACC_MAX_SPEED     / 255.0

        # ── Startup gate ──────────────────────────────────────────────
        # Car stays stationary until lanes are confirmed for N frames.
        # This prevents launching at cruise speed before the camera
        # has acquired the track.
        self._startup_complete      = False
        self._startup_lane_count    = 0
        self._startup_required      = 3   # need 3 consecutive lane detections

        # Last known values (persisted between scheduled frames)
        self._last_steer_x   = 0.0
        self._last_speed_y   = 0.0   # START at zero (not cruise!)

        # Smoothing state (exponential moving average)
        # α close to 1 = heavier smoothing (slower response)
        # Reduced from 0.55 — lane detector already smooths via polynomial history
        self._smooth_steer   = 0.0
        self._smooth_speed   = 0.0   # START at zero (ramp up via EMA once lanes found)
        self._steer_alpha    = 0.35   # lighter smoothing for faster response
        self._speed_alpha    = 0.60

        # Lane-loss fail-safe
        self._lane_lost_frames    = 0
        self._lane_lost_threshold = 3

        # Pothole avoidance state
        self._pothole_active      = False
        self._pothole_steer_x     = 0.0
        self._pothole_start_time  = 0.0
        self._pothole_hold_sec    = 0.8    # hold dodge for 0.8 seconds (time-based, not encoder)
        self._pothole_blend_sec   = 0.4    # gradual release over 0.4 seconds

        # Temporal validation — require 2 consecutive detections
        self._prev_pothole_detected = False

        # Traffic light
        self._last_tl_state = "UNKNOWN"

        # TSR-derived speed cap with auto-expiry
        self._tsr_speed_cap       = None
        self._tsr_last_seen_time  = 0.0
        self._tsr_expiry_sec      = 10.0   # clear speed cap after 10s of no re-detection

        # Perception health
        self._no_perception_frames = 0

        log.info("DecisionManager initialized — car HELD until lanes detected")

    # ─────────────────────────────────────────────────────────────────────────

    def update(self, lane_result=None, tsr_result=None,
               pothole_result=None, acc_result=None, tl_result=None,
               sensor_data=None):
        """
        Combine ADAS module outputs into a single Command.

        All inputs are optional — missing modules are skipped gracefully.
        Returns a Command ready to call .to_serial() on.

        Priority processing order (highest first):
        Emergency Stop > Traffic Light > Pothole > TSR > ACC > LKA > LDW
        """
        cmd = Command()
        has_perception = False
        now = time.monotonic()

        # ═══════════════════════════════════════════════════════════════════
        # STARTUP GATE: Hold car stationary until lanes are confirmed.
        # The car must see lanes for 3 consecutive frames before it moves.
        # This prevents launching off the track on boot.
        # ═══════════════════════════════════════════════════════════════════
        if not self._startup_complete:
            if lane_result is not None and lane_result.lane_detected:
                self._startup_lane_count += 1
                if self._startup_lane_count >= self._startup_required:
                    self._startup_complete = True
                    self._last_speed_y = self._cruise_speed
                    log.info("✓ Lanes confirmed — car RELEASED, starting autonomous drive")
                else:
                    log.info(f"Startup: lanes detected ({self._startup_lane_count}/{self._startup_required})...")
            else:
                self._startup_lane_count = 0  # reset on miss
                log.debug("Startup: waiting for lane detection...")

            if not self._startup_complete:
                # Still waiting — send zero speed, zero steer
                cmd.steer_x = 0.0
                cmd.speed_y = 0.0
                return cmd

        # ── Expire stale TSR speed cap ────────────────────────────────────
        if (self._tsr_speed_cap is not None and
                now - self._tsr_last_seen_time > self._tsr_expiry_sec):
            log.info(f"TSR speed cap expired (no sign for {self._tsr_expiry_sec:.0f}s)")
            self._tsr_speed_cap = None

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 1: Gather inputs from all modules (low → high priority)
        # ═══════════════════════════════════════════════════════════════════

        # ── Priority 7: LDW flag ──────────────────────────────────────────
        if lane_result is not None:
            if lane_result.departure_warning:
                cmd.flags |= 0x01

            if lane_result.lane_detected:
                has_perception = True
                self._lane_lost_frames = 0
                # steering_correction is already normalized: -1.0 … 1.0
                # Weight by detection confidence for smoother single-lane behavior
                confidence_weight = getattr(lane_result, 'confidence', 1.0)
                self._last_steer_x = lane_result.steering_correction * min(1.0, confidence_weight + 0.3)
            else:
                self._lane_lost_frames += 1

        # ── Priority 5: ACC throttle ──────────────────────────────────────
        if acc_result is not None:
            has_perception = True
            # ACCResult has speed_norm (0.0–1.0) — use it directly
            spd = acc_result.speed_norm
            if self._tsr_speed_cap is not None:
                spd = min(spd, self._tsr_speed_cap)
            self._last_speed_y = spd

        # ── Priority 4: TSR speed cap ─────────────────────────────────────
        if tsr_result is not None and tsr_result.sign_detected:
            has_perception = True
            cmd.flags |= 0x08
            self._tsr_last_seen_time = now

            if tsr_result.speed_limit is not None:
                # speed_limit is already normalized (0.0–1.0) from config.TSR_SPEED_LIMITS
                self._tsr_speed_cap = tsr_result.speed_limit

            if tsr_result.is_stop_sign:
                self._tsr_speed_cap = 0.0
                self._last_speed_y = 0.0

        # ── Assemble base command from LKA + ACC ──────────────────────────
        raw_steer = self._last_steer_x
        raw_speed = self._last_speed_y

        # ── Lane-loss fail-safe ───────────────────────────────────────────
        if self._lane_lost_frames >= self._lane_lost_threshold:
            raw_speed = min(raw_speed, self._cruise_speed * 0.5)
            raw_steer = raw_steer * 0.3
            if self._lane_lost_frames == self._lane_lost_threshold:
                log.warning("Lane lost ≥3 frames — 50% speed, dampened steering")

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 2: Apply exponential smoothing to base command
        # This runs BEFORE priority overrides so that E-STOP and TL
        # can bypass it for instant response.
        # ═══════════════════════════════════════════════════════════════════

        self._smooth_steer = (
            self._steer_alpha * self._smooth_steer +
            (1 - self._steer_alpha) * raw_steer
        )
        self._smooth_speed = (
            self._speed_alpha * self._smooth_speed +
            (1 - self._speed_alpha) * raw_speed
        )

        cmd.steer_x = self._smooth_steer
        cmd.speed_y = self._smooth_speed

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 3: Priority overrides (AFTER smoothing — these bypass it)
        # ═══════════════════════════════════════════════════════════════════

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
            # Start avoidance maneuver
            self._pothole_steer_x = pothole_result.avoidance_steer / 100.0
            self._pothole_start_time = now
            self._pothole_active = True
            log.info(f"Pothole avoidance started: steer={self._pothole_steer_x:.2f}")

        if self._pothole_active:
            elapsed = now - self._pothole_start_time
            total_duration = self._pothole_hold_sec + self._pothole_blend_sec

            if elapsed < self._pothole_hold_sec:
                # Full avoidance steering
                cmd.steer_x = self._pothole_steer_x
                cmd.speed_y = cmd.speed_y * 0.6
                cmd.flags |= 0x02
            elif elapsed < total_duration:
                # Gradual blend-back to LKA steering
                blend_progress = (elapsed - self._pothole_hold_sec) / self._pothole_blend_sec
                cmd.steer_x = (1.0 - blend_progress) * self._pothole_steer_x + \
                              blend_progress * self._smooth_steer
                cmd.speed_y = cmd.speed_y * (0.6 + 0.4 * blend_progress)
                cmd.flags |= 0x02
            else:
                # Avoidance complete
                self._pothole_active = False
                log.info(f"Pothole avoidance done ({elapsed:.1f}s)")

        # ── Priority 2: Traffic Light ─────────────────────────────────────
        # Overrides speed directly — bypasses smoothing for safety.
        if tl_result is not None and tl_result.detected:
            has_perception = True
            self._last_tl_state = tl_result.state

            if tl_result.state == "RED":
                cmd.speed_y = 0.0
                cmd.flags |= 0x10
                # Also reset the speed smoother so we don't ramp back up slowly
                self._smooth_speed = 0.0
            elif tl_result.state == "YELLOW":
                cmd.speed_y = cmd.speed_y * 0.3
                cmd.flags |= 0x10
                self._smooth_speed = cmd.speed_y

        # ── Priority 1: Emergency Stop (HIGHEST) ─────────────────────────
        # Completely overrides everything — no smoothing, instant effect.
        if acc_result is not None and acc_result.emergency_stop:
            cmd.steer_x = 0.0
            cmd.speed_y = 0.0
            cmd.flags |= 0x04
            # Reset smoothers so we don't ramp back up after obstacle clears
            self._smooth_steer = 0.0
            self._smooth_speed = 0.0

        # ── Perception health fallback ────────────────────────────────────
        if has_perception:
            self._no_perception_frames = 0
        else:
            self._no_perception_frames += 1
            if self._no_perception_frames >= 5:
                # Gradual slowdown if nothing is working
                cmd.speed_y = cmd.speed_y * 0.85
                cmd.steer_x = 0.0
                if self._no_perception_frames == 5:
                    log.warning("No perception for 5 frames — gradual safe slowdown")

        # ── Hard clamp ───────────────────────────────────────────────────
        cmd.steer_x = max(-1.0, min(1.0, cmd.steer_x))
        cmd.speed_y = max(0.0,  min(1.0, cmd.speed_y))

        log.debug(f"Decision: {cmd}")
        return cmd
