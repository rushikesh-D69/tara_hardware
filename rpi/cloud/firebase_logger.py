"""
TARA ADAS — Firebase Cloud Logger
Asynchronous, non-blocking cloud data upload that runs entirely
in a background thread. NEVER delays the real-time ML pipeline.

Uploads:
  1. Telemetry    (every N seconds) → Firebase Realtime DB
  2. Event frames (on detection)    → Firebase Storage (JPEG)
  3. Session log  (on stop)         → Firebase Realtime DB

Free tier limits (more than enough for prototype):
  - Realtime DB: 1 GB stored, 10 GB/month download
  - Storage:     5 GB stored, 1 GB/day upload
"""
import os
import json
import time
import threading
import queue
import cv2
import numpy as np
from datetime import datetime
from utils.logger import get_logger

log = get_logger("Cloud")

# Firebase SDK is optional — if not installed, cloud logging is silently disabled
_firebase_available = False
try:
    import firebase_admin
    from firebase_admin import credentials, db, storage
    _firebase_available = True
except ImportError:
    log.warning("firebase-admin not installed. Cloud logging disabled. "
                "Install with: pip install firebase-admin")


class FirebaseLogger:
    """
    Asynchronous Firebase logger for TARA ADAS.
    All uploads happen in a background thread via a queue, so the
    real-time inference pipeline is never blocked.
    """

    def __init__(self, config):
        """
        Args:
            config: Config module with CLOUD_* parameters
        """
        self.cfg = config
        self._enabled = False
        self._running = False
        self._thread = None

        # Queue for async uploads — bounded to prevent memory buildup
        # if WiFi is down. Old items are dropped when full.
        self._queue = queue.Queue(maxsize=200)

        # Telemetry throttle
        self._last_telemetry_time = 0
        self._telemetry_interval = getattr(config, 'CLOUD_TELEMETRY_INTERVAL', 2.0)

        # Session tracking
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_start = time.monotonic()
        self._event_count = 0
        self._frame_count = 0

        # Stats per feature
        self._detection_counts = {
            "lane_departures": 0,
            "signs_detected": 0,
            "potholes_detected": 0,
            "emergency_stops": 0,
        }

    def connect(self):
        """
        Initialize Firebase connection.
        Returns True if connected, False if disabled/failed.
        """
        if not _firebase_available:
            log.info("Firebase SDK not available — cloud logging off")
            return False

        cred_path = getattr(self.cfg, 'FIREBASE_CREDENTIALS_PATH', None)
        db_url = getattr(self.cfg, 'FIREBASE_DB_URL', None)
        bucket = getattr(self.cfg, 'FIREBASE_STORAGE_BUCKET', None)

        if not cred_path or not os.path.exists(cred_path):
            log.warning(f"Firebase credentials not found at: {cred_path}")
            log.info("Cloud logging disabled. See docs/cloud_setup.md for setup.")
            return False

        if not db_url:
            log.warning("FIREBASE_DB_URL not configured")
            return False

        try:
            # Initialize Firebase app (only once)
            if not firebase_admin._apps:
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred, {
                    'databaseURL': db_url,
                    'storageBucket': bucket,
                })

            # Test connection
            ref = db.reference(f'/tara/sessions/{self._session_id}')
            ref.set({
                'started_at': datetime.now().isoformat(),
                'status': 'running',
                'device': 'rpi4b_4gb',
            })

            self._enabled = True
            self._running = True

            # Start background upload thread
            self._thread = threading.Thread(
                target=self._upload_worker, daemon=True)
            self._thread.start()

            log.info(f"Firebase connected. Session: {self._session_id}")
            return True

        except Exception as e:
            log.error(f"Firebase connection failed: {e}")
            log.info("Cloud logging disabled — prototype will run locally only")
            return False

    def _upload_worker(self):
        """
        Background thread that processes the upload queue.
        Runs until stop() is called.
        """
        while self._running:
            try:
                # Block for up to 1 second waiting for items
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                item_type = item.get('type')

                if item_type == 'telemetry':
                    self._upload_telemetry(item['data'])

                elif item_type == 'event_frame':
                    self._upload_event_frame(
                        item['frame'], item['event_type'],
                        item['metadata'])

                elif item_type == 'session_end':
                    self._upload_session_summary(item['data'])

            except Exception as e:
                log.debug(f"Upload failed (non-critical): {e}")

            finally:
                self._queue.task_done()

    def _upload_telemetry(self, data):
        """Push telemetry data point to Firebase Realtime DB."""
        ref = db.reference(
            f'/tara/sessions/{self._session_id}/telemetry')
        ref.push(data)

    def _upload_event_frame(self, frame, event_type, metadata):
        """
        Upload a detection event frame to Firebase Storage.

        Args:
            frame: BGR numpy array
            event_type: "pothole", "sign", "departure", "emergency"
            metadata: Dict with detection details
        """
        # Encode frame as JPEG (quality 70 = small file, decent quality)
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        jpg_bytes = buf.tobytes()

        # Upload to Firebase Storage
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        blob_path = (f"tara/{self._session_id}/events/"
                     f"{event_type}_{timestamp}.jpg")

        bucket = storage.bucket()
        blob = bucket.blob(blob_path)
        blob.upload_from_string(jpg_bytes, content_type='image/jpeg')

        # Log event metadata to Realtime DB
        ref = db.reference(
            f'/tara/sessions/{self._session_id}/events')
        ref.push({
            'type': event_type,
            'timestamp': datetime.now().isoformat(),
            'image_path': blob_path,
            **metadata,
        })

        self._event_count += 1

    def _upload_session_summary(self, data):
        """Write final session summary to Firebase."""
        ref = db.reference(f'/tara/sessions/{self._session_id}')
        ref.update(data)

    # ── Public API (called from main pipeline) ────────────────────────

    def log_telemetry(self, fps, command, acc_result=None,
                      lane_result=None):
        """
        Log a telemetry data point. Throttled to CLOUD_TELEMETRY_INTERVAL.
        Non-blocking — drops data rather than delaying pipeline.

        Args:
            fps: Current FPS
            command: Latest Command object
            acc_result: Latest ACCResult or None
            lane_result: Latest LaneDetectionResult or None
        """
        if not self._enabled:
            return

        now = time.monotonic()
        if now - self._last_telemetry_time < self._telemetry_interval:
            return
        self._last_telemetry_time = now

        self._frame_count += 1

        data = {
            'timestamp': datetime.now().isoformat(),
            'fps': round(fps, 1),
            'steering': command.steering if command else 0,
            'speed': command.speed if command else 0,
            'flags': command.flags if command else 0,
        }

        if acc_result:
            data['distance_cm'] = round(acc_result.distance_cm, 1)
            data['acc_mode'] = acc_result.mode
            data['current_speed'] = round(acc_result.current_speed, 1)

        if lane_result:
            data['lane_detected'] = lane_result.lane_detected
            data['lane_offset'] = round(lane_result.lane_center_offset, 1)
            data['departure_warning'] = lane_result.departure_warning

        # Non-blocking put — if queue is full, drop this data point
        try:
            self._queue.put_nowait({'type': 'telemetry', 'data': data})
        except queue.Full:
            pass  # Drop rather than block

    def log_event(self, frame, event_type, metadata=None):
        """
        Log a detection event with a camera frame snapshot.
        Only called on actual detections (pothole, sign, etc.),
        not every frame.

        Args:
            frame: BGR camera frame
            event_type: "pothole", "sign", "departure", "emergency_stop"
            metadata: Optional dict with detection details
        """
        if not self._enabled:
            return

        if metadata is None:
            metadata = {}

        # Track detection counts
        if event_type == "pothole":
            self._detection_counts["potholes_detected"] += 1
        elif event_type == "sign":
            self._detection_counts["signs_detected"] += 1
        elif event_type == "departure":
            self._detection_counts["lane_departures"] += 1
        elif event_type == "emergency_stop":
            self._detection_counts["emergency_stops"] += 1

        # Downscale frame for upload (320x240 is plenty for logging)
        small_frame = cv2.resize(frame, (320, 240))

        try:
            self._queue.put_nowait({
                'type': 'event_frame',
                'frame': small_frame,
                'event_type': event_type,
                'metadata': metadata,
            })
        except queue.Full:
            pass  # Drop rather than block

    def stop(self):
        """
        Finalize session and upload summary.
        Called when ADAS pipeline stops.
        """
        if not self._enabled:
            return

        runtime = time.monotonic() - self._session_start

        summary = {
            'status': 'completed',
            'ended_at': datetime.now().isoformat(),
            'runtime_seconds': round(runtime, 1),
            'total_events': self._event_count,
            'detection_counts': self._detection_counts,
        }

        try:
            self._queue.put_nowait({
                'type': 'session_end',
                'data': summary,
            })
        except queue.Full:
            pass

        # Wait for remaining uploads (max 5 seconds)
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

        remaining = self._queue.qsize()
        if remaining > 0:
            log.warning(f"{remaining} uploads dropped (timeout)")

        log.info(f"Cloud session ended. Events: {self._event_count}, "
                 f"Runtime: {runtime:.0f}s")

    @property
    def is_enabled(self):
        return self._enabled


class LocalSessionRecorder:
    """
    Records session data locally to the RPi SD card / USB drive.
    Fallback when cloud is unavailable, and useful for bulk upload later.

    Saves:
      - sensor_log.csv: Timestamped sensor data
      - events/: Detection event frame snapshots
    """

    def __init__(self, config):
        self.cfg = config
        base = getattr(config, 'LOCAL_RECORDING_DIR',
                       os.path.join(config.BASE_DIR, 'recordings'))
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_dir = os.path.join(base, self._session_id)
        self._events_dir = os.path.join(self._session_dir, 'events')
        self._csv_file = None
        self._csv_path = None
        self._enabled = False
        self._event_count = 0

        # Throttle CSV writes
        self._last_csv_time = 0
        self._csv_interval = 0.5  # Write every 500ms

    def start(self):
        """Create session directory and open CSV file."""
        try:
            os.makedirs(self._events_dir, exist_ok=True)
            self._csv_path = os.path.join(self._session_dir, 'sensor_log.csv')
            self._csv_file = open(self._csv_path, 'w')
            self._csv_file.write(
                "timestamp,fps,steering,speed,flags,"
                "distance_cm,acc_mode,lane_detected,lane_offset,"
                "departure_warning\n"
            )
            self._enabled = True
            log.info(f"Local recording started: {self._session_dir}")
            return True
        except Exception as e:
            log.error(f"Failed to start local recording: {e}")
            return False

    def log_telemetry(self, fps, command, acc_result=None,
                      lane_result=None):
        """Write a telemetry row to CSV (throttled)."""
        if not self._enabled:
            return

        now = time.monotonic()
        if now - self._last_csv_time < self._csv_interval:
            return
        self._last_csv_time = now

        row = [
            datetime.now().isoformat(),
            f"{fps:.1f}",
            str(command.steering) if command else "0",
            str(command.speed) if command else "0",
            str(command.flags) if command else "0",
            f"{acc_result.distance_cm:.1f}" if acc_result else "",
            acc_result.mode if acc_result else "",
            str(lane_result.lane_detected) if lane_result else "",
            f"{lane_result.lane_center_offset:.1f}" if lane_result else "",
            str(lane_result.departure_warning) if lane_result else "",
        ]

        try:
            self._csv_file.write(",".join(row) + "\n")
            self._csv_file.flush()
        except Exception:
            pass

    def log_event(self, frame, event_type, metadata=None):
        """Save an event frame as JPEG locally."""
        if not self._enabled:
            return

        timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
        filename = f"{event_type}_{timestamp}.jpg"
        filepath = os.path.join(self._events_dir, filename)

        try:
            # Save at reduced size
            small = cv2.resize(frame, (320, 240))
            cv2.imwrite(filepath, small, [cv2.IMWRITE_JPEG_QUALITY, 80])
            self._event_count += 1
        except Exception:
            pass

    def stop(self):
        """Close CSV and finalize recording."""
        if self._csv_file:
            self._csv_file.close()
        if self._enabled:
            log.info(f"Local recording saved: {self._session_dir} "
                     f"({self._event_count} events)")

    @property
    def is_enabled(self):
        return self._enabled
