"""
TARA ADAS — Threaded Camera Capture
Supports both live camera devices and video files.

Live camera: streams latest frame, drops stale frames via BUFFERSIZE=1.
Video file:  plays frame-by-frame at the video's native FPS; loops at EOF.
"""
import cv2
import os
import threading
import time
from utils.logger import get_logger

log = get_logger("Camera")


class CameraCapture:
    """
    Threaded camera capture that always provides the latest frame.

    Works with:
      - Live cameras: index=0 (or 1, 2 …)
      - Video files:  index="/path/to/video.mp4"

    Usage:
        cam = CameraCapture(index=0, width=640, height=480)
        cam.start()
        frame = cam.read()   # Returns the latest frame
        cam.stop()
    """

    def __init__(self, index=0, width=640, height=480, fps=30):
        """
        Args:
            index: Camera device index (int) or video file path (str)
            width: Desired capture width (ignored for video files)
            height: Desired capture height (ignored for video files)
            fps: Target FPS — used as the playback rate for video files
        """
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps

        # Detect whether we are opening a video file or a live camera
        self._is_video_file = isinstance(index, str) and os.path.isfile(index)

        self._cap = None
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._frame_count = 0
        self._eof_loops = 0      # number of times the video looped

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """Open the source and start the capture thread."""
        src_type = "video file" if self._is_video_file else "camera"
        log.info(f"Opening {src_type} {self.index!r} "
                 f"at {self.width}x{self.height} @ {self.fps}fps")

        self._cap = cv2.VideoCapture(self.index)

        if not self._cap.isOpened():
            log.error(f"Failed to open {src_type} {self.index!r}")
            raise RuntimeError(f"Cannot open {src_type}: {self.index}")

        if not self._is_video_file:
            # Live camera — configure resolution and buffer
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always get latest frame
        else:
            # Video file — read native FPS for correct playback throttling
            native_fps = self._cap.get(cv2.CAP_PROP_FPS)
            if native_fps > 0:
                self.fps = native_fps
            total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            log.info(f"Video: {total} frames @ {self.fps:.1f} fps")

        # Grab one frame to verify and populate self._frame before the thread starts
        ret, frame = self._cap.read()
        if not ret:
            log.error("Opened but failed to read initial frame")
            raise RuntimeError("Initial frame read failed")

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info(f"Camera ready: actual resolution {actual_w}x{actual_h}")

        self._frame = frame
        self._running = True
        self._thread = threading.Thread(
            target=self._video_loop if self._is_video_file else self._camera_loop,
            daemon=True,
        )
        self._thread.start()
        return self

    def read(self):
        """
        Return the latest frame (thread-safe copy).

        Returns:
            numpy.ndarray BGR image, or None if no frame is available yet.
        """
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    @property
    def frame_count(self):
        """Total frames captured / decoded since start."""
        return self._frame_count

    @property
    def is_running(self):
        """Whether the capture thread is active."""
        return self._running

    def stop(self):
        """Stop the capture thread and release the resource."""
        log.info("Stopping camera capture...")
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        log.info(f"Camera stopped. Total frames captured: {self._frame_count}")

    # ── Context manager support ───────────────────────────────────────────────

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.stop()

    # ── Internal loops ────────────────────────────────────────────────────────

    def _camera_loop(self):
        """
        Live camera: grabs frames as fast as the camera delivers them.
        BUFFERSIZE=1 already ensures we always get the most recent frame.
        """
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
                    self._frame_count += 1
            else:
                # Transient read error on live camera — warn and retry
                log.warning("Camera frame read failed, retrying...")
                time.sleep(0.01)

    def _video_loop(self):
        """
        Video file: decode at native FPS and loop back to frame 0 at EOF.
        This lets the tester replay the same video continuously without
        restarting the pipeline.
        """
        frame_interval = 1.0 / self.fps  # seconds per frame

        while self._running:
            t0 = time.monotonic()

            ret, frame = self._cap.read()

            if ret:
                with self._lock:
                    self._frame = frame
                    self._frame_count += 1

                # Throttle to video FPS so we don't burn 100% CPU
                elapsed = time.monotonic() - t0
                sleep_for = frame_interval - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)

            else:
                # EOF reached — loop back to the beginning
                self._eof_loops += 1
                log.info(f"Video EOF — looping (loop #{self._eof_loops})")
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                # Small pause to avoid hammering in case seek fails
                time.sleep(0.05)
