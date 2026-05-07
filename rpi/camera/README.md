# camera/ — Threaded Camera Capture

Provides `CameraCapture`, a thread-safe wrapper around OpenCV `VideoCapture`
that always delivers the **latest** frame without blocking the ADAS pipeline loop.

---

## Usage

```python
from camera.capture import CameraCapture

# Live camera
cam = CameraCapture(index=0, width=640, height=480, fps=30)

# Video file (loops at end-of-file for continuous replay)
cam = CameraCapture(index="challenge_video.mp4")

cam.start()
frame = cam.read()   # Returns latest BGR numpy array
cam.stop()
```

Context manager is also supported:

```python
with CameraCapture(index=0) as cam:
    frame = cam.read()
```

---

## Architecture

```
Main thread (ADAS pipeline)          Capture daemon thread
┌──────────────────────┐            ┌──────────────────────┐
│ cam.read()           │            │ _camera_loop()       │
│  -> lock             │            │  cap.read() -> frame │
│  -> copy(frame)      │            │  lock                │
│  -> unlock           │            │  self._frame = frame │
│  -> return copy      │            │  unlock              │
└──────────────────────┘            └──────────────────────┘
          shared: self._frame (threading.Lock protected)
```

The main thread never waits for the camera driver — it always gets the most recently decoded frame.

---

## Frame Rate Mathematics

### Live Camera

`CAP_PROP_BUFFERSIZE = 1` is set so OpenCV only buffers one frame. The capture thread calls `cap.read()` as fast as the USB camera delivers frames (typically 30 fps). Pipeline latency is bounded by one frame period:

```
max_latency = 1 / camera_fps = 1/30 ≈ 33 ms
```

### Video File Playback

The video loop reads native FPS from the file header and throttles with `time.sleep`:

```
frame_interval = 1.0 / fps

# per loop iteration:
elapsed   = monotonic() - t0
sleep_for = frame_interval - elapsed
if sleep_for > 0:
    sleep(sleep_for)
```

At EOF, the video rewinds via `CAP_PROP_POS_FRAMES = 0` for seamless looping.

### Effective Pipeline FPS

The pipeline FPS is constrained by the slowest step:

```
t_pipeline = t_lane + t_tsr/N_tsr + t_pothole/N_pothole + t_decision

FPS_effective = 1 / t_pipeline
```

Where N_tsr = 4 (TSR runs every 4th frame) and N_pothole = 4. This is why per-module latencies reported by `FPSCounter` matter — the total cycle time must stay below 1/30 s ≈ 33 ms for real-time operation.

---

## Design Notes

- Both modes run a **daemon thread** so the process exits cleanly on `Ctrl+C` without a join timeout issue.
- `read()` returns a `frame.copy()` — the pipeline owns a private copy and the capture thread can overwrite `self._frame` immediately.
- A single initial `cap.read()` in `start()` blocks until the first frame is available, guaranteeing `read()` never returns `None` on the first call.
