# utils/ — Shared Utilities

Small, focused helpers used across the ADAS pipeline.

---

## `fps_counter.py` — FPSCounter

Tracks overall pipeline FPS and per-module processing times using a rolling window.

### FPS Calculation

Given a deque of N = 30 recent frame timestamps `[t_0, t_1, ..., t_{N-1}]`:

```
FPS = (N - 1) / (t_{N-1} - t_0)
```

This is more stable than an instantaneous `1 / dt` measurement because it averages over 30 frames, suppressing single-frame spikes.

### Per-Module Timing

Each module is wrapped with a token:

```python
token = fps.start_module("Lane")        # records (name, t_start)
# ... run module ...
duration_ms = fps.stop_module(token)    # records (t_now - t_start) * 1000
```

The duration is appended to a per-module deque of depth 30. Average latency:

```
avg_ms = sum(durations) / len(durations)
```

### Usage

```python
from utils.fps_counter import FPSCounter

fps = FPSCounter(window_size=30)

fps.tick()                              # call once per main loop iteration
t = fps.start_module("Lane")
# ... run lane detection ...
fps.stop_module(t)

print(fps.fps())                        # e.g. 27.4
print(fps.summary())
# "FPS: 27.4 | Lane: 4.2ms (avg: 4.1ms) | TSR: 21.1ms (avg: 20.8ms)"
```

---

## `logger.py` — Structured Logger

Sets up a `logging.Logger` with:
- Coloured console output (DEBUG = cyan, INFO = white, WARNING = yellow, ERROR = red)
- Optional rotating file handler (writes to `tara_adas.log`)
- Child loggers per module via `get_logger("ModuleName")`

### Usage

```python
from utils.logger import setup_logger, get_logger

# Call once at startup in main.py
setup_logger("TARA", level="INFO", log_file="tara_adas.log")

# In each module
log = get_logger("LaneDet")
log.info("Lane detector initialized")
log.debug(f"Offset: {offset:.2f} px, steering: {steer:.4f}")
log.warning("Lane temporarily lost")
log.error("Camera failed to open")
```

Log level is set in `config.LOG_LEVEL` (default `"INFO"`). Switch to `"DEBUG"` to see per-frame inference scores and decision traces.
