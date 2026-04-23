"""
TARA ADAS — FPS Counter & Performance Monitor
Tracks frame processing times for each module.
"""
import time
from collections import deque


class FPSCounter:
    """Tracks FPS using a rolling window of frame timestamps."""

    def __init__(self, window_size=30):
        self.window_size = window_size
        self.timestamps = deque(maxlen=window_size)
        self.module_times = {}  # module_name → deque of durations

    def tick(self):
        """Record a new frame timestamp."""
        self.timestamps.append(time.monotonic())

    def fps(self):
        """Calculate current FPS from rolling window."""
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.timestamps) - 1) / elapsed

    def start_module(self, name):
        """Start timing a module. Returns a token to pass to stop_module."""
        return (name, time.monotonic())

    def stop_module(self, token):
        """Stop timing a module and record its duration."""
        name, start_time = token
        duration = (time.monotonic() - start_time) * 1000  # Convert to ms
        if name not in self.module_times:
            self.module_times[name] = deque(maxlen=self.window_size)
        self.module_times[name].append(duration)
        return duration

    def module_avg_ms(self, name):
        """Get average processing time for a module in milliseconds."""
        if name not in self.module_times or len(self.module_times[name]) == 0:
            return 0.0
        return sum(self.module_times[name]) / len(self.module_times[name])

    def summary(self):
        """Return a formatted summary string of all module timings."""
        lines = [f"FPS: {self.fps():.1f}"]
        for name, times in sorted(self.module_times.items()):
            avg = sum(times) / len(times) if times else 0
            latest = times[-1] if times else 0
            lines.append(f"  {name}: {latest:.1f}ms (avg: {avg:.1f}ms)")
        return " | ".join(lines)

    def reset(self):
        """Reset all counters."""
        self.timestamps.clear()
        self.module_times.clear()
