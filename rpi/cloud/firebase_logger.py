"""
TARA ADAS — Firebase Logger (REMOVED)
Cloud logging has been removed from this project.
This stub file exists so any stale imports don't cause ImportError.
"""


class FirebaseLogger:
    """No-op stub — Firebase removed."""
    def __init__(self, *a, **kw): pass
    def connect(self): return False
    def log_telemetry(self, *a, **kw): pass
    def log_event(self, *a, **kw): pass
    def stop(self): pass
    is_enabled = False


class LocalSessionRecorder:
    """No-op stub — local CSV recorder removed."""
    def __init__(self, *a, **kw): pass
    def start(self): return False
    def log_telemetry(self, *a, **kw): pass
    def log_event(self, *a, **kw): pass
    def stop(self): pass
    is_enabled = False
