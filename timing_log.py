"""
Per-request timing log — appends to a single shared file.

Usage:
    from timing_log import tlog, reset_timing, save_timing

    reset_timing()           # call once at the start of each request
    tlog("something: 1.2s")  # replaces print() for timing lines
    save_timing(path)        # appends this request's timing block to path
"""
import threading
from datetime import datetime

_local = threading.local()


def reset_timing():
    """Clear the timing log for the current thread (new request)."""
    _local.lines = []


def tlog(msg: str):
    """Print the timing message and append it to the per-request log."""
    print(msg)
    if not hasattr(_local, "lines"):
        _local.lines = []
    _local.lines.append(msg)


def save_timing(path: str):
    """Append the accumulated timing lines for this request to a shared file."""
    lines = getattr(_local, "lines", [])
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"Request — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'=' * 60}\n")
        for line in lines:
            f.write(line.strip() + "\n")
