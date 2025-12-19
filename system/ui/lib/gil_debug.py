"""
GIL (Global Interpreter Lock) contention debugger.

This module helps identify Python code that's holding the GIL and causing UI stutters.

Usage:
    Set GIL_DEBUG=1 environment variable to enable.
    Set GIL_THRESHOLD_MS=10 to set the threshold in milliseconds (default: 10ms).
    Set GIL_STACK_DEPTH=20 to control stack trace depth (default: 20).

When enabled, it will:
    1. Monitor GIL acquisition/release times
    2. Log stack traces when GIL is held longer than threshold
    3. Track which functions/code paths hold the GIL longest
"""

import os
import sys
import time
import threading
import traceback
import faulthandler
from collections import defaultdict
from typing import Optional

try:
    from openpilot.common.swaglog import cloudlog
    HAS_CLOUDLOG = True
except ImportError:
    HAS_CLOUDLOG = False

# Configuration
ENABLED = os.getenv("GIL_DEBUG", "0") == "1"
THRESHOLD_MS = float(os.getenv("GIL_THRESHOLD_MS", "10"))  # Log if GIL held > 10ms
STACK_DEPTH = int(os.getenv("GIL_STACK_DEPTH", "20"))
LOG_TO_FILE = os.getenv("GIL_LOG_FILE", "")
USE_CLOUDLOG = os.getenv("GIL_USE_CLOUDLOG", "1") == "1"  # Use cloudlog by default on device

# Global state
_gil_stats: dict[str, list[float]] = defaultdict(list)
_gil_lock = threading.Lock()
_last_gil_acquire_time: Optional[float] = None
_last_gil_acquire_stack: Optional[list] = None
_monitor_thread: Optional[threading.Thread] = None
_monitoring = False


def _get_stack_trace(depth: int = STACK_DEPTH) -> list[str]:
    """Get current stack trace as list of strings."""
    stack = []
    for frame_info in traceback.extract_stack()[:-2][-depth:]:  # Exclude this function
        stack.append(f"  File \"{frame_info.filename}\", line {frame_info.lineno}, in {frame_info.name}\n    {frame_info.line}")
    return stack


def _log_message(msg: str):
    """Log message to appropriate destination."""
    if LOG_TO_FILE:
        with open(LOG_TO_FILE, "a") as f:
            f.write(msg + "\n")
            f.flush()
    elif USE_CLOUDLOG and HAS_CLOUDLOG:
        cloudlog.warning(msg)
    else:
        print(msg, file=sys.stderr, flush=True)


def _monitor_gil():
    """Background thread that periodically checks for long GIL holds.

    Note: This is a best-effort approach. For more accurate detection,
    use GILTracker context managers around suspected code blocks.
    """
    if LOG_TO_FILE:
        log_file = open(LOG_TO_FILE, "a")
    else:
        log_file = None

    last_check = time.monotonic()

    while _monitoring:
        time.sleep(0.005)  # Check every 5ms

        current_time = time.monotonic()
        with _gil_lock:
            if _last_gil_acquire_time is not None:
                hold_time_ms = (current_time - _last_gil_acquire_time) * 1000

                if hold_time_ms > THRESHOLD_MS:
                    # GIL held too long - log it
                    stack_str = "\n".join(_last_gil_acquire_stack) if _last_gil_acquire_stack else "No stack trace"

                    msg = f"GIL HELD FOR {hold_time_ms:.2f}ms (> {THRESHOLD_MS}ms threshold)\nStack trace:\n{stack_str}"

                    if log_file:
                        log_file.write(f"\n{'='*80}\n")
                        log_file.write(msg)
                        log_file.write(f"\n{'='*80}\n\n")
                        log_file.flush()
                    else:
                        _log_message(msg)

                    # Track stats by top frame
                    if _last_gil_acquire_stack:
                        top_frame = _last_gil_acquire_stack[0] if _last_gil_acquire_stack else "unknown"
                        _gil_stats[top_frame].append(hold_time_ms)

        last_check = current_time

    if LOG_TO_FILE:
        log_file.close()


def _gil_acquired():
    """Called when GIL is acquired."""
    global _last_gil_acquire_time, _last_gil_acquire_stack
    if not ENABLED:
        return

    with _gil_lock:
        _last_gil_acquire_time = time.monotonic()
        _last_gil_acquire_stack = _get_stack_trace()


def _gil_released():
    """Called when GIL is released."""
    global _last_gil_acquire_time, _last_gil_acquire_stack
    if not ENABLED:
        return

    with _gil_lock:
        if _last_gil_acquire_time is not None:
            hold_time_ms = (time.monotonic() - _last_gil_acquire_time) * 1000
            if hold_time_ms > THRESHOLD_MS:
                # Track stats
                if _last_gil_acquire_stack:
                    top_frame = _last_gil_acquire_stack[0] if _last_gil_acquire_stack else "unknown"
                    _gil_stats[top_frame].append(hold_time_ms)

        _last_gil_acquire_time = None
        _last_gil_acquire_stack = None


class GILTracker:
    """Context manager to track GIL hold time for a specific code block."""

    def __init__(self, name: str = ""):
        self.name = name
        self.start_time: Optional[float] = None

    def __enter__(self):
        if ENABLED:
            self.start_time = time.monotonic()
            _gil_acquired()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if ENABLED and self.start_time is not None:
            hold_time_ms = (time.monotonic() - self.start_time) * 1000
            if hold_time_ms > THRESHOLD_MS:
                stack = _get_stack_trace()
                msg = f"[GIL_DEBUG] {self.name} held GIL for {hold_time_ms:.2f}ms\n" + "\n".join(stack)
                _log_message(msg)
            _gil_released()


def start_monitoring():
    """Start GIL monitoring in background thread."""
    global _monitor_thread, _monitoring

    if not ENABLED:
        return

    if _monitor_thread is None or not _monitor_thread.is_alive():
        _monitoring = True
        _monitor_thread = threading.Thread(target=_monitor_gil, daemon=True)
        _monitor_thread.start()
        msg = f"[GIL_DEBUG] Started monitoring (threshold: {THRESHOLD_MS}ms)"
        if LOG_TO_FILE:
            msg += f" - logging to {LOG_TO_FILE}"
        elif USE_CLOUDLOG and HAS_CLOUDLOG:
            msg += " - logging to cloudlog (/data/log/)"
        else:
            msg += " - logging to stderr"
        _log_message(msg)


def stop_monitoring():
    """Stop GIL monitoring."""
    global _monitoring
    _monitoring = False


def print_stats():
    """Print statistics about GIL hold times."""
    if not ENABLED or not _gil_stats:
        return

    print("\n" + "="*80, file=sys.stderr)
    print("GIL CONTENTION STATISTICS", file=sys.stderr)
    print("="*80, file=sys.stderr)

    # Sort by total time
    stats_list = []
    for frame, times in _gil_stats.items():
        stats_list.append((frame, len(times), sum(times), max(times), sum(times)/len(times)))

    stats_list.sort(key=lambda x: x[2], reverse=True)  # Sort by total time

    print(f"\n{'Frame':<60} {'Count':<8} {'Total ms':<12} {'Max ms':<10} {'Avg ms':<10}", file=sys.stderr)
    print("-"*80, file=sys.stderr)

    for frame, count, total_ms, max_ms, avg_ms in stats_list[:20]:  # Top 20
        frame_short = frame[:58] + ".." if len(frame) > 60 else frame
        print(f"{frame_short:<60} {count:<8} {total_ms:<12.2f} {max_ms:<10.2f} {avg_ms:<10.2f}", file=sys.stderr)

    print("="*80 + "\n", file=sys.stderr)


# Monkey-patch sys.settrace to track function calls (more invasive but more detailed)
_original_settrace = sys.settrace

def _gil_trace(frame, event, arg):
    """Trace function to detect GIL holds."""
    if event == 'call':
        _gil_acquired()
    elif event == 'return':
        _gil_released()
    return _gil_trace


if ENABLED:
    # Use faulthandler to dump stack on signal
    faulthandler.enable()

    # Start monitoring thread
    start_monitoring()

    # Optionally enable function tracing (very verbose, use with caution)
    if os.getenv("GIL_TRACE_FUNCTIONS", "0") == "1":
        sys.settrace(_gil_trace)
        print("[GIL_DEBUG] Function tracing enabled (very verbose!)", file=sys.stderr)

