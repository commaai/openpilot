#!/usr/bin/env python3
"""
GIL Flamegraph Profiler

Generates flamegraphs showing where the UI thread spends time (GIL holds).

Usage:
    # Enable profiling
    export GIL_FLAMEGRAPH=1
    export GIL_FLAMEGRAPH_DURATION=10  # seconds to profile
    ./selfdrive/ui/ui.py

    # Or trigger manually with signal (default: SIGUSR2)
    kill -USR2 <UI_PID>

    # Output will be saved to /tmp/gil_flamegraph.svg
"""

import os
import sys
import signal
import time
import threading
import subprocess
from pathlib import Path
from typing import Optional

ENABLED = os.getenv("GIL_FLAMEGRAPH", "0") == "1"
DURATION = float(os.getenv("GIL_FLAMEGRAPH_DURATION", "10"))
OUTPUT_FILE = os.getenv("GIL_FLAMEGRAPH_OUTPUT", "/tmp/gil_flamegraph.svg")
SAMPLE_RATE = int(os.getenv("GIL_FLAMEGRAPH_RATE", "100"))  # Hz
TRIGGER_SIGNAL = signal.SIGUSR2

_profiling = False
_profiler_process: Optional[subprocess.Popen] = None
_ui_pid = os.getpid()


def _check_py_spy():
    """Check if py-spy is available."""
    try:
        result = subprocess.run(["py-spy", "--version"],
                              capture_output=True, timeout=2)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _generate_flamegraph(duration: float = DURATION):
    """Generate flamegraph using py-spy."""
    global _profiler_process

    if not _check_py_spy():
        print(f"[GIL_FLAMEGRAPH] py-spy not found. Install with: pip install py-spy", file=sys.stderr)
        return False

    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[GIL_FLAMEGRAPH] Starting profiling for {duration}s...", file=sys.stderr)

    # Use py-spy to record
    cmd = [
        "py-spy", "record",
        "-o", str(output_path),
        "--pid", str(_ui_pid),
        "--duration", str(int(duration)),
        "--rate", str(SAMPLE_RATE),
        "--subprocesses",  # Include subprocesses
    ]

    try:
        _profiler_process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        stdout, stderr = _profiler_process.communicate()

        if _profiler_process.returncode == 0:
            print(f"[GIL_FLAMEGRAPH] Flamegraph saved to: {output_path}", file=sys.stderr)
            print(f"[GIL_FLAMEGRAPH] View with: xdg-open {output_path} (or open in browser)", file=sys.stderr)
            return True
        else:
            print(f"[GIL_FLAMEGRAPH] Error: {stderr.decode()}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"[GIL_FLAMEGRAPH] Failed to generate flamegraph: {e}", file=sys.stderr)
        return False
    finally:
        _profiler_process = None


def _signal_handler(signum, frame):
    """Handle signal to trigger profiling."""
    global _profiling

    if _profiling:
        print("[GIL_FLAMEGRAPH] Already profiling, ignoring signal", file=sys.stderr)
        return

    _profiling = True
    print(f"[GIL_FLAMEGRAPH] Signal received, starting profiling...", file=sys.stderr)

    # Run in thread to avoid blocking
    thread = threading.Thread(target=_generate_flamegraph, daemon=True)
    thread.start()

    # Reset flag after duration
    def reset_flag():
        time.sleep(DURATION + 1)
        _profiling = False

    threading.Thread(target=reset_flag, daemon=True).start()


def start_profiling():
    """Start profiling if enabled."""
    global _profiling

    if not ENABLED:
        return

    # Register signal handler
    signal.signal(TRIGGER_SIGNAL, _signal_handler)

    print(f"[GIL_FLAMEGRAPH] Enabled. Send SIGUSR2 to trigger profiling:", file=sys.stderr)
    print(f"[GIL_FLAMEGRAPH]   kill -USR2 {_ui_pid}", file=sys.stderr)
    print(f"[GIL_FLAMEGRAPH]   Output: {OUTPUT_FILE}", file=sys.stderr)

    # Auto-start if duration is set and > 0
    if DURATION > 0:
        thread = threading.Thread(target=_generate_flamegraph, daemon=True)
        thread.start()


if __name__ == "__main__":
    # Standalone mode - profile current process
    if len(sys.argv) > 1:
        pid = int(sys.argv[1])
        duration = float(sys.argv[2]) if len(sys.argv) > 2 else DURATION
        _ui_pid = pid
        _generate_flamegraph(duration)
    else:
        print("Usage: python gil_flamegraph.py <PID> [duration]")
        print("Or set GIL_FLAMEGRAPH=1 in UI process")

