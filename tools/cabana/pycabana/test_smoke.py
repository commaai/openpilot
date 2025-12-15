#!/usr/bin/env python3
"""Smoke test for pycabana - runs the app headlessly and checks for errors."""

import os
import signal
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TEST_ROUTE = "a2a0ccea32023010|2023-07-27--13-01-19/0"

ERROR_PATTERNS = [
  "_pythonToCppCopy",
  "QThread: Destroyed while",
  "Traceback",
  "Segmentation fault",
  "Aborted",
  "core dumped",
]


def main():
  env = {**os.environ, }#"QT_QPA_PLATFORM": "offscreen"}
  cmd = [sys.executable, "tools/cabana/pycabana/cabana.py", TEST_ROUTE]

  print(f"Starting pycabana with route {TEST_ROUTE}")
  proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=REPO_ROOT, env=env)

  # Let it run for a few seconds
  time.sleep(5)

  # Send ctrl-c
  print("Sending SIGINT...")
  os.kill(proc.pid, signal.SIGINT)

  # Wait for exit
  try:
    stdout, stderr = proc.communicate(timeout=5)
  except subprocess.TimeoutExpired:
    proc.kill()
    stdout, stderr = proc.communicate()
    print("FAILED: Process hung after SIGINT")
    return 1

  # Check for error patterns
  output = stdout + stderr
  errors = [p for p in ERROR_PATTERNS if p in output]
  if errors:
    print(f"FAILED: Found error patterns: {errors}")
    print("--- stderr ---")
    print(stderr)
    return 1

  # Check exit code (0 or -2/SIGINT are OK)
  if proc.returncode not in (0, -2, -signal.SIGINT):
    print(f"FAILED: Bad exit code {proc.returncode}")
    print("--- stderr ---")
    print(stderr)
    return 1

  print(f"OK (exit code {proc.returncode})")
  return 0


if __name__ == "__main__":
  sys.exit(main())
