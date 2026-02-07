#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path
from openpilot.selfdrive.ui.tests.diff.diff import DIFF_OUT_DIR


def run_replay(output_name):
  env = os.environ.copy()
  env["RECORD_OUTPUT"] = output_name
  cmd = [sys.executable, str(Path(__file__).parent / "replay.py")]
  return subprocess.run(cmd, env=env, check=False).returncode


def run_diff(video1, video2):
  cmd = [
    sys.executable,
    str(Path(__file__).parent / "diff.py"),
    str(DIFF_OUT_DIR / video1),
    str(DIFF_OUT_DIR / video2),
  ]
  return subprocess.run(cmd, check=False).returncode


def main():
  DIFF_OUT_DIR.mkdir(parents=True, exist_ok=True)

  video1 = "mici_ui_replay_1.mp4"
  video2 = "mici_ui_replay_2.mp4"

  print("Running replay 1...")
  rc = run_replay(video1)
  if rc != 0:
    print(f"Replay 1 failed with exit code {rc}", file=sys.stderr)
    return rc

  print("Running replay 2...")
  rc = run_replay(video2)
  if rc != 0:
    print(f"Replay 2 failed with exit code {rc}", file=sys.stderr)
    return rc

  print("Running diff...")
  return run_diff(video1, video2)


if __name__ == "__main__":
  raise SystemExit(main())
