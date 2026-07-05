#!/usr/bin/env python3
"""Verify a rerun_bridge .rrd recording has expected content."""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("rrd", help="Path to .rrd file")
  parser.add_argument("--min-series", type=int, default=500)
  parser.add_argument("--require-camera", action="store_true")
  args = parser.parse_args()

  output = subprocess.check_output(["rerun", "rrd", "print", args.rrd], text=True, stderr=subprocess.STDOUT)
  checks = {
    "carState/vEgo": "carState/vEgo" in output or "/carState/vEgo" in output,
    "logs": "logs" in output,
    "map": "map" in output,
    "camera/road": "camera/road" in output,
    "Scalars": "Scalars" in output or "scalar" in output.lower(),
  }

  failed = [name for name, ok in checks.items() if not ok]
  if args.require_camera and not checks["camera/road"]:
    failed.append("camera/road(required)")

  print(output[:4000])
  print("---")
  print("checks:", checks)
  if failed:
    print("FAILED:", ", ".join(failed))
    return 1
  if len(output) < args.min_series:
    print(f"FAILED: recording output too small ({len(output)} chars)")
    return 1
  print("OK")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())