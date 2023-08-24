#!/usr/bin/env python3
"""Script to fill up storage with fake data"""

from pathlib import Path

from openpilot.system.loggerd.config import ROOT, get_available_percent
from openpilot.system.loggerd.tests.loggerd_tests_common import create_random_file


if __name__ == "__main__":
  segment_idx = 0
  while True:
    seg_name = f"1970-01-01--00-00-00--{segment_idx}"
    seg_path = Path(ROOT) / seg_name

    print(seg_path)

    create_random_file(seg_path / "fcamera.hevc", 36)
    create_random_file(seg_path / "rlog.bz2", 2)

    segment_idx += 1

    # Fill up to 99 percent
    available_percent = get_available_percent()
    if available_percent < 1.0:
      break
