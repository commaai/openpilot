#!/usr/bin/env python3
"""Script to fill up EON with fake data"""

import os

from selfdrive.loggerd.config import ROOT, get_available_percent
from selfdrive.loggerd.tests.loggerd_tests_common import create_random_file


if __name__ == "__main__":
  segment_idx = 0
  while True:
    seg_name = "1970-01-01--00-00-00--%d" % segment_idx
    seg_path = os.path.join(ROOT, seg_name)

    print(seg_path)

    create_random_file(os.path.join(seg_path, 'fcamera.hevc'), 36)
    create_random_file(os.path.join(seg_path, 'rlog.bz2'), 2)

    segment_idx += 1

    # Fill up to 99 percent
    available_percent = get_available_percent()
    if available_percent < 1.0:
        break
