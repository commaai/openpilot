#!/usr/bin/env python3

import time
import statistics
import cereal.messaging as messaging

from typing import Dict

camera_states = [
  'roadCameraState',
  'wideRoadCameraState',
  'driverCameraState'
]

def fmt(val):
  ref = 0.05
  return f"{val:.6f} ({100 * val / ref:.2f}%)"

if __name__ == "__main__":
  sm = messaging.SubMaster(camera_states)

  prev_sof = {state: None for state in camera_states}
  diffs: Dict[str, list] = {state: [] for state in camera_states}

  st = time.monotonic()
  while True:
    sm.update()

    for state in camera_states:
      if sm.updated[state]:
        if prev_sof[state] is not None:
          diffs[state].append((sm[state].timestampSof - prev_sof[state]) / 1e9)
        prev_sof[state] = sm[state].timestampSof

    if time.monotonic() - st > 10:
      for state in camera_states:
        values = diffs[state]
        ref = 0.05
        print(f"{state}  \tMean: {fmt(statistics.mean(values))} \t Min: {fmt(min(values))} \t Max: {fmt(max(values))} \t Std: {statistics.stdev(values):.6f} \t Num frames: {len(values)}")
        diffs[state] = []

      print()
      st = time.monotonic()
