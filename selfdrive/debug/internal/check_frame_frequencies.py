#!/usr/bin/env python3

import time
import statistics
import cereal.messaging as messaging

camera_states = [
  'roadCameraState',
  'wideRoadCameraState',
  'driverCameraState'
]

def format(val):
  ref = 0.05
  return f"{val:.6f} ({100 * val / ref:.2f}%)"

if __name__ == "__main__":
  sm = messaging.SubMaster(camera_states)

  prev_sof = {state: None for state in camera_states}
  diffs = {state: [] for state in camera_states}

  st = time.monotonic()
  while True:
    sm.update()

    for state in camera_states:
      if sm.updated[state]:
        if prev_sof[state] is not None:
          diffs[state].append((sm[state].timestampSof - prev_sof[state]) / 1e9)
        prev_sof[state] = sm[state].timestampSof

    if time.monotonic() - st > 5:
      for state in camera_states:
        values = diffs[state]
        ref = 0.05
        print(f"{state}  \tMean: {format(statistics.mean(values))} \t Min: {format(min(values))} \t Max: {format(max(values))} \t Std: {statistics.stdev(values):.6f} \t Num frames: {len(values)}")
        diffs[state] = []

      print()
      st = time.monotonic()
    
