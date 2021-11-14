#!/usr/bin/env python3
import time
import cereal.messaging as messaging


if __name__ == "__main__":
  sm = messaging.SubMaster(['deviceState', 'pandaStates', 'modelV2', 'liveCalibration', 'driverMonitoringState', 'longitudinalPlan', 'lateralPlan'])

  i = 0
  while True:
    sm.update(0)

    i += 1
    if i % 100 == 0:
      print()
      print("alive", sm.alive)
      print("valid", sm.valid)

    time.sleep(0.01)
