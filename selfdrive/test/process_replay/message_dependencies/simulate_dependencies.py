#!/usr/bin/env python3

import random

'''
Dummy example.

Each proccess has sub msg requirements and pub msg actions

RULES = {
  "A": (["d", "g"], ['a', 'aa']),
  "B": (["a", "c"], ['b', 'bb']),
  "C": (["aa", "d"], ['c']),
  "D": (["a"], ['d']),
  "E": (["b"], ['e']),
  "F": (["bb"], ['f']),
  "G": (["e", "f"], ['g']),
}

Answer: A, D, C, B, E/F, G
'''

#TODO: some processes have more than one thread, all of the sub requirements are not necessary
# Based on image in blog post
RULES = {
  'manager': (['deviceState'], ['managerState']),
  'thermald': (['pandaState', 'gpsLocationExternal', 'managerState'], ['deviceState']),
  'sensord': (['deviceState'], ['sensorEvents']),
  'boardd': (['deviceState', 'driverCameraState'], ['can', 'ubloxRaw', 'pandaState']),
  'camerad': (['sensorEvents'], ['driverCameraState', 'wideRoadCameraState', 'roadCameraState']),
  'modeld': (['roadCameraState', 'wideRoadCameraState', 'liveCalibration', 'lateralPlan'], ['modelV2', 'cameraOdometry']),
  'dmonitoringmodeld': (['driverCameraState'], ['driverState']),
  'dmonitoringd': (['driverState', 'modelV2', 'carState', 'liveCalibration', 'controlsState'], ['driverMonitoringState']),
  'ubloxd': (['ubloxRaw'], ['gpsLocationExternal']),
  'locationd': (['gpsLocationExternal', 'sensorEvents', 'liveCalibration', 'cameraOdometry', 'carState'], ['liveLocationKalman']),
  'paramsd': (['carState', 'liveLocationKalman'], ['liveParameters']),
  'calibrationd': (['carState', 'cameraOdometry'], ['liveCalibration']),
  'radard': (['can', 'modelV2', 'carState'], ['radarState']),
  'plannerd': (['radarState', 'modelV2', 'carState', 'controlsState'], ['longitudinalPlan', 'lateralPlan']),
  'controlsd': (['radarState', 'longitudinalPlan', 'lateralPlan', 'liveCalibration', 'managerState', 'deviceState', 'modelV2', 'pandaState', 'can', 'liveLocationKalman', 'wideRoadCameraState', 'roadCameraState', 'liveParameters', 'driverMonitoringState'], ['carState', 'controlsState']),
}

# shuffle to ensure no cheating
l = list(RULES.items())
random.shuffle(l)
status = {name: {s: False for s in sub_pub[0]} for name, sub_pub in l}

def run_proc(proc):
  for msg in RULES[proc][1]:
    for name, sub_status in status.items():
      if msg in sub_status:
        status[name][msg] = True
  status[proc] = {s: False for s in RULES[proc][0]}

# run some processes to give some inital messages
run_proc("calibrationd")
run_proc("thermald")
run_proc("camerad")
run_proc("plannerd")
run_proc("controlsd")

# simulate system
for _ in range(100):
  for proc, sub_status in status.items():
    if all(sub_status.values()):
      run_proc(proc)
      print(proc)

#exit()

# check for processes which where almost able to run
print("Exit status")
for name, sub_status in status.items():
  print(name)
  count = 0
  for sub, stat in sub_status.items():
    if not stat:
      print("  ", sub)
    else:
      count += 1
  if count < len(sub_status):
    print("   Check:", count)

