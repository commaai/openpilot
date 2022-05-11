import argparse
import sys
import cereal.messaging as messaging
import json
from tools.lib.logreader import LogReader
from tools.lib.route import Route

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

'''
printing the gap between interrupts in a histogram to check if the
frequency is what we expect, the bmx is not interrup driven for as we
get interrupts in a 2kHz rate.
'''

SRC_BMX = "bmx055"
SRC_LSM = "lsm6ds3trc"

def parseEvents(log_reader):
  bmx_data = {
    "accel": [], 
    "gyro":  []
  } 
  
  lsm_data = {
    "accel": [], 
    "gyro":  []
  } 

  for m in log_reader:
    # only sensorEvents
    if m.which() != 'sensorEvents':
      continue
  
    for se in m.sensorEvents:
      # convert data to dictionary
      d = se.to_dict()
      
      if d["timestamp"] == 0:
        continue # empty event?

      if d["source"] == SRC_BMX and "acceleration" in d:
        bmx_data["accel"].append(d["timestamp"])
  
      if d["source"] == SRC_BMX and "gyroUncalibrated" in d:
        bmx_data["gyro"].append(d["timestamp"])
  
      if d["source"] == SRC_LSM and "acceleration" in d:
        lsm_data["accel"].append(d["timestamp"])
  
      if d["source"] == SRC_LSM and "gyroUncalibrated" in d:
        lsm_data["gyro"].append(d["timestamp"])

  return bmx_data, lsm_data


def cleanData(data):
  if len(data) == 0:
    return [], []

  data.sort()
  prev = data[0]
  diffs = []
  for v in data[1:]:
    diffs.append(v - prev)
    prev = v
  return data, diffs


def logAvgValues(data, sensor):
  if len(data) == 0:
    print(f"{sensor}: no data to average")
    return

  avg = sum(data) / len(data)
  hz  = 1/(avg*10**-9)
  print(f"{sensor}: data_points: {len(data)} avg [ns]: {avg} avg [Hz]: {hz}")




if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("route", type=str, nargs=1, help="route name + segment number for offline usage")
  parser.add_argument("route_id", type=int, nargs=1, help="route id")
  args = parser.parse_args()

  route  = args.route[0]
  r = Route(route)
  #logs = [q_log if r_log is None else r_log for (q_log, r_log) in zip(r.qlog_paths(), r.log_paths())]
  logs = r.log_paths()
 
  if len(logs) == 0:
    print("NO data routes")
    sys.exit(0)
  
  route_id = args.route_id[0]
  if route_id > len(logs):
    print(f"RouteID: {route_id} out of range, max: {len(logs) -1}")
    sys.exit(0)

  lr = LogReader(logs[route_id])
  bmx_data, lsm_data = parseEvents(lr)

  # sort bmx accel data, and then cal all the diffs, and to a histogram of those
  bmx_accel, bmx_accel_diffs = cleanData(bmx_data["accel"])
  bmx_gyro, bmx_gyro_diffs = cleanData(bmx_data["gyro"])
  lsm_accel, lsm_accel_diffs = cleanData(lsm_data["accel"])
  lsm_gyro, lsm_gyro_diffs = cleanData(lsm_data["gyro"])
  
  # get out the averages
  logAvgValues(bmx_accel_diffs, "bmx accel")
  logAvgValues(bmx_gyro_diffs,  "bmx gyro ")
  logAvgValues(lsm_accel_diffs, "lsm accel")
  logAvgValues(lsm_gyro_diffs,  "lsm gyro ")
  
  fig, axs = plt.subplots(1, 2, tight_layout=True)
  axs[0].hist(bmx_accel_diffs, bins=50)
  axs[0].set_title("bmx_accel")
  axs[1].hist(bmx_gyro_diffs,  bins=50)
  axs[1].set_title("bmx_gyro")
  
  figl, axsl = plt.subplots(1, 2, tight_layout=True)
  axsl[0].hist(lsm_accel_diffs, bins=50)
  axsl[0].set_title("lsm_accel")
  axsl[1].hist(lsm_gyro_diffs,  bins=50)
  axsl[1].set_title("lsm_gyro")
  
  print("check plot...")
  plt.show()

