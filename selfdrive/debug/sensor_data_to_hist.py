#!/usr/bin/env python3
'''
printing the gap between interrupts in a histogram to check if the
frequency is what we expect, the bmx is not interrup driven for as we
get interrupts in a 2kHz rate.
'''

import argparse
import sys
from collections import defaultdict

from tools.lib.logreader import LogReader
from tools.lib.route import Route

import matplotlib.pyplot as plt

SRC_BMX = "bmx055"
SRC_LSM = "lsm6ds3"


def parseEvents(log_reader):
  bmx_data = defaultdict(list)
  lsm_data = defaultdict(list)

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
        bmx_data["accel"].append(d["timestamp"] / 1e9)

      if d["source"] == SRC_BMX and "gyroUncalibrated" in d:
        bmx_data["gyro"].append(d["timestamp"] / 1e9)

      if d["source"] == SRC_LSM and "acceleration" in d:
        lsm_data["accel"].append(d["timestamp"] / 1e9)

      if d["source"] == SRC_LSM and "gyroUncalibrated" in d:
        lsm_data["gyro"].append(d["timestamp"] / 1e9)

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
  hz  = 1 / avg
  print(f"{sensor}: data_points: {len(data)} avg [ns]: {avg} avg [Hz]: {hz}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("route", type=str, help="route name")
  parser.add_argument("segment", type=int, help="segment number")
  args = parser.parse_args()

  r = Route(args.route)
  logs = r.log_paths()

  if len(logs) == 0:
    print("NO data routes")
    sys.exit(0)

  if args.segment >= len(logs):
    print(f"RouteID: {args.segment} out of range, max: {len(logs) -1}")
    sys.exit(0)

  lr = LogReader(logs[args.segment])
  bmx_data, lsm_data = parseEvents(lr)

  # sort bmx accel data, and then cal all the diffs, and to a histogram of those
  bmx_accel, bmx_accel_diffs = cleanData(bmx_data["accel"])
  bmx_gyro, bmx_gyro_diffs = cleanData(bmx_data["gyro"])
  lsm_accel, lsm_accel_diffs = cleanData(lsm_data["accel"])
  lsm_gyro, lsm_gyro_diffs = cleanData(lsm_data["gyro"])

  # get out the averages
  logAvgValues(bmx_accel_diffs, "bmx accel")
  logAvgValues(bmx_gyro_diffs, "bmx gyro ")
  logAvgValues(lsm_accel_diffs, "lsm accel")
  logAvgValues(lsm_gyro_diffs, "lsm gyro ")

  fig, axs = plt.subplots(1, 2, tight_layout=True)
  axs[0].hist(bmx_accel_diffs, bins=50)
  axs[0].set_title("bmx_accel")
  axs[1].hist(bmx_gyro_diffs, bins=50)
  axs[1].set_title("bmx_gyro")

  figl, axsl = plt.subplots(1, 2, tight_layout=True)
  axsl[0].hist(lsm_accel_diffs, bins=50)
  axsl[0].set_title("lsm_accel")
  axsl[1].hist(lsm_gyro_diffs, bins=50)
  axsl[1].set_title("lsm_gyro")

  print("check plot...")
  plt.show()

