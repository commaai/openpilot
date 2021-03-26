#!/usr/bin/env python3
# pylint: skip-file
# flake8: noqa
# type: ignore

import math
import multiprocessing

import numpy as np
from tqdm import tqdm

from selfdrive.locationd.paramsd import ParamsLearner, States
from tools.lib.logreader import LogReader
from tools.lib.route import Route

ROUTE = "b2f1615665781088|2021-03-14--17-27-47"
PLOT = True


def load_segment(segment_name):
  print(f"Loading {segment_name}")
  if segment_name is None:
    return []

  try:
    return list(LogReader(segment_name))
  except ValueError as e:
    print(f"Error parsing {segment_name}: {e}")
    return []


if __name__ == "__main__":
  route = Route(ROUTE)

  msgs = []
  with multiprocessing.Pool(24) as pool:
    for d in pool.map(load_segment, route.log_paths()):
      msgs += d

  for m in msgs:
    if m.which() == 'carParams':
      CP = m.carParams
      break

  params = {
    'carFingerprint': CP.carFingerprint,
    'steerRatio': CP.steerRatio,
    'stiffnessFactor': 1.0,
    'angleOffsetAverageDeg': 0.0,
  }

  for m in msgs:
    if m.which() == 'liveParameters':
      params['steerRatio'] = m.liveParameters.steerRatio
      params['angleOffsetAverageDeg'] = m.liveParameters.angleOffsetAverageDeg
      break

  for m in msgs:
    if m.which() == 'carState':
      last_carstate = m
      break

  print(params)
  learner = ParamsLearner(CP, params['steerRatio'], params['stiffnessFactor'], math.radians(params['angleOffsetAverageDeg']))
  msgs = [m for m in tqdm(msgs) if m.which() in ('liveLocationKalman', 'carState', 'liveParameters')]
  msgs = sorted(msgs, key=lambda m: m.logMonoTime)

  ts = []
  ts_log = []
  results = []
  results_log = []
  for m in tqdm(msgs):
    if m.which() == 'carState':
      last_carstate = m

    elif m.which() == 'liveLocationKalman':
      t = last_carstate.logMonoTime / 1e9
      learner.handle_log(t, 'carState', last_carstate.carState)

      t = m.logMonoTime / 1e9
      learner.handle_log(t, 'liveLocationKalman', m.liveLocationKalman)

      x = learner.kf.x
      sr = float(x[States.STEER_RATIO])
      st = float(x[States.STIFFNESS])
      ao_avg = math.degrees(x[States.ANGLE_OFFSET])
      ao = ao_avg + math.degrees(x[States.ANGLE_OFFSET_FAST])
      r = [sr, st, ao_avg, ao]
      if any(math.isnan(v) for v in r):
        print("NaN", t)

      ts.append(t)
      results.append(r)

    elif m.which() == 'liveParameters':
      t = m.logMonoTime / 1e9
      mm = m.liveParameters

      r = [mm.steerRatio, mm.stiffnessFactor, mm.angleOffsetAverageDeg, mm.angleOffsetDeg]
      if any(math.isnan(v) for v in r):
        print("NaN in log", t)
      ts_log.append(t)
      results_log.append(r)

  results = np.asarray(results)
  results_log = np.asarray(results_log)

  if PLOT:
    import matplotlib.pyplot as plt
    plt.figure()

    plt.subplot(3, 2, 1)
    plt.plot(ts, results[:, 0], label='Steer Ratio')
    plt.grid()
    plt.ylim([0, 20])
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(ts, results[:, 1], label='Stiffness')
    plt.ylim([0, 2])
    plt.grid()
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(ts, results[:, 2], label='Angle offset (average)')
    plt.plot(ts, results[:, 3], label='Angle offset (instant)')
    plt.ylim([-5, 5])
    plt.grid()
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(ts_log, results_log[:, 0], label='Steer Ratio')
    plt.grid()
    plt.ylim([0, 20])
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(ts_log, results_log[:, 1], label='Stiffness')
    plt.ylim([0, 2])
    plt.grid()
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.plot(ts_log, results_log[:, 2], label='Angle offset (average)')
    plt.plot(ts_log, results_log[:, 3], label='Angle offset (instant)')
    plt.ylim([-5, 5])
    plt.grid()
    plt.legend()
    plt.show()

