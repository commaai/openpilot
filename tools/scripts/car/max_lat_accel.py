#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm
from typing import NamedTuple
from openpilot.tools.lib.logreader import LogReader
from openpilot.selfdrive.locationd.models.pose_kf import EARTH_G

RLOG_MIN_LAT_ACTIVE = 50
RLOG_MIN_STEERING_UNPRESSED = 50
RLOG_MIN_REQUESTING_MAX = 25  # sample many times after reaching max torque

QLOG_DECIMATION = 10


class Event(NamedTuple):
  lateral_accel: float
  speed: float
  roll: float
  timestamp: float  # relative to start of route (s)


def find_events(lr: LogReader, extrapolate: bool = False, qlog: bool = False) -> list[Event]:
  min_lat_active = RLOG_MIN_LAT_ACTIVE // QLOG_DECIMATION if qlog else RLOG_MIN_LAT_ACTIVE
  min_steering_unpressed = RLOG_MIN_STEERING_UNPRESSED // QLOG_DECIMATION if qlog else RLOG_MIN_STEERING_UNPRESSED
  min_requesting_max = RLOG_MIN_REQUESTING_MAX // QLOG_DECIMATION if qlog else RLOG_MIN_REQUESTING_MAX

  # if we test with driver torque safety, max torque can be slightly noisy
  steer_threshold = 0.7 if extrapolate else 0.95

  events = []

  # state tracking
  steering_unpressed = 0  # frames
  requesting_max = 0  # frames
  lat_active = 0  # frames

  # current state
  curvature = 0
  v_ego = 0
  roll = 0
  out_torque = 0

  start_ts = 0
  for msg in lr:
    if msg.which() == 'carControl':
      if start_ts == 0:
        start_ts = msg.logMonoTime

      lat_active = lat_active + 1 if msg.carControl.latActive else 0

    elif msg.which() == 'carOutput':
      out_torque = msg.carOutput.actuatorsOutput.torque
      requesting_max = requesting_max + 1 if abs(out_torque) > steer_threshold else 0

    elif msg.which() == 'carState':
      steering_unpressed = steering_unpressed + 1 if not msg.carState.steeringPressed else 0
      v_ego = msg.carState.vEgo

    elif msg.which() == 'controlsState':
      curvature = msg.controlsState.curvature

    elif msg.which() == 'liveParameters':
      roll = msg.liveParameters.roll

    if lat_active > min_lat_active and steering_unpressed > min_steering_unpressed and requesting_max > min_requesting_max:
      # TODO: record max lat accel at the end of the event, need to use the past lat accel as overriding can happen before we detect it
      requesting_max = 0

      factor = 1 / abs(out_torque)
      current_lateral_accel = (curvature * v_ego ** 2 * factor) - roll * EARTH_G
      events.append(Event(current_lateral_accel, v_ego, roll, round((msg.logMonoTime - start_ts) * 1e-9, 2)))
      print(events[-1])

  return events


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Find max lateral acceleration events",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("route", nargs='+')
  parser.add_argument("-e", "--extrapolate", action="store_true", help="Extrapolates max lateral acceleration events linearly. " +
                                                                       "This option can be far less accurate.")
  args = parser.parse_args()

  events = []
  for route in tqdm(args.route):
    try:
      lr = LogReader(route, sort_by_time=True)
    except Exception:
      print(f'Skipping {route}')
      continue

    qlog = route.endswith('/q')
    if qlog:
      print('WARNING: Treating route as qlog!')

    print('Finding events...')
    events += lr.run_across_segments(8, partial(find_events, extrapolate=args.extrapolate, qlog=qlog), disable_tqdm=True)

  print()
  print(f'Found {len(events)} events')

  perc_left_accel = -np.percentile([-ev.lateral_accel for ev in events if ev.lateral_accel < 0] or [0], 90)
  perc_right_accel = np.percentile([ev.lateral_accel for ev in events if ev.lateral_accel > 0] or [0], 90)

  CP = lr.first('carParams')

  plt.ion()
  plt.clf()
  plt.suptitle(f'{CP.carFingerprint} - Max lateral acceleration events')
  plt.title(', '.join(args.route))
  plt.scatter([ev.speed for ev in events], [ev.lateral_accel for ev in events], label='max lateral accel events')

  plt.plot([0, 35], [3, 3], c='r', label='ISO 11270 - 3 m/s^2')
  plt.plot([0, 35], [-3, -3], c='r')

  plt.plot([0, 35], [perc_left_accel, perc_left_accel], c='g', linestyle='--', label='90th percentile left lateral accel')
  plt.plot([0, 35], [perc_right_accel, perc_right_accel], c='#ff7f0e', linestyle='--', label='90th percentile right lateral accel')
  plt.text(0.4, float(perc_left_accel + 0.4), f'{perc_left_accel:.2f} m/s^2', verticalalignment='center', fontsize=12)
  plt.text(0.4, float(perc_right_accel - 0.4), f'{perc_right_accel:.2f} m/s^2', verticalalignment='center', fontsize=12)

  plt.xlim(0, 35)
  plt.ylim(-5, 5)
  plt.xlabel('speed (m/s)')
  plt.ylabel('lateral acceleration (m/s^2)')
  plt.legend()
  plt.show(block=True)
