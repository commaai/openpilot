#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
from typing import NamedTuple
from openpilot.tools.lib.logreader import LogReader
from openpilot.selfdrive.locationd.models.pose_kf import EARTH_G

RLOG_MIN_LAT_ACTIVE = 50
RLOG_MIN_STEERING_UNPRESSED = 50
RLOG_MIN_REQUESTING_MAX = 80

QLOG_DECIMATION = 10


class Event(NamedTuple):
  lateral_accel: float
  speed: float
  roll: float


def find_events(lr: LogReader, qlog=False) -> list[Event]:
  if qlog:
    MIN_LAT_ACTIVE = RLOG_MIN_LAT_ACTIVE // QLOG_DECIMATION
    MIN_STEERING_UNPRESSED = RLOG_MIN_STEERING_UNPRESSED // QLOG_DECIMATION
    MIN_REQUESTING_MAX = RLOG_MIN_REQUESTING_MAX // QLOG_DECIMATION

  events = []

  start_ts = None
  platform = None
  route = None

  # state tracking
  steering_unpressed = 0  # frames
  requesting_max = 0  # frames
  lat_active = 0  # frames
  valid_segment = False

  # current state
  current_lateral_accel = None
  curvature = None
  v_ego = None
  roll = None

  for msg in lr:
    if msg.which() == 'carControl':
      if start_ts is None:
        start_ts = msg.logMonoTime

      lat_active = lat_active + 1 if msg.carControl.latActive else 0

    elif msg.which() == 'carOutput':
      # if we test with driver torque safety, max torque can be slightly noisy
      requesting_max = requesting_max + 1 if abs(msg.carOutput.actuatorsOutput.steer) > 0.95 else 0

    elif msg.which() == 'carState':
      steering_unpressed = steering_unpressed + 1 if not msg.carState.steeringPressed else 0
      v_ego = msg.carState.vEgo

    elif msg.which() == 'controlsState':
      curvature = msg.controlsState.curvature

    elif msg.which() == 'liveParameters':
      roll = msg.liveParameters.roll

    # if msg.which() == 'carControl':
    if lat_active > MIN_LAT_ACTIVE and steering_unpressed > MIN_STEERING_UNPRESSED and requesting_max > MIN_REQUESTING_MAX:
      lat_active = 0
      steering_unpressed = 0
      requesting_max = 0

      current_lateral_accel = curvature * v_ego ** 2 - roll * EARTH_G
      events.append(Event(current_lateral_accel, v_ego, roll))
      print('valid segment', (msg.logMonoTime - start_ts) * 1e-9, current_lateral_accel, curvature, v_ego, roll)

      valid_segment = True
      # current_lateral_accel = curvature * v_ego ** 2 - roll * EARTH_G
      # print('Max torque found:', steering_unpressed, requesting_max, lat_active)
      # print('state', current_lateral_accel, curvature, v_ego, roll)

    # else:
    #   if valid_segment:
    # current_lateral_accel = curvature * v_ego ** 2 - roll * EARTH_G
    # print('end of valid segment', (msg.logMonoTime - start_ts) * 1e-9, current_lateral_accel, curvature, v_ego, roll)
    # valid_segment = False

  return events


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Find max lateral acceleration events",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("route")
  args = parser.parse_args()

  lr = LogReader(args.route, sort_by_time=True)
  qlog = args.route.endswith('/q')
  if qlog:
    print('WARNING: Treating route as qlog!')

  events = find_events(lr, qlog=qlog)

  print()
  print(f'Found {len(events)} events')
  print('\n'.join(map(str, events)))

  CP = lr.first('carParams')

  plt.ion()
  plt.clf()
  plt.suptitle(f'{CP.carFingerprint} - Max lateral acceleration events')
  plt.title(args.route)
  plt.scatter([ev.speed for ev in events], [ev.lateral_accel for ev in events], label='max lateral accel events')
  plt.plot([0, 35], [3, 3], c='r', label='ISO 11270 - 3 m/s^2')
  plt.plot([0, 35], [-3, -3], c='r', label='ISO 11270 - 3 m/s^2')
  plt.xlim(0, 35)
  plt.ylim(-5, 5)
  plt.xlabel('speed (m/s)')
  plt.ylabel('lateral acceleration (m/s^2)')
  plt.show(block=True)
