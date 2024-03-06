#!/usr/bin/env python3
# type: ignore

import os
import time
import argparse
import signal
from collections import defaultdict

import cereal.messaging as messaging
from openpilot.tools.lib.logreader import LogReader

def sigint_handler(signal, frame):
  exit(0)
signal.signal(signal.SIGINT, sigint_handler)

class SteeringAccuracyTool:
  all_groups = {"germany":  (45, "45 - up m/s  //  162 - up km/h  //  101 - up mph"),
                "veryfast": (35, "35 - 45 m/s  //  126 - 162 km/h  //  78 - 101 mph"),
                "fast":     (25, "25 - 35 m/s  //  90 - 126 km/h  //  56 - 78 mph"),
                "medium":   (15, "15 - 25 m/s  //  54 - 90 km/h  //  34 - 56 mph"),
                "slow":     (5,  " 5 - 15 m/s  //  18 - 54 km/h  //  11 - 34 mph"),
                "crawl":    (0,  " 0 - 5 m/s  //  0 - 18 km/h  //  0 - 11 mph")}

  def __init__(self, args):
    self.msg_cnt = 0
    self.cnt = 0
    self.total_error = 0

    if args.group == "all":
      self.display_groups = self.all_groups.keys()
    elif args.group in self.all_groups.keys():
      self.display_groups = [args.group]
    else:
      raise ValueError("invalid speed group, see help")

    self.speed_group_stats = {}
    for group in self.all_groups:
      self.speed_group_stats[group] = defaultdict(lambda: {'err': 0, "cnt": 0, "=": 0, "+": 0, "-": 0, "steer": 0, "limited": 0, "saturated": 0, "dpp": 0})

  def update(self, sm):
    self.msg_cnt += 1

    lateralControlState = sm['controlsState'].lateralControlState
    control_type = list(lateralControlState.to_dict().keys())[0]
    control_state = lateralControlState.__getattr__(control_type)

    v_ego = sm['carState'].vEgo
    active = sm['controlsState'].active
    steer = sm['carOutput'].actuatorsOutput.steer
    standstill = sm['carState'].standstill
    steer_limited = abs(sm['carControl'].actuators.steer - sm['carControl'].actuatorsOutput.steer) > 1e-2
    overriding = sm['carState'].steeringPressed
    changing_lanes = sm['modelV2'].meta.laneChangeState != 0
    model_points = sm['modelV2'].position.y
    # must be engaged, not at standstill, not overriding steering, and not changing lanes
    if active and not standstill and not overriding and not changing_lanes:
      self.cnt += 1

      # wait 5 seconds after engage / standstill / override / lane change
      if self.cnt >= 500:
        actual_angle = control_state.steeringAngleDeg
        desired_angle = control_state.steeringAngleDesiredDeg

        # calculate error before rounding, then round for stats grouping
        angle_error = abs(desired_angle - actual_angle)
        actual_angle = round(actual_angle, 1)
        desired_angle = round(desired_angle, 1)
        angle_error = round(angle_error, 2)
        angle_abs = int(abs(round(desired_angle, 0)))

        for group, group_props in self.all_groups.items():
          if v_ego > group_props[0]:
            # collect stats
            self.speed_group_stats[group][angle_abs]["cnt"] += 1
            self.speed_group_stats[group][angle_abs]["err"] += angle_error
            self.speed_group_stats[group][angle_abs]["steer"] += abs(steer)
            if len(model_points):
              self.speed_group_stats[group][angle_abs]["dpp"] += abs(model_points[0])
            if steer_limited:
              self.speed_group_stats[group][angle_abs]["limited"] += 1
            if control_state.saturated:
              self.speed_group_stats[group][angle_abs]["saturated"] += 1
            if actual_angle == desired_angle:
              self.speed_group_stats[group][angle_abs]["="] += 1
            else:
              if desired_angle == 0.:
                overshoot = True
              else:
                overshoot = desired_angle < actual_angle if desired_angle > 0. else desired_angle > actual_angle
              self.speed_group_stats[group][angle_abs]["+" if overshoot else "-"] += 1
            break
    else:
      self.cnt = 0

    if self.msg_cnt % 100 == 0:
      print(chr(27) + "[2J")
      if self.cnt != 0:
        print("COLLECTING ...\n")
      else:
        print("DISABLED (not active, standstill, steering override, or lane change)\n")
      for group in self.display_groups:
        if len(self.speed_group_stats[group]) > 0:
          print(f"speed group: {group:10s} {self.all_groups[group][1]:>96s}")
          print(f"  {'-'*118}")
          for k in sorted(self.speed_group_stats[group].keys()):
            v = self.speed_group_stats[group][k]
            print(f'  {k:#2}° | actuator:{int(v["steer"] / v["cnt"] * 100):#3}% ' +
                  f'| error: {round(v["err"] / v["cnt"], 2):2.2f}° | -:{int(v["-"] / v["cnt"] * 100):#3}% ' +
                  f'| =:{int(v["="] / v["cnt"] * 100):#3}% | +:{int(v["+"] / v["cnt"] * 100):#3}% | lim:{v["limited"]:#5} ' +
                  f'| sat:{v["saturated"]:#5} | path dev: {round(v["dpp"] / v["cnt"], 2):2.2f}m | total: {v["cnt"]:#5}')
          print("")


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Steering accuracy measurement tool')
  parser.add_argument('--route', help="route name")
  parser.add_argument('--addr', default='127.0.0.1', help="IP address for optional ZMQ listener, default to msgq")
  parser.add_argument('--group', default='all', help="speed group to display, [crawl|slow|medium|fast|veryfast|germany|all], default to all")
  parser.add_argument('--cache', default=False, action='store_true', help="use cached data, default to False")
  args = parser.parse_args()

  if args.cache:
    os.environ['FILEREADER_CACHE'] = '1'

  tool = SteeringAccuracyTool(args)

  if args.route is not None:
    print(f"loading {args.route}...")
    lr = LogReader(args.route, sort_by_time=True)

    sm = {}
    for msg in lr:
      if msg.which() == 'carState':
        sm['carState'] = msg.carState
      elif msg.which() == 'carControl':
        sm['carControl'] = msg.carControl
      elif msg.which() == 'controlsState':
        sm['controlsState'] = msg.controlsState
      elif msg.which() == 'modelV2':
        sm['modelV2'] = msg.modelV2

      if msg.which() == 'carControl' and 'carState' in sm and 'controlsState' in sm and 'modelV2' in sm:
        tool.update(sm)

  else:
    if args.addr != "127.0.0.1":
      os.environ["ZMQ"] = "1"
      messaging.context = messaging.Context()

    carControl = messaging.sub_sock('carControl', addr=args.addr, conflate=True)
    sm = messaging.SubMaster(['carState', 'carControl', 'carOutput', 'controlsState', 'modelV2'], addr=args.addr)
    time.sleep(1)  # Make sure all submaster data is available before going further

    print("waiting for messages...")
    while messaging.recv_one(carControl):
      sm.update()
      tool.update(sm)
