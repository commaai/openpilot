#!/usr/bin/env python3
# type: ignore

import os
import time
import argparse
import signal
from collections import defaultdict

import cereal.messaging as messaging

def sigint_handler(signal, frame):
  print("handler!")
  exit(0)
signal.signal(signal.SIGINT, sigint_handler)

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Sniff a communication socket')
  parser.add_argument('control_type', help="[pid|indi|lqr|angle]")
  parser.add_argument('--addr', default='127.0.0.1', help="IP address for optional ZMQ listener, default to msgq")
  parser.add_argument('--group', default='all', help="speed group to display, [crawl|slow|medium|fast|veryfast|germany|all], default to all")
  args = parser.parse_args()

  if args.addr != "127.0.0.1":
    os.environ["ZMQ"] = "1"
    messaging.context = messaging.Context()

  all_groups = {"germany":  (45, "45 - up m/s  //  162 - up km/h  //  101 - up mph"),
                "veryfast": (35, "35 - 45 m/s  //  126 - 162 km/h  //  78 - 101 mph"),
                "fast":     (25, "25 - 35 m/s  //  90 - 126 km/h  //  56 - 78 mph"),
                "medium":   (15, "15 - 25 m/s  //  54 - 90 km/h  //  34 - 56 mph"),
                "slow":     (5,  " 5 - 15 m/s  //  18 - 54 km/h  //  11 - 34 mph"),
                "crawl":    (0,  " 0 - 5 m/s  //  0 - 18 km/h  //  0 - 11 mph")}

  if args.group == "all":
    display_groups = all_groups.keys()
  elif args.group in all_groups.keys():
    display_groups = [args.group]
  else:
    raise ValueError("invalid speed group, see help")

  speed_group_stats = {}
  for group in all_groups:
    speed_group_stats[group] = defaultdict(lambda: {'err': 0, "cnt": 0, "=": 0, "+": 0, "-": 0, "steer": 0, "limited": 0, "saturated": 0, "dpp": 0})

  carControl = messaging.sub_sock('carControl', addr=args.addr, conflate=True)
  sm = messaging.SubMaster(['carState', 'carControl', 'controlsState', 'lateralPlan'], addr=args.addr)
  time.sleep(1)  # Make sure all submaster data is available before going further

  msg_cnt = 0
  cnt = 0
  total_error = 0

  while messaging.recv_one(carControl):
    sm.update()
    msg_cnt += 1

    if args.control_type == "pid":
      control_state = sm['controlsState'].lateralControlState.pidState
    elif args.control_type == "indi":
      control_state = sm['controlsState'].lateralControlState.indiState
    elif args.control_type == "lqr":
      control_state = sm['controlsState'].lateralControlState.lqrState
    elif args.control_type == "angle":
      control_state = sm['controlsState'].lateralControlState.angleState
    else:
      raise ValueError("invalid lateral control type, see help")

    v_ego = sm['carState'].vEgo
    active = sm['controlsState'].active
    steer = sm['carControl'].actuatorsOutput.steer
    standstill = sm['carState'].standstill
    steer_limited = sm['carState'].steeringRateLimited
    overriding = sm['carState'].steeringPressed
    changing_lanes = sm['lateralPlan'].laneChangeState != 0
    d_path_points = sm['lateralPlan'].dPathPoints
    # must be engaged, not at standstill, not overriding steering, and not changing lanes
    if active and not standstill and not overriding and not changing_lanes:
      cnt += 1

      # wait 5 seconds after engage / standstill / override / lane change
      if cnt >= 500:
        actual_angle = control_state.steeringAngleDeg
        desired_angle = control_state.steeringAngleDesiredDeg

        # calculate error before rounding, then round for stats grouping
        angle_error = abs(desired_angle - actual_angle)
        actual_angle = round(actual_angle, 1)
        desired_angle = round(desired_angle, 1)
        angle_error = round(angle_error, 2)
        angle_abs = int(abs(round(desired_angle, 0)))

        for group, group_props in all_groups.items():
          if v_ego > group_props[0]:
            # collect stats
            speed_group_stats[group][angle_abs]["cnt"] += 1
            speed_group_stats[group][angle_abs]["err"] += angle_error
            speed_group_stats[group][angle_abs]["steer"] += abs(steer)
            if len(d_path_points):
              speed_group_stats[group][angle_abs]["dpp"] += abs(d_path_points[0])
            if steer_limited:
              speed_group_stats[group][angle_abs]["limited"] += 1
            if control_state.saturated:
              speed_group_stats[group][angle_abs]["saturated"] += 1
            if actual_angle == desired_angle:
              speed_group_stats[group][angle_abs]["="] += 1
            else:
              if desired_angle == 0.:
                overshoot = True
              else:
                overshoot = desired_angle < actual_angle if desired_angle > 0. else desired_angle > actual_angle
              speed_group_stats[group][angle_abs]["+" if overshoot else "-"] += 1
            break
    else:
      cnt = 0

    if msg_cnt % 100 == 0:
      print(chr(27) + "[2J")
      if cnt != 0:
        print("COLLECTING ...\n")
      else:
        print("DISABLED (not active, standstill, steering override, or lane change)\n")
      for group in display_groups:
        if len(speed_group_stats[group]) > 0:
          print(f"speed group: {group:10s} {all_groups[group][1]:>96s}")
          print(f"  {'-'*118}")
          for k in sorted(speed_group_stats[group].keys()):
            v = speed_group_stats[group][k]
            print(f'  {k:#2}° | actuator:{int(v["steer"] / v["cnt"] * 100):#3}% | error: {round(v["err"] / v["cnt"], 2):2.2f}° | -:{int(v["-"] / v["cnt"] * 100):#3}% | =:{int(v["="] / v["cnt"] * 100):#3}% | +:{int(v["+"] / v["cnt"] * 100):#3}% | lim:{v["limited"]:#5} | sat:{v["saturated"]:#5} | path dev: {round(v["dpp"] / v["cnt"], 2):2.2f}m | total: {v["cnt"]:#5}')
          print("")
