#!/usr/bin/env python3
# type: ignore

import os
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
  parser.add_argument('--addr', default='127.0.0.1')
  args = parser.parse_args()

  if args.addr != "127.0.0.1":
    os.environ["ZMQ"] = "1"
    messaging.context = messaging.Context()

  carControl = messaging.sub_sock('carControl', addr=args.addr, conflate=True)
  sm = messaging.SubMaster(['carState', 'carControl', 'controlsState'], addr=args.addr)

  msg_cnt = 0
  stats = defaultdict(lambda: {'err': 0, "cnt": 0, "=": 0, "+": 0, "-": 0})
  cnt = 0
  total_error = 0

  while messaging.recv_one(carControl):
    sm.update()
    msg_cnt += 1

    actual_speed = sm['carState'].vEgo
    enabled = sm['controlsState'].enabled
    steer_override = sm['controlsState'].steerOverride

    # must be above 10 m/s, engaged and not overriding steering
    if actual_speed > 10.0 and enabled and not steer_override:
      cnt += 1

      # wait 5 seconds after engage/override
      if cnt >= 500:
        # calculate error before rounding
        actual_angle = sm['controlsState'].angleSteers
        desired_angle = sm['carControl'].actuators.steeringAngleDeg
        angle_error = abs(desired_angle - actual_angle)

        # round numbers
        actual_angle = round(actual_angle, 1)
        desired_angle = round(desired_angle, 1)
        angle_error = round(angle_error, 2)
        angle_abs = int(abs(round(desired_angle, 0)))

        # collect stats
        stats[angle_abs]["err"] += angle_error
        stats[angle_abs]["cnt"] += 1
        if actual_angle == desired_angle:
          stats[angle_abs]["="] += 1
        else:
          if desired_angle == 0.:
            overshoot = True
          else:
            overshoot = desired_angle < actual_angle if desired_angle > 0. else desired_angle > actual_angle
          stats[angle_abs]["+" if overshoot else "-"] += 1
    else:
      cnt = 0

    if msg_cnt % 100 == 0:
      print(chr(27) + "[2J")
      if cnt != 0:
        print("COLLECTING ...")
      else:
        print("DISABLED (speed too low, not engaged, or steer override)")
      for k in sorted(stats.keys()):
        v = stats[k]
        print(f'angle: {k:#2} | error: {round(v["err"] / v["cnt"], 2):2.2f} | =:{int(v["="] / v["cnt"] * 100):#3}% | +:{int(v["+"] / v["cnt"] * 100):#4}% | -:{int(v["-"] / v["cnt"] * 100):#3}% | count: {v["cnt"]:#4}')
