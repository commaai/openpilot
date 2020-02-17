#!/usr/bin/env python3
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

  parser = argparse.ArgumentParser(description='Sniff a communcation socket')
  parser.add_argument('--addr', default='127.0.0.1')
  args = parser.parse_args()

  if args.addr != "127.0.0.1":
    os.environ["ZMQ"] = "1"
    messaging.context = messaging.Context()

  carControl = messaging.sub_sock('carControl', addr=args.addr, conflate=True)
  sm = messaging.SubMaster(['carState', 'carControl', 'controlsState'], addr=args.addr)

  miss_cnt = 0
  speed = [0] * 10
  angle_actual = [0] * 10
  angle_desire = [0] * 10
  error = [0] * 10
  stats = defaultdict(lambda: {'err': 0, "cnt": 0, "=": 0, "+": 0, "-": 0})
  cnt = 0
  total_error = 0
  
  while messaging.recv_one(carControl):
    sm.update()
    miss_cnt += 1

    actual_speed = sm['carState'].vEgo
    enabled = sm['controlsState'].enabled
    steer_override = sm['controlsState'].steerOverride
    actual_angle = sm['controlsState'].angleSteers
    desired_angle = sm['carControl'].actuators.steerAngle
    angle_error = desired_angle - actual_angle

    if actual_speed > 10.0 and enabled and not steer_override:
      speed.append(actual_speed)
      speed.pop(0)
      angle_actual.append(actual_angle)
      angle_actual.pop(0)
      angle_desire.append(desired_angle)
      angle_desire.pop(0)
      error.append(angle_error)
      error.pop(0)
      cnt += 1

      if cnt >= 200 and cnt % 10 == 0:
        speed_avg = int(round(sum(speed) / len(speed), 0))
        actual_avg = int(round(sum(angle_actual) / len(angle_actual), 0))
        desire_avg = int(round(sum(angle_desire) / len(angle_desire), 0))
        error_avg = abs(round(sum(error) / len(error), 2))

        angle_abs = abs(actual_avg)
        stats[angle_abs]["err"] += error_avg
        stats[angle_abs]["cnt"] += 1
        if actual_avg == desire_avg:
          stats[angle_abs]["="] += 1
        else:
          stats[angle_abs]["+" if actual_avg > desire_avg else "-"] += 1
      if cnt > 0 and miss_cnt % 100 == 0:
        print(chr(27) + "[2J")
        for k in sorted(stats.keys()):
          v = stats[k]
          print(f'angle: {k:#2} | error: {round(v["err"] / v["cnt"], 2):2.2f} | =:{int(v["="] / v["cnt"] * 100):#3}% | +:{int(v["+"] / v["cnt"] * 100):#3}% | -:{int(v["-"] / v["cnt"] * 100):#3}% | count: {v["cnt"]:#4}')

    else:
      speed = [0] * 10
      angle_actual = [0] * 10
      angle_desire = [0] * 10
      error = [0] * 10
      cnt = 0
