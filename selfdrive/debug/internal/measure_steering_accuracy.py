#!/usr/bin/env python3
import os
import argparse
import signal
from collections import deque, defaultdict
from statistics import mean

import cereal.messaging as messaging

QUEUE_LEN = 10

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

  msg_cnt = 0
  speed = deque(maxlen=QUEUE_LEN)
  angle_actual = deque(maxlen=QUEUE_LEN)
  angle_desire = deque(maxlen=QUEUE_LEN)
  error = deque(maxlen=QUEUE_LEN)
  stats = defaultdict(lambda: {'err': 0, "cnt": 0, "=": 0, "+": 0, "-": 0})
  cnt = 0
  total_error = 0
  
  while messaging.recv_one(carControl):
    sm.update()
    msg_cnt += 1

    actual_speed = sm['carState'].vEgo
    enabled = sm['controlsState'].enabled
    steer_override = sm['controlsState'].steerOverride
    actual_angle = sm['controlsState'].angleSteers
    desired_angle = sm['carControl'].actuators.steerAngle
    angle_error = abs(desired_angle - actual_angle)

    if actual_speed > 10.0 and enabled and not steer_override:
      speed.append(actual_speed)
      angle_actual.append(actual_angle)
      angle_desire.append(desired_angle)
      error.append(angle_error)
      cnt += 1

      if cnt >= 500 and cnt % QUEUE_LEN == 0:
        speed_avg = round(mean(speed), 2)
        actual_avg = round(mean(angle_actual), 1)
        desire_avg = round(mean(angle_desire), 1)
        error_avg = round(mean(error), 2)
        angle_abs = int(abs(round(desire_avg, 0)))

        stats[angle_abs]["err"] += error_avg
        stats[angle_abs]["cnt"] += 1
        if actual_avg == desire_avg:
          stats[angle_abs]["="] += 1
        else:
          if desired_angle == 0.:
            overshoot = True
          else:
            overshoot = desire_avg < actual_avg if desire_avg > 0. else desire_avg > actual_avg
          stats[angle_abs]["+" if overshoot else "-"] += 1
      if cnt > 0 and msg_cnt % 100 == 0:
        print(chr(27) + "[2J")
        for k in sorted(stats.keys()):
          v = stats[k]
          print(f'angle: {k:#2} | error: {round(v["err"] / v["cnt"], 2):2.2f} | =:{int(v["="] / v["cnt"] * 100):#3}% | +:{int(v["+"] / v["cnt"] * 100):#3}% | -:{int(v["-"] / v["cnt"] * 100):#3}% | count: {v["cnt"]:#4}')

    else:
      speed.clear()
      angle_actual.clear()
      angle_desire.clear()
      error.clear()
      cnt = 0
