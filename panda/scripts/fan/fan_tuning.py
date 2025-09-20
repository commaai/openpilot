#!/usr/bin/env python3
import json
import time
import threading

from panda import Panda

def drain_serial(p):
  ret = []
  while True:
    d = p.serial_read(0)
    if len(d) == 0:
      break
    ret.append(d)
  return ret


fan_cmd = 0.

def logger(event):
  # requires a build with DEBUG_FAN
  with Panda(claim=False) as p, open('/tmp/fan_log', 'w') as f:
    power = None
    target_rpm = None
    rpm_fast = None
    t = time.monotonic()

    drain_serial(p)
    while not event.is_set():
      p.set_fan_power(fan_cmd)

      for l in drain_serial(p)[::-1]:
        ns = l.decode('utf8').strip().split(' ')
        if len(ns) == 4:
          target_rpm, rpm_fast, power = (int(n, 16) for n in ns)
          break

      dat = {
        't': time.monotonic() - t,
        'cmd_power': fan_cmd,
        'pwm_power': power,
        'target_rpm': target_rpm,
        'rpm_fast': rpm_fast,
        'rpm': p.get_fan_rpm(),
      }
      f.write(json.dumps(dat) + '\n')
      time.sleep(1/16.)
    p.set_fan_power(0)

def get_overshoot_rpm(p, power):
  global fan_cmd

  # make sure the fan is stopped completely
  fan_cmd = 0.
  while p.get_fan_rpm() > 100:
    time.sleep(0.1)
  time.sleep(3)

  # set it to 30% power to mimic going onroad
  fan_cmd = power
  max_rpm = 0
  max_power = 0
  for _ in range(70):
    max_rpm = max(max_rpm, p.get_fan_rpm())
    max_power = max(max_power, p.health()['fan_power'])
    time.sleep(0.1)

  # tolerate 10% overshoot
  expected_rpm = Panda.MAX_FAN_RPMs[bytes(p.get_type())] * power / 100
  overshoot = (max_rpm / expected_rpm) - 1

  return overshoot, max_rpm, max_power


if __name__ == "__main__":
  event = threading.Event()
  threading.Thread(target=logger, args=(event, )).start()

  try:
    p = Panda()
    for power in range(10, 101, 10):
      overshoot, max_rpm, max_power = get_overshoot_rpm(p, power)
      print(f"Fan power {power}%: overshoot {overshoot:.2%}, Max RPM {max_rpm}, Max power {max_power}%")
  finally:
    event.set()
