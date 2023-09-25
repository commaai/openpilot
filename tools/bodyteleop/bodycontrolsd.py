#!/usr/bin/env python3
import json
import logging
import time
import argparse

from cereal import messaging
from openpilot.common.realtime import Ratekeeper

TIME_GAP_THRESHOLD = 0.5
last_control_send_time = time.monotonic()
logger = logging.getLogger("pc")
logging.basicConfig(level=logging.INFO)


def send_control_message(pm, x, y, source):
  global last_control_send_time
  msg = messaging.new_message('testJoystick')
  msg.testJoystick.axes = [x, y]
  msg.testJoystick.buttons = [False]
  pm.send('testJoystick', msg)
  logger.info(f"bodycontrol|{source} (x, y): ({x}, {y})")
  last_control_send_time = time.monotonic()


def main(remote_ip=None):
  rk = Ratekeeper(20.0)
  pm = messaging.PubMaster(['testJoystick'])
  sm = messaging.SubMaster(['bodyReserved0'])
  if remote_ip:
    sm_remote = messaging.SubMaster(['bodyReserved1'], addr=remote_ip)

  while True:
    sm.update(0)
    if remote_ip:
      sm_remote.update(0)

    if sm.updated['bodyReserved0']:
      controls = json.loads(sm['bodyReserved0'])
      send_control_message(pm, controls['x'], controls['y'], 'wasd')
    elif remote_ip and sm_remote.updated['bodyReserved1']:
      # ToDo: do something with the yolo outputs
      print(sm_remote['bodyReserved1'])
    else:
      now = time.monotonic()
      if now > last_control_send_time + TIME_GAP_THRESHOLD:
        send_control_message(pm, 0.0, 0.0, 'dummy')

    rk.keep_time()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Body control daemon")
  parser.add_argument("addr", help="remote ip address")
  args = parser.parse_args()

  main(args.addr)
