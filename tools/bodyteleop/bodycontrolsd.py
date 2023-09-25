import json
import time
from cereal import messaging
from openpilot.common.realtime import Ratekeeper

TIME_GAP_THRESHOLD = 0.5


def send_control_message(pm, last_control_send_time, x, y):
  msg = messaging.new_message('testJoystick')
  msg.testJoystick.axes = [x, y]
  msg.testJoystick.buttons = [False]
  pm.send('testJoystick', msg)
  last_control_send_time = time.monotonic()


def main():
  rk = Ratekeeper(20.0)
  pm = messaging.PubMaster(['testJoystick'])
  sm = messaging.SubMaster(['bodyReserved0'])
  last_control_send_time = time.monotonic()

  while True:
    now = time.monotonic()
    if now > last_control_send_time + TIME_GAP_THRESHOLD:
      send_control_message(pm, last_control_send_time, 0.0, 0.0)
    sm.update(0)
    if sm.updated['bodyReserved0']:
      controls = json.loads(sm['bodyReserved0'])
      send_control_message(pm, last_control_send_time, controls['x'], controls['y'])

    rk.keep_time()


if __name__ == "__main__":
  main()
