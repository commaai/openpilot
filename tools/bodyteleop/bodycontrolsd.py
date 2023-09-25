import json
import time
from cereal import messaging

TIME_GAP_THRESHOLD = 0.5

last_control_send_time = time.monotonic()


def send_control_message(x, y):
  msg = messaging.new_message('testJoystick')
  msg.testJoystick.axes = [x, y]
  msg.testJoystick.buttons = [False]
  pm.send('testJoystick', msg)
  last_control_send_time = time.monotonic()


def main():
  pm = messaging.PubMaster(['testJoystick'])
  sm = messaging.SubMaster(['bodyReserved0'])
  while True:
    now = time.monotonic()
    if now > last_control_send_time + TIME_GAP_THRESHOLD:
      send_control_message(0.0, 0.0)
    sm.update(0)
    if sm.updated['bodyReserved0']:
      controls = json.loads(sm['bodyReserved0'])
      send_control_message(controls['x'], controls['y'])


if __name__ == "__main__":
  main()
