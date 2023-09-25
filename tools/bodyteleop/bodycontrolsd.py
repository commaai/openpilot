import json
import logging
import time
from cereal import messaging
from openpilot.common.realtime import Ratekeeper

TIME_GAP_THRESHOLD = 0.5
last_control_send_time = time.monotonic()
logger = logging.getLogger("pc")
logging.basicConfig(level=logging.INFO)

def send_control_message(pm, x, y):
  global last_control_send_time
  msg = messaging.new_message('testJoystick')
  msg.testJoystick.axes = [x, y]
  msg.testJoystick.buttons = [False]
  pm.send('testJoystick', msg)
  logger.info(f"bodycontrol (x, y): ({x}, {y})")
  last_control_send_time = time.monotonic()


def main():
  rk = Ratekeeper(20.0)
  pm = messaging.PubMaster(['testJoystick'])
  sm = messaging.SubMaster(['bodyReserved0'])

  while True:
    sm.update(0)
    if sm.updated['bodyReserved0']:
      controls = json.loads(sm['bodyReserved0'])
      send_control_message(pm, controls['x'], controls['y'])

    else:
      now = time.monotonic()
      if now > last_control_send_time + TIME_GAP_THRESHOLD:
        send_control_message(pm, 0.0, 0.0)

    rk.keep_time()


if __name__ == "__main__":
  main()
