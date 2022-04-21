#!/usr/bin/env python
import os
import argparse
import threading
from inputs import get_gamepad

import cereal.messaging as messaging
from common.realtime import Ratekeeper
from common.numpy_fast import interp, clip
from common.params import Params
from tools.lib.kbhit import KBHit


class Keyboard:
  def __init__(self):
    self.kb = KBHit()
    self.axis_increment = 0.05  # 5% of full actuation each key press
    self.axes_map = {'w': 'gb', 's': 'gb',
                     'a': 'steer', 'd': 'steer'}
    self.axes_values = {'gb': 0., 'steer': 0.}
    self.axes_order = ['gb', 'steer']
    self.cancel = False

  def update(self):
    key = self.kb.getch().lower()
    self.cancel = False
    if key == 'r':
      self.axes_values = {ax: 0. for ax in self.axes_values}
    elif key == 'c':
      self.cancel = True
    elif key in self.axes_map:
      axis = self.axes_map[key]
      incr = self.axis_increment if key in ['w', 'a'] else -self.axis_increment
      self.axes_values[axis] = clip(self.axes_values[axis] + incr, -1, 1)
    else:
      return False
    return True


class Joystick:
  def __init__(self):
    # TODO: find a way to get this from API, perhaps "inputs" doesn't support it
    self.min_axis_value = {'ABS_Y': 0., 'ABS_RZ': 0.}
    self.max_axis_value = {'ABS_Y': 1023., 'ABS_RZ': 255.}
    self.cancel_button = 'BTN_TRIGGER'
    self.axes_values = {'ABS_Y': 0., 'ABS_RZ': 0.}  # gb, steer
    self.axes_order = ['ABS_Y', 'ABS_RZ']
    self.cancel = False

  def update(self):
    joystick_event = get_gamepad()[0]
    event = (joystick_event.code, joystick_event.state)
    if event[0] == self.cancel_button:
      if event[1] == 1:
        self.cancel = True
      elif event[1] == 0:   # state 0 is falling edge
        self.cancel = False
    elif event[0] in self.axes_values:
      self.max_axis_value[event[0]] = max(event[1], self.max_axis_value[event[0]])
      self.min_axis_value[event[0]] = min(event[1], self.min_axis_value[event[0]])

      norm = -interp(event[1], [self.min_axis_value[event[0]], self.max_axis_value[event[0]]], [-1., 1.])
      self.axes_values[event[0]] = norm if abs(norm) > 0.05 else 0.  # center can be noisy, deadzone of 5%
    else:
      return False
    return True


def send_thread(joystick):
  joystick_sock = messaging.pub_sock('testJoystick')
  rk = Ratekeeper(100, print_delay_threshold=None)
  while 1:
    dat = messaging.new_message('testJoystick')
    dat.testJoystick.axes = [joystick.axes_values[a] for a in joystick.axes_order]
    dat.testJoystick.buttons = [joystick.cancel]
    joystick_sock.send(dat.to_bytes())
    print('\n' + ', '.join(f'{name}: {round(v, 3)}' for name, v in joystick.axes_values.items()))
    if "WEB" in os.environ:
      import requests
      requests.get("http://"+os.environ["WEB"]+":5000/control/%f/%f" % tuple([joystick.axes_values[a] for a in joystick.axes_order][::-1]))
    rk.keep_time()

def joystick_thread(use_keyboard):
  Params().put_bool('JoystickDebugMode', True)
  joystick = Keyboard() if use_keyboard else Joystick()
  threading.Thread(target=send_thread, args=(joystick,), daemon=True).start()
  while True:
    joystick.update()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Publishes events from your joystick to control your car.\n' +
                                               'openpilot must be offroad before starting joysticked.',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--keyboard', action='store_true', help='Use your keyboard instead of a joystick')
  args = parser.parse_args()

  if not Params().get_bool("IsOffroad") and "ZMQ" not in os.environ and "WEB" not in os.environ:
    print("The car must be off before running joystickd.")
    exit()

  print()
  if args.keyboard:
    print('Gas/brake control: `W` and `S` keys')
    print('Steering control: `A` and `D` keys')
    print('Buttons')
    print('- `R`: Resets axes')
    print('- `C`: Cancel cruise control')
  else:
    print('Using joystick, make sure to run cereal/messaging/bridge on your device if running over the network!')

  joystick_thread(args.keyboard)
