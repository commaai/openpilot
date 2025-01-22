#!/usr/bin/env python3
import time
import threading
import argparse
import numpy as np
from pprint import pprint
from inputs import get_gamepad

from kbhit import KBHit

from opendbc.car.structs import CarControl
from opendbc.car.panda_runner import PandaRunner

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
    print(key)
    self.cancel = False
    if key == 'r':
      self.axes_values = {ax: 0. for ax in self.axes_values}
    elif key == 'c':
      self.cancel = True
    elif key in self.axes_map:
      axis = self.axes_map[key]
      incr = self.axis_increment if key in ['w', 'a'] else -self.axis_increment
      self.axes_values[axis] = float(np.clip(self.axes_values[axis] + incr, -1, 1))
    else:
      return False
    return True

class Joystick:
  def __init__(self, gamepad=False):
    # TODO: find a way to get this from API, perhaps "inputs" doesn't support it
    if gamepad:
      self.cancel_button = 'BTN_NORTH'  # (BTN_NORTH=X, ABS_RZ=Right Trigger)
      accel_axis = 'ABS_Y'
      steer_axis = 'ABS_RX'
    else:
      self.cancel_button = 'BTN_TRIGGER'
      accel_axis = 'ABS_Y'
      steer_axis = 'ABS_RX'
    self.min_axis_value = {accel_axis: 0., steer_axis: 0.}
    self.max_axis_value = {accel_axis: 255., steer_axis: 255.}
    self.axes_values = {accel_axis: 0., steer_axis: 0.}
    self.axes_order = [accel_axis, steer_axis]
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

      norm = -float(np.interp(event[1], [self.min_axis_value[event[0]], self.max_axis_value[event[0]]], [-1., 1.]))
      self.axes_values[event[0]] = norm if abs(norm) > 0.05 else 0.  # center can be noisy, deadzone of 5%
    else:
      return False
    return True

def joystick_thread(joystick):
  while True:
    joystick.update()

def main(joystick):
  threading.Thread(target=joystick_thread, args=(joystick,), daemon=True).start()
  with PandaRunner() as p:
    CC = CarControl(enabled=False)
    while True:
      CC.actuators.accel = float(4.0*np.clip(joystick.axes_values['gb'], -1, 1))
      CC.actuators.steer = float(np.clip(joystick.axes_values['steer'], -1, 1))
      pprint(CC)

      p.read()
      p.write(CC)

      # 100Hz
      time.sleep(0.01)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Test the car interface with a joystick. Uses keyboard by default.',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--mode', choices=['keyboard', 'gamepad', 'joystick'], default='keyboard')
  args = parser.parse_args()

  print()
  joystick: Keyboard | Joystick
  if args.mode == 'keyboard':
    print('Gas/brake control: `W` and `S` keys')
    print('Steering control: `A` and `D` keys')
    print('Buttons')
    print('- `R`: Resets axes')
    print('- `C`: Cancel cruise control')
    joystick = Keyboard()
  else:
    joystick = Joystick(gamepad=(args.mode == 'gamepad'))
  main(joystick)