#!/usr/bin/env python

# This process publishes joystick events. Such events can be subscribed by
# mocked car controller scripts.

import sys
import cereal.messaging as messaging

from dataclasses import dataclass

from common.numpy_fast import clip
from common.params import Params
from inputs import get_gamepad
from selfdrive.controls.lib.pid import apply_deadzone
from tools.lib.kbhit import KBHit

AXES = ['gb', 'steer']
BUTTONS = ['cancel', 'engaged_toggle', 'steer_required']
AXES_INCREMENT = 0.05  # 5% of full actuation each key press
POLL_RATE = int(1000 / 10.)  # 10 hz
kb = KBHit()


@dataclass
class Event:
  type = None
  axis = None
  button = None
  value = 0


class Joystick:
  def __init__(self, use_keyboard=True):
    self.use_keyboard = use_keyboard
    self.axes_values = {ax: 0. for ax in AXES}
    self.btn_states = {btn: False for btn in BUTTONS}

    self.buttons = {'r': 'reset', 'c': 'cancel', 'e': 'engaged_toggle', 't': 'steer_required'}
    if self.use_keyboard:
      self.axes = {'gb': ['w', 's'], 'steer': ['a', 'd']}  # first key is positive
    else:
      self.max_axis_value = 255  # tune based on your joystick, 0 to this
      self.axes = {'ABS_X': 'steer', 'ABS_Y': 'gb'}
      self.button_map = {'BTN_TRIGGER': 'cancel', 'BTN_THUMB': 'engaged_toggle',
                         'BTN_TOP': 'steer_required', 'BTN_THUMB2': 'reset'}

  def get_event(self):
    event = Event()
    if self.use_keyboard:
      key = kb.getch().lower()
      if key in self.axes['gb'] + self.axes['steer']:  # if axis event
        event.type = 'axis'
        event.axis = 'gb' if key in self.axes['gb'] else 'steer'
        if self.axes[event.axis].index(key) == 0:
          event.value = self.axes_values[event.axis] + AXES_INCREMENT
        else:
          event.value = self.axes_values[event.axis] - AXES_INCREMENT

      elif key in self.buttons:  # if button event
        event.type = 'button'
        event.button = self.buttons[key]

    else:  # Joystick
      joystick_event = get_gamepad()[0]
      if joystick_event.code in self.axes:
        event.type = 'axis'
        event.axis = self.axes[joystick_event.code]
        v = ((joystick_event.state / self.max_axis_value) - 0.5) * 2  # normalize value from -1 to 1
        event.value = apply_deadzone(-v, 0.05) / (1 - 0.05)  # compensate for deadzone

      # some buttons send events on rising and falling so only allow one state
      elif joystick_event.code in self.button_map and joystick_event.state == 0:
        event.type = 'button'
        event.button = self.button_map[joystick_event.code]

    return event  # returns empty Event if not mapped axis or button

  def update(self):
    event = self.get_event()
    if event.type == 'axis':
      self.axes_values[event.axis] = round(clip(event.value, -1., 1.), 3)
    elif event.type == 'button':
      if event.button == 'reset':
        self.axes_values = {ax: 0. for ax in AXES}
        self.btn_states = {btn: False for btn in BUTTONS}
      else:
        self.btn_states[event.button] = not self.btn_states[event.button]


def joystick_thread(use_keyboard):
  Params().put_bool("JoystickDebugMode", True)

  joystick = Joystick(use_keyboard=use_keyboard)
  joystick_sock = messaging.pub_sock('testJoystick')

  if use_keyboard:
    print('\nGas/brake control: `W` and `S` keys')
    print('Steer control: `A` and `D` keys')
    print('Buttons:\n'
          '- `R`: Resets axes values\n'
          '- `C`: Cancel cruise control\n'
          '- `E`: Toggle enabled\n'
          '- `T`: Steer required HUD')
  else:
    print('Using joystick!')

  # Receive joystick/key events and send testJoystick msg
  try:
    while 1:
      joystick.update()

      dat = messaging.new_message('testJoystick')
      dat.testJoystick.axes = [joystick.axes_values[a] for a in AXES]
      dat.testJoystick.buttons = [joystick.btn_states[btn] for btn in BUTTONS]
      joystick_sock.send(dat.to_bytes())

      print('\n' + ', '.join([f'{name}: {v}' for name, v in joystick.axes_values.items()]))
      print(', '.join([f'{name}: {v}' for name, v in joystick.btn_states.items()]))
  except KeyboardInterrupt:
    print('Interrupted, shutting down!')


if __name__ == "__main__":
  args = sys.argv[1:]
  use_keyboard = len(args) and args[0] == '--keyboard'
  joystick_thread(use_keyboard=use_keyboard)
