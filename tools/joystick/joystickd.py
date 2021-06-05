#!/usr/bin/env python
import argparse
import sys
import cereal.messaging as messaging

from common.numpy_fast import clip
from common.params import Params
from inputs import get_gamepad
from selfdrive.controls.lib.pid import apply_deadzone
from tools.lib.kbhit import KBHit

AXES = {'gb': ['w', 's', 'ABS_Y'], 'steer': ['a', 'd', 'ABS_X']}
BUTTONS = {'cancel': ['c', 'BTN_TRIGGER'], 'engaged_toggle': ['e', 'BTN_THUMB'], 'steer_required': ['t', 'BTN_TOP']}
AXES_INCREMENT = 0.05  # 5% of full actuation each key press
MAX_AXIS_VALUE = 255  # tune based on your joystick, 0 to this
kb = KBHit()


class Event:
  def __init__(self, _type=None, axis=None, button=None, value=0.):
    self.type = _type
    self.axis = axis
    self.button = button
    self.value = value


class Joystick:
  def __init__(self, use_keyboard=True):
    self.use_keyboard = use_keyboard
    self.axes_values = {ax: 0. for ax in AXES}
    self.btn_states = {btn: False for btn in BUTTONS}
    self.buttons = dict(BUTTONS, **{'reset': ['r', 'BTN_THUMB2']})  # adds reset option

  def get_event(self):
    event = Event()
    if self.use_keyboard:
      key = kb.getch().lower()
      if key in AXES['gb'] + AXES['steer']:
        event.type = 'axis'
        event.axis = 'gb' if key in AXES['gb'] else 'steer'
        if key in ['w', 'a']:  # these keys are positive
          event.value = self.axes_values[event.axis] + AXES_INCREMENT
        else:
          event.value = self.axes_values[event.axis] - AXES_INCREMENT

      elif len(btn := [_btn for _btn, keys in self.buttons.items() if key in keys]):
        event.type = 'button'
        event.button = btn[0]

    else:  # Joystick
      joystick_event = get_gamepad()[0]
      if joystick_event.code in AXES['gb'] + AXES['steer']:
        event.type = 'axis'
        event.axis = 'gb' if joystick_event.code in AXES['gb'] else 'steer'
        v = ((joystick_event.state / MAX_AXIS_VALUE) - 0.5) * 2  # normalize value from -1 to 1
        event.value = apply_deadzone(-v, AXES_INCREMENT) / (1 - AXES_INCREMENT)  # compensate for deadzone

      elif len(btn := [_btn for _btn, codes in self.buttons.items() if joystick_event.code in codes]) \
              and joystick_event.state == 0:  # only allow falling edge
        event.type = 'button'
        event.button = btn[0]

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
  Params().put_bool('JoystickDebugMode', True)
  joystick = Joystick(use_keyboard=use_keyboard)
  joystick_sock = messaging.pub_sock('testJoystick')

  if use_keyboard:
    print('\nGas/brake control: `W` and `S` keys')
    print('Steer control: `A` and `D` keys')
    print('Buttons:\n'
          '- `R`: Resets axes and buttons\n'
          '- `C`: Cancel cruise control\n'
          '- `E`: Toggle enabled\n'
          '- `T`: Steer required HUD')
  else:
    print('\nUsing joystick!')

  # Receive joystick/key events and send testJoystick msg
  while True:
    joystick.update()
    print('\n' + ', '.join([f'{name}: {v}' for name, v in joystick.axes_values.items()]))
    print(', '.join([f'{name}: {v}' for name, v in joystick.btn_states.items()]))

    dat = messaging.new_message('testJoystick')
    dat.testJoystick.axes = [joystick.axes_values[a] for a in AXES]
    dat.testJoystick.buttons = [joystick.btn_states[btn] for btn in BUTTONS]
    joystick_sock.send(dat.to_bytes())


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='Publishes joystick events from keyboard or joystick to control your car',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--keyboard', action='store_true', help='Use your keyboard over ssh to control joystickd')
  args = parser.parse_args(sys.argv[1:])

  joystick_thread(use_keyboard=args.keyboard)
