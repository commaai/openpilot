#!/usr/bin/env python
import argparse

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


class Joystick:
  def __init__(self, use_keyboard=True):
    self.use_keyboard = use_keyboard
    self.axes_values = {ax: 0. for ax in AXES}
    self.btn_states = {btn: False for btn in BUTTONS}
    self.buttons = dict(BUTTONS, **{'reset': ['r', 'BTN_THUMB2']})  # adds reset option
    self.kb = KBHit()

  def update(self):
    if self.use_keyboard:
      key = self.kb.getch().lower()
      state = 0
    else:
      joystick_event = get_gamepad()[0]
      key = joystick_event.code
      state = joystick_event.state

    # Button event
    btn = [_btn for _btn in self.buttons if key in self.buttons[_btn]]
    if len(btn) and state == 0:  # only allow falling edge
      if (btn := btn[0]) == 'reset':
        self.axes_values = {ax: 0. for ax in AXES}
        self.btn_states = {btn: False for btn in BUTTONS}
      else:
        self.btn_states[btn] = not self.btn_states[btn]

    # Axis event
    elif key in AXES['gb'] + AXES['steer']:
      axis = 'gb' if key in AXES['gb'] else 'steer'
      if self.use_keyboard:
        value = self.axes_values[axis] + (AXES_INCREMENT if key in ['w', 'a'] else -AXES_INCREMENT)  # these keys are positive
      else:
        norm = ((state / MAX_AXIS_VALUE) - 0.5) * 2
        value = apply_deadzone(-norm, AXES_INCREMENT) / (1 - AXES_INCREMENT)  # center is noisy

      self.axes_values[axis] = round(clip(value, -1., 1.), 3)


def joystick_thread(use_keyboard):
  Params().put_bool('JoystickDebugMode', True)
  joystick_sock = messaging.pub_sock('testJoystick')
  joystick = Joystick(use_keyboard=use_keyboard)

  if use_keyboard:
    print('\nGas/brake control: `W` and `S` keys')
    print('Steer control: `A` and `D` keys')
    print('Buttons:\n'
          '- `R`: Resets axes and buttons\n'
          '- `C`: Cancel cruise control\n'
          '- `E`: Toggle enabled\n'
          '- `T`: Steer required HUD')
  else:
    print('\nUsing joystick, don\'t forget to run unbridge on your device!')

  # Receive joystick/key events and send testJoystick msg
  while True:
    joystick.update()
    dat = messaging.new_message('testJoystick')
    dat.testJoystick.axes = [joystick.axes_values[a] for a in AXES]
    dat.testJoystick.buttons = [joystick.btn_states[btn] for btn in BUTTONS]
    joystick_sock.send(dat.to_bytes())

    print('\n' + ', '.join([f'{name}: {v}' for name, v in joystick.axes_values.items()]))
    print(', '.join([f'{name}: {v}' for name, v in joystick.btn_states.items()]))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='Publishes events from your joystick to control your car',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--keyboard', action='store_true', help='Use your keyboard over ssh instead of a joystick')
  args = parser.parse_args()

  joystick_thread(use_keyboard=args.keyboard)
