#!/usr/bin/env python
import argparse

import cereal.messaging as messaging
from common.numpy_fast import interp, clip
from common.params import Params
from inputs import get_gamepad
from tools.lib.kbhit import KBHit

AXES_INCREMENT = 0.05  # 5% of full actuation each key press
MAX_AXIS_VALUE = 255  # tune based on your joystick, 0 to this


class Joystick:
  def __init__(self, use_keyboard=True):
    self.use_keyboard = use_keyboard
    if self.use_keyboard:
      self.kb = KBHit()
      self.buttons = {'c': 'cancel', 'e': 'engaged_toggle', 't': 'steer_required', 'r': 'reset'}
      self.axes = {'w': 'gb', 's': 'gb', 'a': 'steer', 'd': 'steer'}
    else:
      self.buttons = {'BTN_TRIGGER': 'cancel', 'BTN_THUMB': 'engaged_toggle', 'BTN_TOP': 'steer_required', 'BTN_THUMB2': 'reset'}
      self.axes = {'ABS_Y': 'gb', 'ABS_X': 'steer'}

    self.handle_button('reset')

  def handle_button(self, btn_name):
    if btn_name == 'reset':
      self.axes_values = {ax: 0. for ax in self.axes.values()}
      self.btn_states = {btn: False for btn in self.buttons.values() if btn != 'reset'}
    else:
      self.btn_states[btn_name] = not self.btn_states[btn_name]

  def update(self):
    # Get key or joystick event
    if self.use_keyboard:
      key = self.kb.getch().lower()
      if key in self.buttons:
        self.handle_button(self.buttons[key])
      elif key in self.axes:
        incr = AXES_INCREMENT if key in ['w', 'a'] else -AXES_INCREMENT  # these keys increment the axes positively
        self.axes_values[self.axes[key]] = clip(self.axes_values[self.axes[key]] + incr, -1, 1)

    else:  # Joystick
      joystick_event = get_gamepad()[0]
      if joystick_event.code in self.buttons and joystick_event.state == 0:  # state 0 is falling edge
        self.handle_button(self.buttons[joystick_event.code])
      elif joystick_event.code in self.axes:
        norm = interp(joystick_event.state, [0, MAX_AXIS_VALUE], [-1., 1.])
        self.axes_values[self.axes[joystick_event.code]] = norm if abs(norm) > 0.05 else 0.  # center can be noisy, deadzone of 5%

    dat = messaging.new_message('testJoystick')
    dat.testJoystick.axes = [self.axes_values[a] for a in self.axes_values]
    dat.testJoystick.buttons = [self.btn_states[btn] for btn in self.btn_states]
    return dat


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
          '- `E`: Toggle cruise state enabled\n'
          '- `T`: Steer required HUD')
  else:
    print('\nUsing joystick, don\'t forget to run unbridge on your device!')

  # Receive joystick/key events and send testJoystick msg
  while True:
    dat = joystick.update()
    joystick_sock.send(dat.to_bytes())
    print('\n' + ', '.join([f'{name}: {round(v, 3)}' for name, v in joystick.axes_values.items()]))
    print(', '.join([f'{name}: {v}' for name, v in joystick.btn_states.items()]))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='Publishes events from your joystick to control your car',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--keyboard', action='store_true', help='Use your keyboard instead of a joystick')
  args = parser.parse_args()

  joystick_thread(use_keyboard=args.keyboard)
