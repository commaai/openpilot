#!/usr/bin/env python

# This process publishes joystick events. Such events can be subscribed by
# mocked car controller scripts.

from common.params import Params
import zmq
import cereal.messaging as messaging
from common.numpy_fast import clip
from tools.lib.kbhit import KBHit
import multiprocessing
from uuid import uuid4


class Joystick:  # TODO: see if we can clean this class up
  def __init__(self, use_keyboard=True):
    self.kb = KBHit()

    self.use_keyboard = use_keyboard
    self.axes_values = {ax: 0. for ax in AXES}
    self.btn_states = {btn: False for btn in BUTTONS}

    if self.use_keyboard:
      self.axes = {'gb': ['w', 's'], 'steer': ['a', 'd']}  # first key is positive
      self.buttons = {'r': 'reset', 'c': BUTTONS[0], 'e': BUTTONS[1], 't': BUTTONS[2]}
    else:
      raise NotImplementedError("Only keyboard is supported for now")  # TODO: support joystick

  def update(self):
    # self.btn_states = {btn: False for btn in BUTTONS}
    if self.use_keyboard:
      key = self.kb.getch().lower()
      if key in self.axes['gb'] + self.axes['steer']:  # if axis event
        control_type = 'gb' if key in self.axes['gb'] else 'steer'
        if self.axes[control_type].index(key) == 0:
          v = self.axes_values[control_type] + AXES_INCREMENT
        else:
          v = self.axes_values[control_type] - AXES_INCREMENT
        self.axes_values[control_type] = round(clip(v, -1., 1.), 3)

      elif key in self.buttons:
        if self.buttons[key] == 'reset':
          self.axes_values = {ax: 0. for ax in AXES}
        else:
          btn = self.buttons[key]
          self.btn_states[btn] = not self.btn_states[btn]

      else:
        print('Key not assigned to an action!')
      return


AXES = ['gb', 'steer']
BUTTONS = ['cancel', 'engaged_toggle', 'steer_required']
AXES_INCREMENT = 0.05  # 5% of full actuation each key press
POLL_RATE = int(1000 / 10.)  # 10 hz


def send_thread(command_address, joystick):
  zmq.Context._instance = None
  context = zmq.Context.instance()

  command_sock = context.socket(zmq.PULL)
  command_sock.bind(command_address)

  poller = zmq.Poller()
  poller.register(command_sock, zmq.POLLIN)
  joystick_sock = messaging.pub_sock('testJoystick')

  # starting message to send to controlsd if user doesn't type any keys
  msg = joystick
  while True:
    for sock in dict(poller.poll(POLL_RATE)):
      msg = sock.recv_pyobj()  # TODO: only receives axes for now

    dat = messaging.new_message('testJoystick')
    dat.testJoystick.axes = [msg.axes_values[a] for a in ['gb', 'steer']]
    dat.testJoystick.buttons = [msg.btn_states[btn] for btn in BUTTONS]

    joystick_sock.send(dat.to_bytes())
    # print(f'Sent: {dat}')


def joystick_thread():
  use_keyboard = True
  Params().put_bool("JoystickDebugMode", True)
  joystick = Joystick(use_keyboard=use_keyboard)
  command_address = "ipc:///tmp/{}".format(uuid4())

  command_sock = zmq.Context.instance().socket(zmq.PUSH)
  command_sock.connect(command_address)

  send_thread_proc = multiprocessing.Process(target=send_thread, args=(command_address, joystick))
  send_thread_proc.start()

  if use_keyboard:
    print('\nGas/brake control: `W` and `Sa` keys')
    print('Steer control: `A` and `D` keys')
    print('Buttons:\n'
          '- `R`: Resets axes values\n'
          '- `C`: Cancel cruise control\n'
          '- `E`: Toggle enabled\n'
          '- `T`: Steer required HUD')

  # Receive joystick/key events and send to joystick send thread
  try:
    while 1:
      joystick.update()
      print()
      print(', '.join([f'{name}: {v}' for name, v in joystick.axes_values.items()]))
      print(', '.join([f'{name}: {v}' for name, v in joystick.btn_states.items()]))
      command_sock.send_pyobj(joystick)
      # TODO: time shouldn't matter since joystick.update() is blocking
  except KeyboardInterrupt:
    print('Interrupted, shutting down!')
    send_thread_proc.terminate()


if __name__ == "__main__":
  joystick_thread()   # TODO: take in axes increment as arg, clip maybe
