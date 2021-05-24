#!/usr/bin/env python

# This process publishes joystick events. Such events can be subscribed by
# mocked car controller scripts.

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
    self.axes_values = {'steer': 0., 'accel': 0.}

    if self.use_keyboard:
      self.axes = {'w': ['forward', 'accel'], 'a': ['left', 'steer'], 's': ['backward', 'accel'], 'd': ['right', 'steer']}   # add more for the joystick buttons
      self.axes_increment = 0.05  # 5% of full actuation each key press

      self.buttons = {'r': 'reset'}  # TODO: add previous buttons (like pcm_cancel_cmd, toggle engaged, etc.)
    else:
      raise NotImplementedError("Only keyboard is supported for now")  # TODO: support joystick

  def update(self):
    if self.use_keyboard:
      key = self.kb.getch().lower()
      if key in self.axes:
        event, event_type = self.axes[key]
        sign = 1. if event in ['forward', 'left'] else -1.
        v = self.axes_values[event_type]
        self.axes_values[event_type] = round(clip(v + sign * self.axes_increment, -1., 1.), 3)
      elif key in self.buttons:
        if self.buttons[key] == 'reset':
          self.axes_values = {'steer': 0., 'accel': 0.}
      else:
        print('Key not assigned to an action!')
      return


BUTTONS = ['pcm_cancel_cmd', 'engaged_toggle', 'steer_required']
joystick = Joystick(use_keyboard=True)





def send_thread(command_address):
  POLL_RATE = int(1000 / 10.)  # 10 hz
  zmq.Context._instance = None
  context = zmq.Context.instance()

  command_sock = context.socket(zmq.PULL)
  command_sock.bind(command_address)

  poller = zmq.Poller()
  poller.register(command_sock, zmq.POLLIN)
  joystick_sock = messaging.pub_sock('testJoystick')

  # starting message to send to controlsd if user doesn't type any keys
  msg = joystick.axes_values
  while True:
    for sock in dict(poller.poll(POLL_RATE)):
      msg = sock.recv_pyobj()  # TODO: only receives axes for now

    dat = messaging.new_message('testJoystick')
    dat.testJoystick.axes = [msg[a] for a in ['accel', 'steer']]
    dat.testJoystick.buttons = [False for _ in BUTTONS]

    joystick_sock.send(dat.to_bytes())
    print(f'Sent: {dat}')


def joystick_thread():
  command_address = "ipc:///tmp/{}".format(uuid4())

  command_sock = zmq.Context.instance().socket(zmq.PUSH)
  command_sock.connect(command_address)

  send_thread_proc = multiprocessing.Process(target=send_thread, args=(command_address,))
  send_thread_proc.start()

  # Receive joystick/key events and send to joystick send thread
  try:
    while 1:
      joystick.update()
      command_sock.send_pyobj(joystick.axes_values)
      # TODO: time shouldn't matter since joystick.update() is blocking
  except KeyboardInterrupt:
    send_thread_proc.terminate()


if __name__ == "__main__":
  joystick_thread()   # TODO: take in axes increment as arg, clip maybe
