#!/usr/bin/env python

# This process publishes joystick events. Such events can be suscribed by
# mocked car controller scripts.


### this process needs pygame and can't run on the EON ###

import pygame  # pylint: disable=import-error
import time
import zmq
import cereal.messaging as messaging
from common.numpy_fast import clip
from tools.lib.kbhit import KBHit
import multiprocessing
from uuid import uuid4


kb = KBHit()


class Joystick:
  def __init__(self, use_keyboard=True):
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
      key = kb.getch().lower()
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


import sys
import select

joystick = Joystick(use_keyboard=True)


# while True:
#   joystick.update()
#   print(joystick.axes_values)

BUTTONS = ['pcm_cancel_cmd', 'engaged_toggle', 'steer_required']


def new_joystick_msg(axes, btns):
  dat = messaging.new_message('testJoystick')
  dat.testJoystick.axes = axes
  dat.testJoystick.buttons = btns
  return dat


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
    evts = dict(poller.poll(POLL_RATE))  # blocking
    if command_sock in evts:
      msg = command_sock.recv_pyobj()  # only receives axes for now

    dat = new_joystick_msg([msg[a] for a in ['accel', 'steer']],
                           [False for _ in BUTTONS])

    joystick_sock.send(dat.to_bytes())
    print(f'Sent: {dat}')


def joystick_thread():
  command_address = "ipc:///tmp/{}".format(uuid4())

  command_sock = zmq.Context.instance().socket(zmq.PUSH)
  command_sock.connect(command_address)

  poller = zmq.Poller()
  poller.register(command_sock, zmq.POLLIN)

  send_thread_proc = multiprocessing.Process(target=send_thread, args=(command_address,))
  send_thread_proc.start()

  while 1:
    joystick.update()
    command_sock.send_pyobj(joystick.axes_values)
    time.sleep(1 / 100.)  # TODO: abstract this

  # pygame.init()

  # Used to manage how fast the screen updates
  # clock = pygame.time.Clock()

  # Initialize the joysticks
  # pygame.joystick.init()

  # Get count of joysticks
  # joystick_count = pygame.joystick.get_count()
  # if joystick_count > 1:
  #   raise ValueError("More than one joystick attached")
  # elif joystick_count < 1:
  #   raise ValueError("No joystick found")

  # -------- Main Program Loop -----------
  while True:
    # EVENT PROCESSING STEP
    # for event in pygame.event.get():  # User did something
    #   if event.type == pygame.QUIT:  # If user clicked close
    #     pass
    #   Available joystick events: JOYAXISMOTION JOYBALLMOTION JOYBUTTONDOWN JOYBUTTONUP JOYHATMOTION
      # if event.type == pygame.JOYBUTTONDOWN:
      #   print("Joystick button pressed.")
      # if event.type == pygame.JOYBUTTONUP:
      #   print("Joystick button released.")

    # joystick = pygame.joystick.Joystick(0)
    # joystick.init()

    # Usually axis run in pairs, up/down for one, and left/right for
    # the other.
    joystick.update()
    axes = [joystick.axes_values[a] for a in ['accel', 'steer']]
    buttons = [False for _ in BUTTONS]
    print(axes)

    # for a in range(joystick.get_numaxes()):
    #   axes.append(joystick.get_axis(a))
    #
    # for b in range(joystick.get_numbuttons()):
    #   buttons.append(bool(joystick.get_button(b)))
    #
    dat = messaging.new_message('testJoystick')
    dat.testJoystick.axes = axes
    dat.testJoystick.buttons = buttons
    joystick_sock.send(dat.to_bytes())

    # Limit to 100 frames per second
    time.sleep(1 / 100)

if __name__ == "__main__":
  joystick_thread()
