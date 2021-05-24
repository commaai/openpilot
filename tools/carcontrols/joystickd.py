#!/usr/bin/env python

# This process publishes joystick events. Such events can be suscribed by
# mocked car controller scripts.


### this process needs pygame and can't run on the EON ###

import pygame  # pylint: disable=import-error
import time
import cereal.messaging as messaging
from common.numpy_fast import clip
# from tools.sim.lib.keyboard_ctrl import getch, IFLAG, OFLAG, CFLAG, LFLAG, ISPEED, OSPEED, CC
from tools.lib.kbhit import KBHit

kb = KBHit()

class Joystick:
  def __init__(self, use_keyboard=True):
    self.use_keyboard = use_keyboard
    self.axes_values = {'steer': 0., 'accel': 0.}

    if self.use_keyboard:
      # self.axes = {'lat'}
      self.axes = {'w': ['forward', 'accel'], 'a': ['left', 'steer'], 's': ['backward', 'accel'], 'd': ['right', 'steer']}   # add more for the joystick buttons
      self.axes_increment = 0.05  # 5% of full actuation each key press

      self.buttons = {'r': 'reset'}  # TODO: add previous buttons (like pcm_cancel_cmd, toggle engaged, etc.)
    else:
      raise NotImplementedError("Only keyboard is supported for now")  # TODO: support joystick

  def update(self, key):
    if self.use_keyboard:
      old_settings = termios.tcgetattr(sys.stdin)
      try:
        tty.setcbreak(sys.stdin.fileno())
        got_data = False
        while has_data():
          got_data = True
          key = sys.stdin.read(1)
        # else:
        #   return
        # key = kb.getch().lower()
        if not got_data:
          return
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
      finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


import sys
import select
import tty
import termios


def has_data():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

joystick = Joystick(use_keyboard=True)


# old_settings = termios.tcgetattr(sys.stdin)
# try:
#   tty.setcbreak(sys.stdin.fileno())
#   while 1:
#     print('no data')
#     time.sleep(1 / 100)
#     if has_data():
#       c = sys.stdin.read(1)
#       joystick.update(c)
# finally:
#   termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


while True:
  joystick.update(None)
  print(joystick.axes_values)
  time.sleep(0.01)
BUTTONS = ['pcm_cancel_cmd', 'engaged_toggle', 'steer_required']

def joystick_thread():
  joystick_sock = messaging.pub_sock('testJoystick')

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
