#!/usr/bin/env python

# This process publishes joystick events. Such events can be suscribed by
# mocked car controller scripts.


### this process needs pygame and can't run on the EON ###

import pygame
import zmq
import cereal.messaging as messaging


def joystick_thread():
  joystick_sock = messaging.pub_sock('testJoystick')

  pygame.init()

  # Used to manage how fast the screen updates
  clock = pygame.time.Clock()

  # Initialize the joysticks
  pygame.joystick.init()

  # Get count of joysticks
  joystick_count = pygame.joystick.get_count()
  if joystick_count > 1:
    raise ValueError("More than one joystick attached")
  elif joystick_count < 1:
    raise ValueError("No joystick found")

  # -------- Main Program Loop -----------
  while True:
    # EVENT PROCESSING STEP
    for event in pygame.event.get(): # User did something
      if event.type == pygame.QUIT: # If user clicked close
        pass
      # Available joystick events: JOYAXISMOTION JOYBALLMOTION JOYBUTTONDOWN JOYBUTTONUP JOYHATMOTION
      if event.type == pygame.JOYBUTTONDOWN:
        print("Joystick button pressed.")
      if event.type == pygame.JOYBUTTONUP:
        print("Joystick button released.")

    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    # Usually axis run in pairs, up/down for one, and left/right for
    # the other.
    axes = []
    buttons = []

    for a in range(joystick.get_numaxes()):
      axes.append(joystick.get_axis(a))

    for b in range(joystick.get_numbuttons()):
      buttons.append(bool(joystick.get_button(b)))

    dat = messaging.new_message('testJoystick')
    dat.testJoystick.axes = axes
    dat.testJoystick.buttons = buttons
    joystick_sock.send(dat.to_bytes())

    # Limit to 100 frames per second
    clock.tick(100)

if __name__ == "__main__":
  joystick_thread()
