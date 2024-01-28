#!/usr/bin/env python
import argparse

from typing import Any
from multiprocessing import Queue

from openpilot.tools.sim.bridge.metadrive.metadrive_bridge import MetaDriveBridge

def create_bridge(dual_camera, high_quality):
  q: Any = Queue()

  simulator_bridge = MetaDriveBridge(dual_camera, high_quality)
  p = simulator_bridge.run(q)

  return q, p

def main():
  _, p = create_bridge(True, False)
  p.join()

def parse_args(add_args=None):
  parser = argparse.ArgumentParser(description='Bridge between the simulator and openpilot.')
  parser.add_argument('--joystick', action='store_true')
  parser.add_argument('--high_quality', action='store_true')
  parser.add_argument('--dual_camera', action='store_true')

  return parser.parse_args(add_args)

if __name__ == "__main__":
  args = parse_args()

  q, p = create_bridge(args.dual_camera, args.high_quality)

  if args.joystick:
    # start input poll for joystick
    from openpilot.tools.sim.lib.manual_ctrl import wheel_poll_thread

    wheel_poll_thread(q)
  else:
    # start input poll for keyboard
    from openpilot.tools.sim.lib.keyboard_ctrl import keyboard_poll_thread

    keyboard_poll_thread(q)

  p.join()
