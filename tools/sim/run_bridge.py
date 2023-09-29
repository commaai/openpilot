#!/usr/bin/env python
import argparse
import os

from typing import Any
from multiprocessing import Queue

from openpilot.tools.sim.bridge.common import SimulatorBridge
from openpilot.tools.sim.bridge.carla.carla_bridge import CarlaBridge
from openpilot.tools.sim.bridge.metadrive.metadrive_bridge import MetaDriveBridge


def parse_args(add_args=None):
  parser = argparse.ArgumentParser(description='Bridge between the simulator and openpilot.')
  parser.add_argument('--joystick', action='store_true')
  parser.add_argument('--high_quality', action='store_true')
  parser.add_argument('--dual_camera', action='store_true')
  parser.add_argument('--simulator', dest='simulator', type=str, default='carla')

  # Carla specific
  parser.add_argument('--town', type=str, default='Town04_Opt')
  parser.add_argument('--spawn_point', dest='num_selected_spawn_point', type=int, default=16)
  parser.add_argument('--host', dest='host', type=str, default=os.environ.get("CARLA_HOST", '127.0.0.1'))
  parser.add_argument('--port', dest='port', type=int, default=2000)

  return parser.parse_args(add_args)

if __name__ == "__main__":
  q: Any = Queue()
  args = parse_args()

  simulator_bridge: SimulatorBridge
  if args.simulator == "carla":
    simulator_bridge = CarlaBridge(args)
  elif args.simulator == "metadrive":
    simulator_bridge = MetaDriveBridge(args)
  else:
    raise AssertionError("simulator type not supported")
  p = simulator_bridge.run(q)

  if args.joystick:
    # start input poll for joystick
    from openpilot.tools.sim.lib.manual_ctrl import wheel_poll_thread

    wheel_poll_thread(q)
  else:
    # start input poll for keyboard
    from openpilot.tools.sim.lib.keyboard_ctrl import keyboard_poll_thread

    keyboard_poll_thread(q)

  simulator_bridge.shutdown()

  p.join()
