from multiprocessing import Queue
from tools.sim.bridge.carla import CarlaBridge  # pylint: disable = no-name-in-module
from tools.sim.bridge.metadrive import MetaDriveBridge  # pylint: disable = no-name-in-module
from tools.sim.bridge.common import parse_args, SimulatorBridge  # pylint: disable = no-name-in-module
from common.params import Params

from typing import Any

if __name__ == "__main__":
  q: Any = Queue()
  args = parse_args()

  try:
    simulator_bridge: SimulatorBridge
    if args.simulator == "metadrive":
      if args.dual_camera:
        raise AssertionError("Dual camera not supported in MetaDrive simulator for now")
      simulator_bridge = MetaDriveBridge(args)
    elif args.simulator == "carla":
      simulator_bridge = CarlaBridge(args)
    else:
      raise AssertionError("simulator type not supported")
    p = simulator_bridge.run(q)

    if args.joystick:
      # start input poll for joystick
      from tools.sim.lib.manual_ctrl import wheel_poll_thread

      wheel_poll_thread(q)
    else:
      # start input poll for keyboard
      from tools.sim.lib.keyboard_ctrl import keyboard_poll_thread

      keyboard_poll_thread(q)
    p.join()

  finally:
    # Try cleaning up the wide camera param
    # in case users want to use replay after
    Params().remove("WideCameraOnly")
