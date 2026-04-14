"""Launch the comma UI while simulating a comma body."""

import argparse
import os
from threading import Thread
from time import sleep

from cereal import car, log, messaging
from openpilot.common.params import Params


def send_messages():
  pm = messaging.PubMaster(['deviceState', 'pandaStates', 'carParams', 'carState', 'selfdriveState'])

  car_params = messaging.new_message('carParams')
  car_params.carParams.brand = "body"
  car_params.carParams.notCar = True

  device_state = messaging.new_message('deviceState')
  device_state.deviceState.started = True

  panda_states = messaging.new_message('pandaStates', 1)
  panda_states.pandaStates[0].ignitionLine = True
  panda_states.pandaStates[0].pandaType = log.PandaState.PandaType.uno

  car_state = messaging.new_message('carState')
  car_state.carState.charging = True
  car_state.carState.fuelGauge = 0.80

  selfdrive_state = messaging.new_message('selfdriveState')
  selfdrive_state.selfdriveState.enabled = True

  messages = (
    ('carParams', car_params),
    ('deviceState', device_state),
    ('pandaStates', panda_states),
    ('carState', car_state),
    ('selfdriveState', selfdrive_state),
  )

  while True:
    for service, msg in messages:
      pm.send(service, msg)
    sleep(0.01)


def main():
  parser = argparse.ArgumentParser(description="Launch body view UI")
  parser.add_argument("--big", action="store_true", help="Launch in big UI mode (comma 3X)")
  parser.add_argument("--joystick", action="store_true", help="Wait for joystick_control before going onroad")
  parser.add_argument("--monitor", type=int, help="Pin window to specified monitor index (e.g. 0, 1)")
  args = parser.parse_args()

  if args.big:
    os.environ["BIG"] = "1"

  if args.monitor is not None:
    import pyray as rl
    from openpilot.system.ui.lib.application import gui_app

    init_window = gui_app.init_window

    def init_window_on_monitor(*window_args, **window_kwargs):
      init_window(*window_args, **window_kwargs)
      pos = rl.get_monitor_position(args.monitor)
      rl.set_window_position(int(pos.x), int(pos.y))

    gui_app.init_window = init_window_on_monitor

  params = Params()
  params.put("CarParamsPersistent", car.CarParams.new_message(
    notCar=True,
    brand="body",
    wheelbase=1,
    steerRatio=10,
  ).to_bytes())
  params.put_bool("JoystickDebugMode", True)

  if args.joystick:
    sm = messaging.SubMaster(['testJoystick'])
    print("Waiting for joystick_control to start (run: python tools/joystick/joystick_control.py --keyboard) ...")
    while not sm.recv_frame['testJoystick']:
      params.put_bool("IsOffroad", True)
      sm.update(100)
    print("Joystick connected, starting body view.")
    params.remove("IsOffroad")

  Thread(target=send_messages, daemon=True).start()

  from openpilot.selfdrive.ui.ui import main as ui_main
  ui_main()


if __name__ == "__main__":
  main()
