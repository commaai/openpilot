#!/usr/bin/env python3
"""Launch the comma UI, but simulating a comma body."""
import argparse
import os
import time
import threading


import pyray as rl
from cereal import car, log, messaging
from openpilot.common.params import Params


def send_messages():
  pm = messaging.PubMaster(['deviceState', 'pandaStates', 'carParams', 'carState', 'selfdriveState'])

  car_params_msg = messaging.new_message('carParams')
  car_params_msg.carParams.brand = "body"
  car_params_msg.carParams.notCar = True

  device_state_msg = messaging.new_message('deviceState')
  device_state_msg.deviceState.started = False

  panda_msg = messaging.new_message('pandaStates', 1)
  panda_msg.pandaStates[0].ignitionLine = True
  panda_msg.pandaStates[0].pandaType = log.PandaState.PandaType.uno

  car_state_msg = messaging.new_message('carState')
  car_state_msg.carState.charging = True
  car_state_msg.carState.fuelGauge = 0.80

  selfdrive_state_msg = messaging.new_message('selfdriveState')
  selfdrive_state_msg.selfdriveState.enabled = True

  while True:
    pm.send('carParams', car_params_msg)
    pm.send('deviceState', device_state_msg)
    pm.send('pandaStates', panda_msg)
    pm.send('carState', car_state_msg)
    pm.send('selfdriveState', selfdrive_state_msg)
    time.sleep(0.01)


def main():
  parser = argparse.ArgumentParser(description="Launch body view UI")
  parser.add_argument("--big", action="store_true", help="Launch in big UI mode (comma 3X)")
  parser.add_argument("--joystick", action="store_true", help="Wait for joystick_control before going onroad")
  parser.add_argument("--monitor", type=int, default=None, help="Pin window to specified monitor index (e.g. 0, 1)")
  args = parser.parse_args()

  if args.big:
    os.environ["BIG"] = "1"
  from openpilot.system.ui.lib.application import gui_app

  if args.monitor is not None:
    # Pin window to specified monitor after init
    monitor_index = args.monitor
    _orig_init_window = gui_app.init_window
    def _init_window_on_monitor(*a, **kw):
      _orig_init_window(*a, **kw)
      pos = rl.get_monitor_position(monitor_index)
      rl.set_window_position(int(pos.x), int(pos.y))
    gui_app.init_window = _init_window_on_monitor

  # Set CarParamsPersistent so ui_state.CP.notCar is True on startup
  params = Params()
  CP = car.CarParams.new_message(notCar=True, brand="body", wheelbase=1, steerRatio=10)
  params.put("CarParamsPersistent", CP.to_bytes())
  params.put_bool("JoystickDebugMode", True)

  if args.joystick:
    params.put_bool("IsOffroad", True)

    # Wait for joystick_control to start before going "onroad"
    sm = messaging.SubMaster(['testJoystick'])
    print("Waiting for joystick_control to start (run: python tools/joystick/joystick_control.py --keyboard) ...")
    while sm.recv_frame['testJoystick'] == 0:
      params.put_bool("IsOffroad", True)
      sm.update(100)
    print("Joystick connected, starting body view.")
    params.remove("IsOffroad")

  # Start message sender in background
  t = threading.Thread(target=send_messages, daemon=True)
  t.start()

  # Import after env is set so BIG_UI picks it up
  from openpilot.selfdrive.ui.ui import main as ui_main
  ui_main()


if __name__ == "__main__":
  main()
