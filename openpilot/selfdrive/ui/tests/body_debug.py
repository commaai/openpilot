#!/usr/bin/env python3
"""Interactive PC harness for exercising the body UI states."""

import os
import select
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


# Keep the debug UI's params and msgq sockets separate from normal openpilot.
os.environ.setdefault("OPENPILOT_PREFIX", "body_ui_debug")
os.environ.setdefault("PARAMS_ROOT", "/tmp/openpilot_body_ui_debug_params")

from openpilot.cereal import log, messaging
from openpilot.common.params import Params
from openpilot.common.version import terms_version, training_version
from opendbc.car.structs import car


UI_PATH = Path(__file__).parents[1] / "ui.py"
PUBLISH_HZ = 20
UI_SERVICES = [
  "modelV2",
  "controlsState",
  "onroadEvents",
  "liveCalibration",
  "radarState",
  "deviceState",
  "pandaStates",
  "carParams",
  "driverMonitoringState",
  "carState",
  "driverStateV2",
  "roadCameraState",
  "wideRoadCameraState",
  "managerState",
  "selfdriveState",
  "longitudinalPlan",
  "gpsLocationExternal",
  "carOutput",
  "carControl",
  "liveParameters",
  "testJoystick",
  "rawAudioData",
]


@dataclass
class BodyState:
  onroad: bool = True
  charging: bool = True
  charge_percent: int = 75
  livestreaming: bool = False
  speed: float = 0.0
  steer: float = 0.0


def print_help() -> None:
  print(
    """
Commands:
  c                  toggle charging
  charge <0-100>     set battery charge
  + / -              increase/decrease charge by 10%
  l                  toggle livestreaming
  o                  toggle onroad/ignition
  m                  toggle stopped/moving
  speed <m/s>        set body speed
  left/right/center  set steering animation
  s                  print current state
  h                  show this help
  q                  quit
""".strip()
  )


def print_state(state: BodyState) -> None:
  motion = f"{state.speed:g} m/s"
  steer = "left" if state.steer < 0 else "right" if state.steer > 0 else "center"
  status = f"onroad={state.onroad}  charging={state.charging}  charge={state.charge_percent}%  "
  status += f"livestreaming={state.livestreaming}  speed={motion}  steer={steer}"
  print(status)


def setup_params(state: BodyState) -> Params:
  params = Params()
  CP = car.CarParams()
  CP.brand = "body"
  CP.notCar = True
  params.put("CarParamsPersistent", CP.to_bytes(), block=True)
  params.put("HasAcceptedTerms", terms_version, block=True)
  params.put("CompletedTrainingVersion", training_version, block=True)
  params.put_bool("IsLiveStreaming", state.livestreaming, block=True)
  return params


def publish_state(pm: messaging.PubMaster, state: BodyState) -> None:
  device_state = messaging.new_message("deviceState", valid=True)
  device_state.deviceState.started = state.onroad
  pm.send("deviceState", device_state)

  panda_states = messaging.new_message("pandaStates", 1, valid=True)
  panda_states.pandaStates[0].pandaType = log.PandaState.PandaType.dos
  panda_states.pandaStates[0].ignitionLine = state.onroad
  pm.send("pandaStates", panda_states)

  car_state = messaging.new_message("carState", valid=True)
  car_state.carState.charging = state.charging
  car_state.carState.fuelGauge = state.charge_percent / 100
  car_state.carState.vEgo = state.speed
  car_state.carState.standstill = state.speed == 0
  pm.send("carState", car_state)

  joystick = messaging.new_message("testJoystick", valid=True)
  joystick.testJoystick.axes = [0.0, state.steer]
  pm.send("testJoystick", joystick)


def handle_command(command: str, state: BodyState, params: Params) -> bool:
  tokens = command.lower().split()
  if not tokens:
    return True

  try:
    if tokens[0] == "q":
      return False
    elif tokens[0] == "h":
      print_help()
    elif tokens[0] == "s":
      pass
    elif tokens[0] == "c":
      state.charging = not state.charging
    elif tokens[0] == "charge" and len(tokens) == 2:
      state.charge_percent = max(0, min(100, int(tokens[1])))
    elif tokens[0] == "+":
      state.charge_percent = min(100, state.charge_percent + 10)
    elif tokens[0] == "-":
      state.charge_percent = max(0, state.charge_percent - 10)
    elif tokens[0] == "l":
      state.livestreaming = not state.livestreaming
      params.put_bool("IsLiveStreaming", state.livestreaming)
    elif tokens[0] == "o":
      state.onroad = not state.onroad
    elif tokens[0] == "m":
      state.speed = 0.0 if state.speed else 1.0
    elif tokens[0] == "speed" and len(tokens) == 2:
      state.speed = float(tokens[1])
    elif tokens[0] == "left":
      state.steer = -1.0
    elif tokens[0] == "right":
      state.steer = 1.0
    elif tokens[0] == "center":
      state.steer = 0.0
    else:
      print(f"Unknown command: {command!r}. Enter 'h' for help.")
      return True
  except ValueError:
    print(f"Invalid value in command: {command!r}")
    return True

  print_state(state)
  return True


def main() -> None:
  state = BodyState()
  params = setup_params(state)
  shm_root = "/tmp" if sys.platform == "darwin" else "/dev/shm"
  (Path(shm_root) / f"msgq_{os.environ['OPENPILOT_PREFIX']}").mkdir(exist_ok=True)
  # Publishers create the msgq endpoints that the UI's SubMaster connects to.
  pm = messaging.PubMaster(UI_SERVICES)
  ui_process = subprocess.Popen([sys.executable, str(UI_PATH)], env=os.environ.copy())

  try:
    print_help()
    print_state(state)

    running = True
    while running and ui_process.poll() is None:
      publish_state(pm, state)
      readable, _, _ = select.select([sys.stdin], [], [], 1 / PUBLISH_HZ)
      if readable:
        command = sys.stdin.readline()
        if not command:  # stdin closed
          break
        running = handle_command(command.strip(), state, params)
  except KeyboardInterrupt:
    pass
  finally:
    ui_process.terminate()
    try:
      ui_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
      ui_process.kill()
      ui_process.wait()


if __name__ == "__main__":
  main()
