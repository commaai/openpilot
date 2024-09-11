#!/usr/bin/env python3
import io
import os
import time
import base64
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

from cereal import messaging, car
from opendbc.car.structs import CarControl
from opendbc.car.common.conversions import Conversions
from openpilot.common.realtime import DT_CTRL, DT_MDL, Ratekeeper
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.controls.lib.longitudinal_planner import get_accel_from_plan

# TODOs
# - support lateral maneuvers
# - setup: show countdown?


@dataclass
class Action:
  accel: float      # m/s^2
  duration: float   # seconds
  longControlState: CarControl.Actuators.LongControlState = CarControl.Actuators.LongControlState.pid

  def get_msgs(self):
    return [
      (t, CarControl(
        enabled=True,
        longActive=True,
        actuators=CarControl.Actuators(
          accel=self.accel,
          longControlState=self.longControlState,
        ),
      ))
      for t in np.linspace(0, self.duration, int(self.duration/DT_CTRL))
    ]

@dataclass
class Maneuver:
  description: str
  actions: list[Action]
  # TODO: implement repeat
  repeat: int = 1
  initial_speed: float = 0.  # m/s

  _active: bool = False
  _finished: bool = False
  _action_index: int = 0
  _start_frame: int = 0
  _ready_cnt: int = 0
  _active_frames: int = 0

  def get_accel(self, v_ego: float, enabled: bool, standstill: bool, frame: int) -> float:
    ready = abs(v_ego - self.initial_speed) < 0.4 and enabled and not standstill
    self._ready_cnt = (self._ready_cnt + 1) if ready else 0

    if self._ready_cnt > (2. / DT_MDL):
      self._active = True

    if not self._active:
      return self.initial_speed - v_ego

    action = self.actions[self._action_index]

    self._active_frames += 1

    if self._active_frames > (action.duration / DT_MDL):
      self._action_index += 1
      self._active_frames = 0
      if self._action_index == len(self.actions):
        self._finished = True
        return 0.

    return action.accel

  @property
  def finished(self):
    return self._finished

  # def get_msgs(self):
  #   t0 = 0
  #   for action in self.actions:
  #     for lt, msg in action.get_msgs():
  #       yield lt + t0, msg
  #     t0 += lt

MANEUVERS = [
  Maneuver(
   "creep: alternate between +1m/ss and -1m/ss",
   [
     Action(1, 2), Action(-1, 2),
     Action(1, 2), Action(-1, 2),
     Action(1, 2), Action(-1, 2),
   ],
   repeat=2,
   initial_speed=0.,
  ),
  Maneuver(
    "brake step response: -1m/ss from 20mph",
    [Action(-1, 3)],
    repeat=3,
    initial_speed=20. * Conversions.MPH_TO_MS,
  ),
  Maneuver(
    "brake step response: -4m/ss from 20mph",
    [Action(-4, 3)],
    repeat=3,
    initial_speed=20. * Conversions.MPH_TO_MS,
  ),
  Maneuver(
    "gas step response: +1m/ss from 20mph",
    [Action(1, 3)],
    repeat=3,
    initial_speed=20. * Conversions.MPH_TO_MS,
  ),
  Maneuver(
    "gas step response: +4m/ss from 20mph",
    [Action(4, 3)],
    repeat=3,
    initial_speed=20. * Conversions.MPH_TO_MS,
  ),
]


def report(args, logs, fp):
  output_path = Path(__file__).resolve().parent / "longitudinal_reports"
  output_fn = args.output or output_path / f"{fp}_{time.strftime('%Y%m%d-%H_%M_%S')}.html"
  output_path.mkdir(exist_ok=True)
  with open(output_fn, "w") as f:
    f.write("<h1>Longitudinal maneuver report</h1>\n")
    f.write(f"<h3>{fp}</h3>\n")
    if args.desc:
      f.write(f"<h3>{args.desc}</h3>")
    for description, runs in logs.items():
      f.write("<div style='border-top: 1px solid #000; margin: 20px 0;'></div>\n")
      f.write(f"<h2>{description}</h2>\n")
      for run, log in runs.items():
        f.write(f"<h3>Run #{int(run)+1}</h3>\n")
        plt.rcParams['font.size'] = 40
        fig = plt.figure(figsize=(30, 25))
        ax = fig.subplots(4, 1, sharex=True, gridspec_kw={'hspace': 0, 'height_ratios': [5, 3, 1, 1]})

        ax[0].grid(linewidth=4)
        ax[0].plot(log["t"], log["carControl.actuators.accel"], label='accel command', linewidth=6)
        ax[0].plot(log["t"], log["carState.aEgo"], label='aEgo', linewidth=6)
        ax[0].set_ylabel('Acceleration (m/s^2)')
        #ax[0].set_ylim(-6.5, 6.5)
        ax[0].legend()

        ax[1].grid(linewidth=4)
        ax[1].plot(log["t"], log["carState.vEgo"], 'g', label='vEgo', linewidth=6)
        ax[1].set_ylabel('Velocity (m/s)')
        ax[1].legend()

        ax[2].plot(log["t"], log["carControl.enabled"], label='enabled', linewidth=6)
        ax[3].plot(log["t"], log["carState.gasPressed"], label='gasPressed', linewidth=6)
        ax[3].plot(log["t"], log["carState.brakePressed"], label='brakePressed', linewidth=6)
        for i in (2, 3):
          ax[i].set_yticks([0, 1], minor=False)
          ax[i].set_ylim(-1, 2)
          ax[i].legend()

        ax[-1].set_xlabel("Time (s)")
        fig.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        f.write(f"<img src='data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}' style='width:100%; max-width:800px;'>\n")

    import json
    f.write(f"<p style='display: none'>{json.dumps(logs)}</p>")
  print(f"\nReport written to {output_fn}\n")


def main():
  params = Params()
  cloudlog.info("joystickd is waiting for CarParams")
  CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)

  sm = messaging.SubMaster(['carState', 'controlsState', 'selfdriveState', 'modelV2'], poll='modelV2')
  pm = messaging.PubMaster(['longitudinalPlan', 'driverAssistance'])

  maneuvers = iter(MANEUVERS)
  maneuver = None

  while True:
    sm.update()

    if maneuver is None:
      maneuver = next(maneuvers, None)

    if maneuver is None:
      print('We are done!')

    plan_send = messaging.new_message('longitudinalPlan')
    plan_send.valid = sm.all_checks()

    longitudinalPlan = plan_send.longitudinalPlan
    accel = 0
    cs = sm['carState']

    if maneuver is not None:
      accel = maneuver.get_accel(cs.vEgo, cs.cruiseState.enabled, cs.cruiseState.standstill, sm.frame)

    longitudinalPlan.aTarget = accel
    longitudinalPlan.shouldStop = cs.vEgo < CP.vEgoStopping and accel < 0  # should_stop

    longitudinalPlan.allowBrake = True
    longitudinalPlan.allowThrottle = True
    longitudinalPlan.hasLead = True

    pm.send('longitudinalPlan', plan_send)

    assistance_send = messaging.new_message('driverAssistance')
    assistance_send.valid = True
    pm.send('driverAssistance', assistance_send)

    print('finished?', maneuver.finished)
    print('aTarget:', longitudinalPlan.aTarget)

    if maneuver is not None and maneuver.finished:
      maneuver = None


if __name__ == "__main__":
  main()
  exit()

  maneuver_help = "\n".join([f"{i+1}. {m.description}" for i, m in enumerate(MANEUVERS)])
  parser = argparse.ArgumentParser(description="A tool for longitudinal control testing.",
                                   formatter_class=argparse.RawTextHelpFormatter)
  parser.add_argument('--desc', help="Extra description to include in report.")
  parser.add_argument('--output', help="Write out report to this file.", default=None)
  parser.add_argument('maneuvers', nargs='*', type=int, default=None, help=f'Deafult is all.\n{maneuver_help}')
  args = parser.parse_args()
  print(args)

  if "REPORT_TEST" in os.environ:
    with open(os.environ["REPORT_TEST"]) as f:
      import json
      logs = json.loads(f.read().split("none'>")[1].split('</p>')[0])
    report(args, logs, "testing")
    exit()

  assert args.output is None or args.output.endswith(".html"), "Output filename must end with '.html'"

  main(args)
