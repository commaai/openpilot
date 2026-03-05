#!/usr/bin/env python3
import numpy as np
from dataclasses import dataclass

from cereal import messaging, car
from openpilot.common.constants import CV
from openpilot.common.realtime import DT_MDL
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.controls.lib.drive_helpers import MIN_SPEED


@dataclass
class Action:
  accel_bp: list[float]  # lateral acceleration (m/s^2)
  time_bp: list[float]   # seconds

  def __post_init__(self):
    assert len(self.accel_bp) == len(self.time_bp)


@dataclass
class Maneuver:
  description: str
  actions: list[Action]
  repeat: int = 0
  initial_speed: float = 0.  # m/s

  _active: bool = False
  _finished: bool = False
  _action_index: int = 0
  _action_frames: int = 0
  _ready_cnt: int = 0
  _repeated: int = 0

  def get_accel(self, v_ego: float, lat_active: bool) -> float:
    ready = abs(v_ego - self.initial_speed) < 0.3 and lat_active
    self._ready_cnt = (self._ready_cnt + 1) if ready else 0

    if self._ready_cnt > (2. / DT_MDL):
      self._active = True

    if not self._active:
      return 0.

    action = self.actions[self._action_index]
    action_accel = np.interp(self._action_frames * DT_MDL, action.time_bp, action.accel_bp)

    self._action_frames += 1

    # reached duration of action
    if self._action_frames > (action.time_bp[-1] / DT_MDL):
      # next action
      if self._action_index < len(self.actions) - 1:
        self._action_index += 1
        self._action_frames = 0
      # repeat maneuver
      elif self._repeated < self.repeat:
        self._repeated += 1
        self._action_index = 0
        self._action_frames = 0
        self._active = False
      # finish maneuver
      else:
        self._finished = True

    return float(action_accel)

  @property
  def finished(self):
    return self._finished

  @property
  def active(self):
    return self._active


MANEUVERS = [
  Maneuver(
    "step up 30mph a=0.3",
    [Action([0], [0.5]), Action([0.3], [1.5])],
    repeat=2,
    initial_speed=30. * CV.MPH_TO_MS,
  ),
  Maneuver(
    "step down 30mph a=0.3",
    [Action([0.3], [0.5]), Action([0], [1.5])],
    repeat=2,
    initial_speed=30. * CV.MPH_TO_MS,
  ),
  Maneuver(
    "step up 40mph a=0.3",
    [Action([0], [0.5]), Action([0.3], [1.5])],
    repeat=2,
    initial_speed=40. * CV.MPH_TO_MS,
  ),
  Maneuver(
    "step down 40mph a=0.3",
    [Action([0.3], [0.5]), Action([0], [1.5])],
    repeat=2,
    initial_speed=40. * CV.MPH_TO_MS,
  ),
  Maneuver(
    "step up 50mph a=0.3",
    [Action([0], [0.5]), Action([0.3], [1.5])],
    repeat=2,
    initial_speed=50. * CV.MPH_TO_MS,
  ),
  Maneuver(
    "step down 50mph a=0.3",
    [Action([0.3], [0.5]), Action([0], [1.5])],
    repeat=2,
    initial_speed=50. * CV.MPH_TO_MS,
  ),
  Maneuver(
    "S-curve weave",
    [
      Action([0.0, 2.5], [0.0, 1.0]),
      Action([2.5, 2.5], [0.0, 1.5]),
      Action([2.5, -2.5], [0.0, 2.5]),
      Action([-2.5, -2.5], [0.0, 1.5]),
      Action([-2.5, 0.0], [0.0, 1.0]),
    ],
    repeat=2,
    initial_speed=25. * CV.MPH_TO_MS,
  ),
  Maneuver(
    "steady right curve",
    [
      Action([0.0, 2.0], [0.0, 1.5]),
      Action([2.0, 2.0], [0.0, 3.0]),
      Action([2.0, 0.0], [0.0, 1.5]),
    ],
    repeat=2,
    initial_speed=25. * CV.MPH_TO_MS,
  ),
  Maneuver(
    "double lane change",
    [
      Action([0.0, 2.0], [0.0, 0.8]),
      Action([2.0, 2.0], [0.0, 1.2]),
      Action([2.0, -2.0], [0.0, 1.5]),
      Action([-2.0, -2.0], [0.0, 1.2]),
      Action([-2.0, 0.0], [0.0, 0.8]),
    ],
    repeat=2,
    initial_speed=25. * CV.MPH_TO_MS,
  ),
]

def main():
  params = Params()
  cloudlog.info("lateral_maneuversd is waiting for CarParams")
  CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)

  sm = messaging.SubMaster(['carState', 'carControl', 'controlsState', 'selfdriveState', 'modelV2'], poll='modelV2')
  pm = messaging.PubMaster(['lateralManeuverPlan', 'longitudinalPlan', 'driverAssistance', 'alertDebug'])

  maneuvers = iter(MANEUVERS)
  maneuver = None

  while True:
    sm.update()

    if maneuver is None:
      maneuver = next(maneuvers, None)

    alert_msg = messaging.new_message('alertDebug')
    alert_msg.valid = True

    plan_send = messaging.new_message('lateralManeuverPlan')
    plan_send.valid = sm.all_checks()

    accel = 0
    v_ego = max(sm['carState'].vEgo, 0)

    if maneuver is not None:
      accel = maneuver.get_accel(v_ego, sm['carControl'].latActive)

      if maneuver.active:
        alert_msg.alertDebug.alertText1 = f'Maneuver Active: {accel:0.2f} m/s^2'
      else:
        if maneuver._ready_cnt > 0:
          countdown = max(0, 2. - maneuver._ready_cnt * DT_MDL)
          alert_msg.alertDebug.alertText1 = f'Starting in {countdown:.1f}s'
        else:
          alert_msg.alertDebug.alertText1 = f'Set speed to {maneuver.initial_speed * CV.MS_TO_MPH:.0f} mph'
      alert_msg.alertDebug.alertText2 = f'{maneuver.description}'
    else:
      alert_msg.alertDebug.alertText1 = 'Maneuvers Finished'

    pm.send('alertDebug', alert_msg)

    plan_send.lateralManeuverPlan.desiredCurvature = accel / max(v_ego, MIN_SPEED) ** 2
    pm.send('lateralManeuverPlan', plan_send)

    long_plan_send = messaging.new_message('longitudinalPlan')
    long_plan_send.valid = sm.all_checks()
    longitudinalPlan = long_plan_send.longitudinalPlan
    long_accel = 0
    if maneuver is not None:
      long_accel = min(max(maneuver.initial_speed - v_ego, -2.), 2.)
    longitudinalPlan.aTarget = long_accel
    longitudinalPlan.shouldStop = v_ego < CP.vEgoStopping and long_accel < 1e-2
    longitudinalPlan.allowBrake = True
    longitudinalPlan.allowThrottle = True
    longitudinalPlan.hasLead = True
    longitudinalPlan.speeds = [0.2]
    pm.send('longitudinalPlan', long_plan_send)

    assistance_send = messaging.new_message('driverAssistance')
    assistance_send.valid = True
    pm.send('driverAssistance', assistance_send)

    if maneuver is not None and maneuver.finished:
      maneuver = None


if __name__ == "__main__":
  main()
