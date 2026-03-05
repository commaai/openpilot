#!/usr/bin/env python3
import numpy as np
from dataclasses import dataclass

from cereal import messaging, car
from openpilot.common.constants import CV
from openpilot.common.realtime import DT_MDL
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog


@dataclass
class Action:
  curvature_bp: list[float]  # 1/m
  time_bp: list[float]       # seconds

  def __post_init__(self):
    assert len(self.curvature_bp) == len(self.time_bp)


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

  def get_curvature(self, v_ego: float, lat_active: bool) -> float:
    ready = abs(v_ego - self.initial_speed) < 0.3 and lat_active
    self._ready_cnt = (self._ready_cnt + 1) if ready else 0

    if self._ready_cnt > (3. / DT_MDL):
      self._active = True

    if not self._active:
      return 0.

    action = self.actions[self._action_index]
    action_curvature = np.interp(self._action_frames * DT_MDL, action.time_bp, action.curvature_bp)

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

    return float(action_curvature)

  @property
  def finished(self):
    return self._finished

  @property
  def active(self):
    return self._active


def _accel_to_curv(accel, speed_mph):
  """Convert lateral accel (m/s^2) to curvature (1/m) at a given speed."""
  v = speed_mph * CV.MPH_TO_MS
  return accel / v ** 2


MANEUVERS = [
  # Step maneuvers: 0.3 m/s^2 lateral accel converted to curvature at target speed
  Maneuver(
    "step up 30mph a=0.3",
    [Action([0], [1.0]), Action([_accel_to_curv(0.3, 30)], [3.0])],
    repeat=2,
    initial_speed=30. * CV.MPH_TO_MS,
  ),
  Maneuver(
    "step down 30mph a=0.3",
    [Action([_accel_to_curv(0.3, 30)], [2.0]), Action([0], [3.0])],
    repeat=2,
    initial_speed=30. * CV.MPH_TO_MS,
  ),
  Maneuver(
    "step up 50mph a=0.3",
    [Action([0], [1.0]), Action([_accel_to_curv(0.3, 50)], [3.0])],
    repeat=2,
    initial_speed=50. * CV.MPH_TO_MS,
  ),
  Maneuver(
    "step down 50mph a=0.3",
    [Action([_accel_to_curv(0.3, 50)], [2.0]), Action([0], [3.0])],
    repeat=2,
    initial_speed=50. * CV.MPH_TO_MS,
  ),
  Maneuver(
    "step up 70mph a=0.3",
    [Action([0], [1.0]), Action([_accel_to_curv(0.3, 70)], [3.0])],
    repeat=2,
    initial_speed=70. * CV.MPH_TO_MS,
  ),
  Maneuver(
    "step down 70mph a=0.3",
    [Action([_accel_to_curv(0.3, 70)], [2.0]), Action([0], [3.0])],
    repeat=2,
    initial_speed=70. * CV.MPH_TO_MS,
  ),

  # Dynamic maneuvers from lateral-maneuver branch
  Maneuver(
    "S-curve weave",
    [
      Action([0.0, 0.02], [0.0, 1.0]),
      Action([0.02, 0.02], [0.0, 1.5]),
      Action([0.02, -0.02], [0.0, 2.5]),
      Action([-0.02, -0.02], [0.0, 1.5]),
      Action([-0.02, 0.0], [0.0, 1.0]),
    ],
    repeat=2,
    initial_speed=25. * CV.MPH_TO_MS,
  ),
  Maneuver(
    "steady right curve",
    [
      Action([0.0, 0.015], [0.0, 1.5]),
      Action([0.015, 0.015], [0.0, 3.0]),
      Action([0.015, 0.0], [0.0, 1.5]),
    ],
    repeat=2,
    initial_speed=25. * CV.MPH_TO_MS,
  ),
  Maneuver(
    "double lane change",
    [
      Action([0.0, 0.015], [0.0, 0.8]),
      Action([0.015, 0.015], [0.0, 1.2]),
      Action([0.015, -0.015], [0.0, 1.5]),
      Action([-0.015, -0.015], [0.0, 1.2]),
      Action([-0.015, 0.0], [0.0, 0.8]),
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

    curvature = 0
    v_ego = max(sm['carState'].vEgo, 0)

    if maneuver is not None:
      curvature = maneuver.get_curvature(v_ego, sm['carControl'].latActive)

      if maneuver.active:
        alert_msg.alertDebug.alertText1 = f'Maneuver Active: {curvature:0.4f} 1/m'
      else:
        if maneuver._ready_cnt > 0:
          countdown = max(0, 3. - maneuver._ready_cnt * DT_MDL)
          alert_msg.alertDebug.alertText1 = f'Starting in {countdown:.1f}s'
        else:
          alert_msg.alertDebug.alertText1 = f'Set speed to {maneuver.initial_speed * CV.MS_TO_MPH:.0f} mph'
      alert_msg.alertDebug.alertText2 = f'{maneuver.description}'
    else:
      alert_msg.alertDebug.alertText1 = 'Maneuvers Finished'

    pm.send('alertDebug', alert_msg)

    plan_send.lateralManeuverPlan.desiredCurvature = curvature
    pm.send('lateralManeuverPlan', plan_send)

    long_plan_send = messaging.new_message('longitudinalPlan')
    long_plan_send.valid = sm.all_checks()
    longitudinalPlan = long_plan_send.longitudinalPlan
    accel = 0
    if maneuver is not None:
      accel = min(max(maneuver.initial_speed - v_ego, -2.), 2.)
    longitudinalPlan.aTarget = accel
    longitudinalPlan.shouldStop = v_ego < CP.vEgoStopping and accel < 1e-2
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
