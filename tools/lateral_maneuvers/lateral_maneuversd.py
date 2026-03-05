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
  _baseline_curvature: float = 0.0

  def get_accel(self, v_ego: float, lat_active: bool, curvature: float) -> float:
    ready = abs(v_ego - self.initial_speed) < 1.0 and lat_active and abs(curvature) < 0.001
    self._ready_cnt = (self._ready_cnt + 1) if ready else 0

    if self._ready_cnt > (3. / DT_MDL):
      if not self._active:
        self._baseline_curvature = curvature
      self._active = True

    if not self._active:
      return 0.0

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
        self._ready_cnt = 0
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


def _sine_action(amplitude, period, duration):
  t = np.linspace(0, duration, int(duration / DT_MDL) + 1)
  a = amplitude * np.sin(2 * np.pi * t / period)
  return Action(a.tolist(), t.tolist())


MANEUVERS = [
  Maneuver(
    "step right 30mph",
    [Action([0.5], [1.0]), Action([-0.5], [1.0])],
    repeat=2,
    initial_speed=30. * CV.MPH_TO_MS,
  ),
  Maneuver(
    "step left 30mph",
    [Action([-0.5], [1.0]), Action([0.5], [1.0])],
    repeat=2,
    initial_speed=30. * CV.MPH_TO_MS,
  ),
  Maneuver(
    "sine 30mph 1Hz",
    [_sine_action(1.0, 1.0, 1.0)],
    repeat=1,
    initial_speed=30. * CV.MPH_TO_MS,
  ),
  Maneuver(
    "sine 30mph 0.25Hz",
    [_sine_action(0.5, 4.0, 4.0)],
    repeat=1,
    initial_speed=30. * CV.MPH_TO_MS,
  ),
  Maneuver(
    "step right 40mph",
    [Action([0.5], [1.0]), Action([-0.5], [1.0])],
    repeat=2,
    initial_speed=40. * CV.MPH_TO_MS,
  ),
  Maneuver(
    "step left 40mph",
    [Action([-0.5], [1.0]), Action([0.5], [1.0])],
    repeat=2,
    initial_speed=40. * CV.MPH_TO_MS,
  ),
  Maneuver(
    "sine 40mph 1Hz",
    [_sine_action(1.0, 1.0, 1.0)],
    repeat=1,
    initial_speed=40. * CV.MPH_TO_MS,
  ),
  Maneuver(
    "sine 40mph 0.25Hz",
    [_sine_action(0.5, 4.0, 4.0)],
    repeat=1,
    initial_speed=40. * CV.MPH_TO_MS,
  ),
]


def main():
  params = Params()
  cloudlog.info("lateral_maneuversd is waiting for CarParams")
  messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)

  sm = messaging.SubMaster(['carState', 'carControl', 'controlsState', 'selfdriveState', 'modelV2'], poll='modelV2')
  pm = messaging.PubMaster(['lateralManeuverPlan', 'alertDebug'])

  maneuvers = iter(MANEUVERS)
  maneuver = None

  while True:
    sm.update()

    if maneuver is None:
      maneuver = next(maneuvers, None)

    alert_msg = messaging.new_message('alertDebug')
    alert_msg.valid = True

    plan_send = messaging.new_message('lateralManeuverPlan')

    accel = 0
    v_ego = max(sm['carState'].vEgo, 0)
    cur_curvature = sm['controlsState'].curvature

    if maneuver is not None:
      # abort active maneuver on steering override
      if maneuver.active and sm['carState'].steeringPressed:
        maneuver._active = False
        maneuver._action_frames = 0
        maneuver._action_index = 0
        maneuver._ready_cnt = 0

      accel = maneuver.get_accel(v_ego, sm['carControl'].latActive, cur_curvature)

      if maneuver.active:
        action_remaining = maneuver.actions[maneuver._action_index].time_bp[-1] - maneuver._action_frames * DT_MDL
        if maneuver.description.startswith('sine'):
          freq = maneuver.description.split()[-1]
          alert_msg.alertDebug.alertText1 = f'Active sine {freq} {max(action_remaining, 0):.1f}s'
        else:
          alert_msg.alertDebug.alertText1 = f'Active {accel:+.1f}m/s² {max(action_remaining, 0):.1f}s'
        alert_msg.alertDebug.alertText2 = maneuver.description
      elif not (abs(v_ego - maneuver.initial_speed) < 1.0 and sm['carControl'].latActive):
        alert_msg.alertDebug.alertText1 = f'Set speed to {maneuver.initial_speed * CV.MS_TO_MPH:0.0f} mph'
      else:
        ready_time = max(3. - maneuver._ready_cnt * DT_MDL, 0)
        alert_msg.alertDebug.alertText1 = 'Go straight'
        alert_msg.alertDebug.alertText2 = f'{ready_time:0.1f}s'
    else:
      alert_msg.alertDebug.alertText1 = 'Maneuvers Finished'

    pm.send('alertDebug', alert_msg)

    plan_send.valid = maneuver is not None and maneuver.active
    if plan_send.valid:
      plan_send.lateralManeuverPlan.desiredCurvature = maneuver._baseline_curvature + accel / max(v_ego, MIN_SPEED) ** 2
    pm.send('lateralManeuverPlan', plan_send)

    if maneuver is not None and maneuver.finished:
      maneuver = None


if __name__ == "__main__":
  main()
