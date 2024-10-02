#!/usr/bin/env python3
from dataclasses import dataclass

from cereal import messaging, car
from opendbc.car.common.conversions import Conversions as CV
from openpilot.common.realtime import DT_MDL
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog


@dataclass
class Action:
  accel: float      # m/s^2
  duration: float   # seconds


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

  def get_accel(self, v_ego: float, long_active: bool, standstill: bool, cruise_standstill: bool) -> float:
    ready = abs(v_ego - self.initial_speed) < 0.3 and long_active and not cruise_standstill
    if self.initial_speed < 0.01:
      ready = ready and standstill
    self._ready_cnt = (self._ready_cnt + 1) if ready else 0

    if self._ready_cnt > (3. / DT_MDL):
      self._active = True

    if not self._active:
      return min(max(self.initial_speed - v_ego, -2.), 2.)

    action = self.actions[self._action_index]

    self._action_frames += 1

    # reached duration of action
    if self._action_frames > (action.duration / DT_MDL):
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

    return action.accel

  @property
  def finished(self):
    return self._finished

  @property
  def active(self):
    return self._active


MANEUVERS = [
  Maneuver(
    "come to stop",
    [Action(-0.5, 12)],
    repeat=2,
    initial_speed=5.,
  ),
  Maneuver(
   "start from stop",
   [Action(1.5, 5)],
   repeat=3,
   initial_speed=0.,
  ),
  Maneuver(
   "creep: alternate between +1m/s^2 and -1m/s^2",
   [
     Action(1, 3), Action(-1, 3),
     Action(1, 3), Action(-1, 3),
     Action(1, 3), Action(-1, 3),
   ],
   repeat=2,
   initial_speed=0.,
  ),
  Maneuver(
    "brake step response: -1m/s^2 from 20mph",
    [Action(-1, 3)],
    repeat=2,
    initial_speed=20. * CV.MPH_TO_MS,
  ),
  Maneuver(
    "brake step response: -4m/s^2 from 20mph",
    [Action(-4, 3)],
    repeat=2,
    initial_speed=20. * CV.MPH_TO_MS,
  ),
  Maneuver(
    "gas step response: +1m/s^2 from 20mph",
    [Action(1, 3)],
    repeat=2,
    initial_speed=20. * CV.MPH_TO_MS,
  ),
  Maneuver(
    "gas step response: +4m/s^2 from 20mph",
    [Action(4, 3)],
    repeat=2,
    initial_speed=20. * CV.MPH_TO_MS,
  ),
]


def main():
  params = Params()
  cloudlog.info("joystickd is waiting for CarParams")
  CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)

  sm = messaging.SubMaster(['carState', 'carControl', 'controlsState', 'selfdriveState', 'modelV2'], poll='modelV2')
  pm = messaging.PubMaster(['longitudinalPlan', 'driverAssistance', 'alertDebug'])

  maneuvers = iter(MANEUVERS)
  maneuver = None

  while True:
    sm.update()

    if maneuver is None:
      maneuver = next(maneuvers, None)

    alert_msg = messaging.new_message('alertDebug')
    alert_msg.valid = True

    plan_send = messaging.new_message('longitudinalPlan')
    plan_send.valid = sm.all_checks()

    longitudinalPlan = plan_send.longitudinalPlan
    accel = 0
    v_ego = max(sm['carState'].vEgo, 0)

    if maneuver is not None:
      accel = maneuver.get_accel(v_ego, sm['carControl'].longActive, sm['carState'].standstill, sm['carState'].cruiseState.standstill)

      if maneuver.active:
        alert_msg.alertDebug.alertText1 = f'Maneuver Active: {accel:0.2f} m/s^2'
      else:
        alert_msg.alertDebug.alertText1 = f'Setting up to {maneuver.initial_speed * CV.MS_TO_MPH:0.2f} mph'
      alert_msg.alertDebug.alertText2 = f'{maneuver.description}'
    else:
      alert_msg.alertDebug.alertText1 = 'Maneuvers Finished'

    pm.send('alertDebug', alert_msg)

    longitudinalPlan.aTarget = accel
    longitudinalPlan.shouldStop = v_ego < CP.vEgoStopping and accel < 1e-2

    longitudinalPlan.allowBrake = True
    longitudinalPlan.allowThrottle = True
    longitudinalPlan.hasLead = True

    pm.send('longitudinalPlan', plan_send)

    assistance_send = messaging.new_message('driverAssistance')
    assistance_send.valid = True
    pm.send('driverAssistance', assistance_send)

    if maneuver is not None and maneuver.finished:
      maneuver = None
