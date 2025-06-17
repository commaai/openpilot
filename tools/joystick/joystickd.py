#!/usr/bin/env python3

import math
import numpy as np

from cereal import messaging, car
from opendbc.car.vehicle_model import VehicleModel
from openpilot.common.realtime import DT_CTRL, Ratekeeper
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

LongCtrlState = car.CarControl.Actuators.LongControlState
MAX_LAT_ACCEL = 3.0


def joystickd_thread():
  params = Params()
  cloudlog.info("joystickd is waiting for CarParams")
  CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)
  VM = VehicleModel(CP)

  sm = messaging.SubMaster(['carState', 'onroadEvents', 'liveParameters', 'selfdriveState', 'testJoystick'], frequency=1. / DT_CTRL)
  pm = messaging.PubMaster(['carControl', 'controlsState'])

  rk = Ratekeeper(100, print_delay_threshold=None)
  while 1:
    sm.update(0)

    cc_msg = messaging.new_message('carControl')
    cc_msg.valid = True
    CC = cc_msg.carControl
    CC.enabled = sm['selfdriveState'].enabled
    CC.latActive = sm['selfdriveState'].active and not sm['carState'].steerFaultTemporary and not sm['carState'].steerFaultPermanent
    CC.longActive = CC.enabled and not any(e.overrideLongitudinal for e in sm['onroadEvents']) and CP.openpilotLongitudinalControl
    CC.cruiseControl.cancel = sm['carState'].cruiseState.enabled and (not CC.enabled or not CP.pcmCruise)
    CC.hudControl.leadDistanceBars = 2

    actuators = CC.actuators

    # reset joystick if it hasn't been received in a while
    should_reset_joystick = sm.recv_frame['testJoystick'] == 0 or (sm.frame - sm.recv_frame['testJoystick'])*DT_CTRL > 0.2

    if not should_reset_joystick:
      joystick_axes = sm['testJoystick'].axes
    else:
      joystick_axes = [0.0, 0.0]

    if CC.longActive:
      actuators.accel = 4.0 * float(np.clip(joystick_axes[0], -1, 1))
      actuators.longControlState = LongCtrlState.pid if sm['carState'].vEgo > CP.vEgoStopping else LongCtrlState.stopping

    if CC.latActive:
      max_curvature = MAX_LAT_ACCEL / max(sm['carState'].vEgo ** 2, 5)
      max_angle = math.degrees(VM.get_steer_from_curvature(max_curvature, sm['carState'].vEgo, sm['liveParameters'].roll))

      actuators.torque = float(np.clip(joystick_axes[1], -1, 1))
      actuators.steeringAngleDeg, actuators.curvature = actuators.torque * max_angle, actuators.torque * -max_curvature

    pm.send('carControl', cc_msg)

    cs_msg = messaging.new_message('controlsState')
    cs_msg.valid = True
    controlsState = cs_msg.controlsState
    controlsState.lateralControlState.init('debugState')

    lp = sm['liveParameters']
    steer_angle_without_offset = math.radians(sm['carState'].steeringAngleDeg - lp.angleOffsetDeg)
    controlsState.curvature = -VM.calc_curvature(steer_angle_without_offset, sm['carState'].vEgo, lp.roll)

    pm.send('controlsState', cs_msg)

    rk.keep_time()


def main():
  joystickd_thread()


if __name__ == "__main__":
  main()
