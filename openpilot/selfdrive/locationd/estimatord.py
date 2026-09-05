#!/usr/bin/env python3
import os
import numpy as np

import openpilot.cereal.messaging as messaging
from openpilot.cereal.services import SERVICE_LIST
from opendbc.car.structs import car
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process
from openpilot.selfdrive.locationd.lagd import LateralLagEstimator, retrieve_initial_lag
from openpilot.selfdrive.locationd.paramsd import VehicleParamsEstimator, retrieve_initial_vehicle_params
from openpilot.selfdrive.locationd.torqued import TorqueEstimator


PARAMS_SERVICES = ['deviceMotion', 'extrinsicsCalibration', 'carState']
LAG_SERVICES = ['deviceMotion', 'extrinsicsCalibration', 'carState', 'controlsState', 'carControl']
TORQUE_SERVICES = ['carControl', 'carOutput', 'carState', 'extrinsicsCalibration', 'deviceMotion']
SUBSCRIBED_SERVICES = list(dict.fromkeys(PARAMS_SERVICES + LAG_SERVICES + TORQUE_SERVICES))


def main() -> None:
  config_realtime_process([0, 1, 2, 3], 5)

  debug = bool(int(os.getenv('DEBUG', '0')))
  replay = bool(int(os.getenv('REPLAY', '0')))

  pm = messaging.PubMaster(['vehicleParameters', 'lateralDelay', 'lateralTorqueParameters'])
  sm = messaging.SubMaster(SUBSCRIBED_SERVICES, poll='deviceMotion')

  params = Params()
  CP = messaging.log_from_bytes(params.get('CarParams', block=True), car.CarParams)

  steer_ratio, stiffness_factor, angle_offset_deg, p_initial = retrieve_initial_vehicle_params(params, CP, replay, debug)
  params_estimator = VehicleParamsEstimator(CP, steer_ratio, stiffness_factor, np.radians(angle_offset_deg), p_initial)

  lag_estimator = LateralLagEstimator(CP, 1. / SERVICE_LIST['deviceMotion'].frequency)
  if (initial_lag_params := retrieve_initial_lag(params, CP)) is not None:
    lag, valid_blocks = initial_lag_params
    lag_estimator.reset(lag, valid_blocks)

  torque_estimator = TorqueEstimator(CP)
  estimators = (
    (params_estimator, PARAMS_SERVICES),
    (lag_estimator, LAG_SERVICES),
    (torque_estimator, TORQUE_SERVICES),
  )

  while True:
    sm.update()
    valid = sm.all_checks()

    if valid:
      for which in sorted(sm.updated, key=lambda x: sm.logMonoTime[x]):
        if not sm.updated[which]:
          continue

        t = sm.logMonoTime[which] * 1e-9
        for estimator, services in estimators:
          if which in services:
            estimator.handle_log(t, which, sm[which])
      lag_estimator.update_points()

    if not sm.updated['deviceMotion']:
      continue

    params_msg = params_estimator.get_msg(valid, debug=debug)
    params_msg_dat = params_msg.to_bytes()
    if sm.frame % 1200 == 0:  # once a minute
      params.put('LiveParametersV2', params_msg_dat)
    pm.send('vehicleParameters', params_msg_dat)

    # The remaining estimators publish at 4 Hz, driven by deviceMotion.
    if sm.frame % 5 != 0:
      continue

    lag_estimator.update_estimate()
    lag_msg = lag_estimator.get_msg(valid, debug)
    lag_msg_dat = lag_msg.to_bytes()
    pm.send('lateralDelay', lag_msg_dat)

    # Feed the new lag directly rather than subscribing to our own publication.
    torque_estimator.handle_log(sm.logMonoTime['deviceMotion'] * 1e-9, 'lateralDelay', lag_msg.lateralDelay)
    pm.send('lateralTorqueParameters', torque_estimator.get_msg(valid=valid, with_points=debug))

    if sm.frame % 1200 == 0:  # once a minute
      params.put('LiveDelay', lag_msg_dat)

    if sm.frame % 240 == 0:  # preserve torqued's cache cadence
      torque_msg = torque_estimator.get_msg(valid=valid, with_points=True)
      params.put('LiveTorqueParameters', torque_msg.to_bytes())


if __name__ == '__main__':
  main()
