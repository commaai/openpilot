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


class Estimators:
  def __init__(self, params: Params, CP: car.CarParams, replay: bool = False, debug: bool = False):
    self.params = params
    self.debug = debug

    steer_ratio, stiffness_factor, angle_offset_deg, p_initial = retrieve_initial_vehicle_params(params, CP, replay, debug)
    self.params_estimator = VehicleParamsEstimator(CP, steer_ratio, stiffness_factor, np.radians(angle_offset_deg), p_initial)

    self.lag_estimator = LateralLagEstimator(CP, 1. / SERVICE_LIST['deviceMotion'].frequency)
    if (initial_lag_params := retrieve_initial_lag(params, CP)) is not None:
      lag, valid_blocks = initial_lag_params
      self.lag_estimator.reset(lag, valid_blocks)

    self.torque_estimator = TorqueEstimator(CP)

  @staticmethod
  def _handle_updated(sm: messaging.SubMaster, services: list[str], estimator) -> None:
    if not sm.all_checks(services):
      return

    for which in sorted(services, key=lambda x: sm.logMonoTime[x]):
      if sm.updated[which]:
        estimator.handle_log(sm.logMonoTime[which] * 1e-9, which, sm[which])

  def update(self, sm: messaging.SubMaster, pm: messaging.PubMaster) -> None:
    params_valid = sm.all_checks(PARAMS_SERVICES)
    lag_valid = sm.all_checks(LAG_SERVICES)
    torque_valid = sm.all_checks(TORQUE_SERVICES) and lag_valid
    self._handle_updated(sm, PARAMS_SERVICES, self.params_estimator)
    self._handle_updated(sm, LAG_SERVICES, self.lag_estimator)
    if torque_valid:
      # torqued historically handled messages in SubMaster service order.
      for which in TORQUE_SERVICES:
        if sm.updated[which]:
          self.torque_estimator.handle_log(sm.logMonoTime[which] * 1e-9, which, sm[which])

    if lag_valid:
      self.lag_estimator.update_points()

    if not sm.updated['deviceMotion']:
      return

    params_msg = self.params_estimator.get_msg(params_valid, debug=self.debug)
    params_msg_dat = params_msg.to_bytes()
    if sm.frame % 1200 == 0:  # once a minute
      self.params.put('LiveParametersV2', params_msg_dat)
    pm.send('vehicleParameters', params_msg_dat)

    # The remaining estimators publish at 4 Hz, driven by deviceMotion.
    if sm.frame % 5 != 0:
      return

    self.lag_estimator.update_estimate()
    lag_msg = self.lag_estimator.get_msg(lag_valid, self.debug)
    lag_msg_dat = lag_msg.to_bytes()
    pm.send('lateralDelay', lag_msg_dat)

    # Feed the new lag directly rather than subscribing to our own publication.
    self.torque_estimator.handle_log(sm.logMonoTime['deviceMotion'] * 1e-9, 'lateralDelay', lag_msg.lateralDelay)
    pm.send('lateralTorqueParameters', self.torque_estimator.get_msg(valid=torque_valid, with_points=self.debug))

    if sm.frame % 1200 == 0:  # once a minute
      self.params.put('LiveDelay', lag_msg_dat)

    if sm.frame % 240 == 0:  # preserve torqued's cache cadence
      torque_msg = self.torque_estimator.get_msg(valid=torque_valid, with_points=True)
      self.params.put('LiveTorqueParameters', torque_msg.to_bytes())


def main() -> None:
  config_realtime_process([0, 1, 2, 3], 5)

  debug = bool(int(os.getenv('DEBUG', '0')))
  replay = bool(int(os.getenv('REPLAY', '0')))

  pm = messaging.PubMaster(['vehicleParameters', 'lateralDelay', 'lateralTorqueParameters'])
  sm = messaging.SubMaster(SUBSCRIBED_SERVICES, poll='deviceMotion')

  params = Params()
  CP = messaging.log_from_bytes(params.get('CarParams', block=True), car.CarParams)
  estimators = Estimators(params, CP, replay, debug)

  while True:
    sm.update()
    estimators.update(sm, pm)


if __name__ == '__main__':
  main()
