#!/usr/bin/env python3
import os
import json
import numpy as np

import cereal.messaging as messaging
from cereal import car, log
from cereal.services import SERVICE_LIST
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.locationd.estimators.vehicle_params import VehicleParamsLearner
from openpilot.selfdrive.locationd.estimators.lateral_lag import LateralLagEstimator


# TODO: Remove this function after few releases (added in 0.9.9)
def migrate_cached_vehicle_params_if_needed(params_reader: Params):
  last_parameters_data = params_reader.get("LiveParameters")
  if last_parameters_data is None:
    return

  try:
    last_parameters_dict = json.loads(last_parameters_data)
    last_parameters_msg = messaging.new_message('liveParameters')
    last_parameters_msg.liveParameters.valid = True
    last_parameters_msg.liveParameters.steerRatio = last_parameters_dict['steerRatio']
    last_parameters_msg.liveParameters.stiffnessFactor = last_parameters_dict['stiffnessFactor']
    last_parameters_msg.liveParameters.angleOffsetAverageDeg = last_parameters_dict['angleOffsetAverageDeg']
    params_reader.put("LiveParameters", last_parameters_msg.to_bytes())
  except Exception:
    pass


def retrieve_initial_vehicle_params(params_reader: Params, CP: car.CarParams, replay: bool, debug: bool):
  last_parameters_data = params_reader.get("LiveParameters")
  last_carparams_data = params_reader.get("CarParamsPrevRoute")

  steer_ratio, stiffness_factor, angle_offset_deg, p_initial = CP.steerRatio, 1.0, 0.0, None

  retrieve_success = False
  if last_parameters_data is not None and last_carparams_data is not None:
    try:
      with log.Event.from_bytes(last_parameters_data) as last_lp_msg, car.CarParams.from_bytes(last_carparams_data) as last_CP:
        lp = last_lp_msg.liveParameters
        # Check if car model matches
        if last_CP.carFingerprint != CP.carFingerprint:
          raise Exception("Car model mismatch")

        # Check if starting values are sane
        min_sr, max_sr = 0.5 * CP.steerRatio, 2.0 * CP.steerRatio
        steer_ratio_sane = min_sr <= lp.steerRatio <= max_sr
        if not steer_ratio_sane:
          raise Exception(f"Invalid starting values found {lp}")

        initial_filter_std = np.array(lp.debugFilterState.std)
        if debug and len(initial_filter_std) != 0:
          p_initial = np.diag(initial_filter_std)

        steer_ratio, stiffness_factor, angle_offset_deg = lp.steerRatio, lp.stiffnessFactor, lp.angleOffsetAverageDeg
        retrieve_success = True
    except Exception as e:
      cloudlog.error(f"Failed to retrieve initial values: {e}")

  if not replay:
    # When driving in wet conditions the stiffness can go down, and then be too low on the next drive
    # Without a way to detect this we have to reset the stiffness every drive
    stiffness_factor = 1.0

  if not retrieve_success:
    cloudlog.info("Parameter learner resetting to default values")

  return steer_ratio, stiffness_factor, angle_offset_deg, p_initial


def retrieve_initial_lag(params_reader: Params, CP: car.CarParams):
  last_lag_data = params_reader.get("LiveLag")
  last_carparams_data = params_reader.get("CarParamsPrevRoute")

  if last_lag_data is not None:
    try:
      with log.Event.from_bytes(last_lag_data) as last_lag_msg, car.CarParams.from_bytes(last_carparams_data) as last_CP:
        ld = last_lag_msg.liveDelay
        if last_CP.carFingerprint != CP.carFingerprint:
          raise Exception("Car model mismatch")

        lag, valid_blocks = ld.lateralDelayEstimate, ld.validBlocks
        return lag, valid_blocks
    except Exception as e:
      cloudlog.error(f"Failed to retrieve initial lag: {e}")

  return None


def main():
  config_realtime_process([0, 1, 2, 3], 5)

  DEBUG = bool(int(os.getenv("DEBUG", "0")))
  REPLAY = bool(int(os.getenv("REPLAY", "0")))

  pm = messaging.PubMaster(['liveParameters', 'liveDelay'])
  sm = messaging.SubMaster(['livePose', 'liveCalibration', 'carState', 'controlsState', 'carControl'], poll='livePose')

  params_reader = Params()
  CP = messaging.log_from_bytes(params_reader.get("CarParams", block=True), car.CarParams)

  migrate_cached_vehicle_params_if_needed(params_reader)

  steer_ratio, stiffness_factor, angle_offset_deg, p_initial = retrieve_initial_vehicle_params(params_reader, CP, REPLAY, DEBUG)
  params_learner = VehicleParamsLearner(CP, steer_ratio, stiffness_factor, np.radians(angle_offset_deg), p_initial)

  lag_learner = LateralLagEstimator(CP, 1. / SERVICE_LIST['livePose'].frequency)
  if (initial_lag_params := retrieve_initial_lag(params_reader, CP)) is not None:
    lag, valid_blocks = initial_lag_params
    lag_learner.reset(lag, valid_blocks)

  while True:
    sm.update()
    if sm.all_checks():
      for which in sorted(sm.updated.keys(), key=lambda x: sm.logMonoTime[x]):
        if sm.updated[which]:
          t = sm.logMonoTime[which] * 1e-9
          if which in params_learner.inputs:
            params_learner.handle_log(t, which, sm[which])
          if which in lag_learner.inputs:
            lag_learner.handle_log(t, which, sm[which])
      lag_learner.update_points()

    params_msg_dat, lag_msg_dat = None, None
    if sm.updated['livePose']:
      params_msg = params_learner.get_msg(sm.all_checks(), debug=DEBUG)
      params_msg_dat = params_msg.to_bytes()
      pm.send('liveParameters', params_msg_dat)

    # 4Hz driven by livePose
    if sm.frame % 5 == 0:
      lag_learner.update_estimate()
      lag_msg = lag_learner.get_msg(sm.all_checks(), DEBUG)
      lag_msg_dat = lag_msg.to_bytes()
      pm.send('liveDelay', lag_msg_dat)

    if sm.frame % 1200 == 0: # cache every 60 seconds
      if params_msg_dat is not None:
        params_reader.put_nonblocking("LiveParameters", params_msg_dat)
      if lag_msg_dat is not None:
        params_reader.put_nonblocking("LiveLag", lag_msg_dat)

if __name__ == "__main__":
  main()
