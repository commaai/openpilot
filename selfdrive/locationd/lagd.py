#!/usr/bin/env python3
import os

import cereal.messaging as messaging
from cereal import car, log
from cereal.services import SERVICE_LIST
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.locationd.estimators.lateral_lag import LateralLagEstimator


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

  pm = messaging.PubMaster(['liveDelay'])
  sm = messaging.SubMaster(['livePose', 'liveCalibration', 'carState', 'controlsState', 'carControl'], poll='livePose')

  params_reader = Params()
  CP = messaging.log_from_bytes(params_reader.get("CarParams", block=True), car.CarParams)

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
          lag_learner.handle_log(t, which, sm[which])
      lag_learner.update_points()

    # 4Hz driven by livePose
    if sm.frame % 5 == 0:
      lag_learner.update_estimate()
      lag_msg = lag_learner.get_msg(sm.all_checks(), DEBUG)
      lag_msg_dat = lag_msg.to_bytes()
      pm.send('liveDelay', lag_msg_dat)

      if sm.frame % 1200 == 0: # cache every 60 seconds
        params_reader.put_nonblocking("LiveLag", lag_msg_dat)
