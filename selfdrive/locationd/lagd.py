#!/usr/bin/env python3
import numpy as np
from collections import deque

import cereal.messaging as messaging
from cereal import car
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process, DT_CTRL
from openpilot.selfdrive.locationd.helpers import ParameterEstimator

MIN_LAG_VEL = 15.0
MAX_SANE_LAG = 3.0
MIN_HIST_LEN_SEC = 60
MAX_LAG_HIST_LEN_SEC = 300
MOVING_CORR_WINDOW = 60
OVERLAP_FACTOR = 0.25


class LagEstimator(ParameterEstimator):
  def __init__(self, CP, dt, min_hist_len_sec, max_lag_hist_len_sec, moving_corr_window, overlap_factor):
    self.dt = dt
    self.min_hist_len = int(min_hist_len_sec / self.dt)
    self.window_len = int(moving_corr_window / self.dt)
    self.initial_lag = CP.steerActuatorDelay
    self.current_lag = self.initial_lag

    self.lat_active = False
    self.steering_pressed = False
    self.v_ego = 0.0
    self.lags = deque(maxlen= int(max_lag_hist_len_sec / (moving_corr_window * overlap_factor)))
    self.curvature = deque(maxlen=int(moving_corr_window / self.dt))
    self.desired_curvature = deque(maxlen=int(moving_corr_window / self.dt))
    self.frame = 0

  def correlation_lags(self, sig_len, dt):
    return np.arange(0, sig_len) * dt

  def actuator_delay(self, expected_sig, actual_sig, dt, max_lag=MAX_SANE_LAG):
    assert len(expected_sig) == len(actual_sig)
    correlations = np.correlate(expected_sig, actual_sig, mode='full')
    lags = self.correlation_lags(len(expected_sig), dt)

    # only consider negative time shifts within the max_lag
    n_frames_max_delay = int(max_lag / dt)
    correlations = correlations[len(expected_sig) - 1: len(expected_sig) - 1 + n_frames_max_delay]
    lags = lags[:n_frames_max_delay]

    max_corr_index = np.argmax(correlations)

    lag, corr = lags[max_corr_index], correlations[max_corr_index]
    return lag, corr

  def handle_log(self, t, which, msg) -> None:
    if which == "carControl":
      self.lat_active = msg.latActive
    elif which == "carState":
      self.steering_pressed = msg.steeringPressed
      self.v_ego = msg.vEgo
    elif which == "controlsState":
      curvature = msg.curvature
      desired_curvature = msg.desiredCurvature
      if self.lat_active and not self.steering_pressed:
        self.curvature.append((t, curvature))
        self.desired_curvature.append((t, desired_curvature))
    self.frame += 1

  def get_msg(self, valid: bool, with_points: bool):
    if len(self.curvature) >= self.min_hist_len:
      if self.frame % int(self.window_len * OVERLAP_FACTOR) == 0:
        _, curvature = zip(*self.curvature)
        _, desired_curvature = zip(*self.desired_curvature)
        delay_curvature, _ = self.actuator_delay(curvature, desired_curvature, self.dt)
        if delay_curvature != 0.0:
          self.lags.append(delay_curvature)
      # FIXME: this is fragile and ugly, refactor this
      if len(self.lags) > 0:
        steer_actuation_delay = float(np.mean(self.lags))
      else:
        steer_actuation_delay = self.initial_lag
    else:
      steer_actuation_delay = self.initial_lag

    msg = messaging.new_message('liveActuatorDelay')
    msg.valid = valid

    liveActuatorDelay = msg.liveActuatorDelay
    liveActuatorDelay.steerActuatorDelay = steer_actuation_delay
    liveActuatorDelay.totalPoints = len(self.curvature)

    if with_points:
      liveActuatorDelay.points = [[c, dc] for ((_, c), (_, dc)) in zip(self.curvature, self.desired_curvature)]

    return msg


def main():
  config_realtime_process([0, 1, 2, 3], 5)

  pm = messaging.PubMaster(['liveActuatorDelay', 'alertDebug'])
  sm = messaging.SubMaster(['carControl', 'carState', 'controlsState'], poll='controlsState')

  params = Params()
  CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)
  estimator = LagEstimator(CP, DT_CTRL, MIN_HIST_LEN_SEC, MAX_LAG_HIST_LEN_SEC, MOVING_CORR_WINDOW, OVERLAP_FACTOR)

  while True:
    sm.update()
    if sm.all_checks():
      for which in sm.updated.keys():
        if sm.updated[which]:
          t = sm.logMonoTime[which] * 1e-9
          estimator.handle_log(t, which, sm[which])

    if sm.frame % 25 == 0:
      msg = estimator.get_msg(sm.all_checks(), with_points=True)
      alert_msg = messaging.new_message('alertDebug')
      alert_msg.alertDebug.alertText1 = f"Lag estimateb (fixed: {CP.steerActuatorDelay:.2f} s)"
      alert_msg.alertDebug.alertText2 = f"{msg.liveActuatorDelay.steerActuatorDelay:.2f} s"

      pm.send('liveActuatorDelay', msg)
      pm.send('alertDebug', alert_msg)


if __name__ == "__main__":
  main()
