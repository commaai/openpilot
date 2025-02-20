#!/usr/bin/env python3
import numpy as np
from collections import deque

import cereal.messaging as messaging
from cereal import car
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process, DT_CTRL
from openpilot.selfdrive.locationd.helpers import ParameterEstimator, PoseCalibrator, Pose

MIN_LAG_VEL = 15.0
MAX_SANE_LAG = 3.0
MIN_ABS_YAW_RATE_DEG = 1
MAX_LAG_HIST_LEN_SEC = 600
MOVING_CORR_WINDOW = 300
MIN_OKAY_WINDOW = 60


class BaseLagEstimator:
  def __init__(self, CP, dt, moving_corr_window, min_okay_window):
    self.dt = dt
    self.window_len = int(moving_corr_window / self.dt)
    self.min_okay_window_len = int(min_okay_window / self.dt)
    self.initial_lag = CP.steerActuatorDelay

    self.calibrator = PoseCalibrator()

    lag_limit = int(moving_corr_window / (self.dt * 25))
    self.lags = deque(maxlen=lag_limit)
    self.correlations = deque(maxlen=lag_limit)
    self.times = deque(maxlen=int(moving_corr_window / self.dt))
    self.curvature = deque(maxlen=int(moving_corr_window / self.dt))
    self.desired_curvature = deque(maxlen=int(moving_corr_window / self.dt))
    self.okay = deque(maxlen=int(moving_corr_window / self.dt))

  def actuator_delay(self, expected_sig, actual_sig, is_okay, dt, max_lag):
    raise NotImplementedError

  def handle_log(self, t, which, msg) -> None:
    if which == "carControl":
      self.lat_active = msg.latActive
    elif which == "carState":
      self.steering_pressed = msg.steeringPressed
      self.v_ego = msg.vEgo
    elif which == "controlsState":
      curvature = msg.curvature
      desired_curvature = msg.desiredCurvature
      okay = self.lat_active and not self.steering_pressed and self.v_ego > MIN_LAG_VEL and abs(self.yaw_rate) > np.radians(MIN_ABS_YAW_RATE_DEG)
      self.times.append(t)
      self.okay.append(okay)
      self.curvature.append(curvature)
      self.desired_curvature.append(desired_curvature)
    elif which == "livePose":
      device_pose = Pose.from_live_pose(msg)
      calibrated_pose = self.calibrator.build_calibrated_pose(device_pose)

      self.yaw_rate = calibrated_pose.angular_velocity.z
    elif which == 'liveCalibration':
      self.calibrator.feed_live_calib(msg)

  def get_msg(self, valid: bool, with_points: bool):
    okay_count = np.count_nonzero(self.okay)
    if len(self.curvature) >= self.window_len and okay_count >= self.min_okay_window_len:
      curvature = np.array(self.curvature)
      desired_curvature = np.array(self.desired_curvature)
      okay = np.array(self.okay)
      try:
        delay_curvature, correlation = self.actuator_delay(desired_curvature, curvature, okay, self.dt, MAX_SANE_LAG)
        self.lags.append(delay_curvature)
        self.correlations.append(correlation)
      except ValueError:
        pass

    if len(self.lags) > 0:
      steer_actuation_delay = np.mean(self.lags)
      steer_correlation = np.mean(self.correlations)
      is_estimated = True
    else:
      steer_actuation_delay = self.initial_lag
      steer_correlation = np.nan
      is_estimated = False

    msg = messaging.new_message('liveActuatorDelay')
    msg.valid = valid

    liveActuatorDelay = msg.liveActuatorDelay
    liveActuatorDelay.steerActuatorDelay = steer_actuation_delay
    liveActuatorDelay.totalPoints = len(self.curvature)
    liveActuatorDelay.isEstimated = is_estimated

    if with_points:
      liveActuatorDelay.points = [p for p in zip(self.curvature, self.desired_curvature)]

    return steer_actuation_delay, steer_correlation, okay_count, is_estimated


class LagEstimator(ParameterEstimator):
  def correlation_lags(self, sig_len, dt):
    return np.arange(0, sig_len) * dt

  def actuator_delay(self, expected_sig, actual_sig, is_okay, dt, max_lag):
    from skimage.registration._masked_phase_cross_correlation import cross_correlate_masked
    # masked (gated) normalized cross-correlation
    # normalized, can be used for anything, like comparsion

    assert len(expected_sig) == len(actual_sig)

    xcorr = cross_correlate_masked(actual_sig, expected_sig, is_okay, is_okay, axes=tuple(range(actual_sig.ndim)),)
    lags = self.correlation_lags(len(expected_sig), dt)

    n_frames_max_delay = int(max_lag / dt)
    xcorr = xcorr[len(expected_sig) - 1: len(expected_sig) - 1 + n_frames_max_delay]
    lags = lags[:n_frames_max_delay]

    max_corr_index = np.argmax(xcorr)

    lag, corr = lags[max_corr_index], xcorr[max_corr_index]
    return lag, corr


def main():
  config_realtime_process([0, 1, 2, 3], 5)

  pm = messaging.PubMaster(['liveActuatorDelay', 'alertDebug'])
  sm = messaging.SubMaster(['livePose', 'liveCalibration', 'carControl', 'carState', 'controlsState'], poll='controlsState')

  params = Params()
  CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)
  estimator = LagEstimator(CP, DT_CTRL, MOVING_CORR_WINDOW, MIN_OKAY_WINDOW)

  while True:
    sm.update()
    if sm.all_checks():
      for which in sm.services:
        if sm.updated[which]:
          t = sm.logMonoTime[which] * 1e-9
          estimator.handle_log(t, which, sm[which])

    if sm.frame % 25 == 0:
      msg = estimator.get_msg(sm.all_checks(), with_points=True)
      alert_msg = messaging.new_message('alertDebug')
      alert_msg.alertDebug.alertText1 = f"Lag estimate (fixed: {CP.steerActuatorDelay:.2f} s)"
      alert_msg.alertDebug.alertText2 = f"{msg.liveActuatorDelay.steerActuatorDelay:.2f} s"

      pm.send('liveActuatorDelay', msg)
      pm.send('alertDebug', alert_msg)


if __name__ == "__main__":
  main()
