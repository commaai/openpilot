#!/usr/bin/env python3
import numpy as np
from collections import deque

from skimage.registration._masked_phase_cross_correlation import cross_correlate_masked

import cereal.messaging as messaging
from cereal import car
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process, DT_CTRL, DT_MDL
from openpilot.selfdrive.locationd.helpers import PoseCalibrator, Pose

MIN_LAG_VEL = 20.0
MAX_SANE_LAG = 3.0
MIN_ABS_YAW_RATE_DEG = 1
MOVING_CORR_WINDOW = 300.0
MIN_OKAY_WINDOW = 25.0
MIN_NCC = 0.95


class Samples:
  def __init__(self, maxlen):
    self.x = deque(maxlen=maxlen)
    self.t = deque(maxlen=maxlen)

  def add(self, t, x):
    self.x.append(x)
    self.t.append(t)


class BaseLagEstimator:
  def __init__(self, CP, dt, moving_corr_window, min_okay_window):
    self.dt = dt
    self.window_len = int(moving_corr_window / self.dt)
    self.min_okay_window_len = int(min_okay_window / self.dt)
    self.initial_lag = CP.steerActuatorDelay

    self.t = 0.0

    self.calibrator = PoseCalibrator()

    lag_limit = int(moving_corr_window / (self.dt * 25))
    self.lags = deque(maxlen=lag_limit)
    self.correlations = deque(maxlen=lag_limit)

    self.lat_active = Samples(int(moving_corr_window / DT_CTRL))
    self.steering_pressed = Samples(int(moving_corr_window / DT_CTRL))
    self.vego = Samples(int(moving_corr_window / DT_CTRL))
    self.curvature = Samples(int(moving_corr_window / DT_CTRL))
    self.desired_curvature = Samples(int(moving_corr_window / DT_CTRL))
    self.yaw_rate = Samples(int(moving_corr_window / DT_MDL))

  def actuator_delay(self, expected_sig, actual_sig, is_okay, dt, max_lag):
    raise NotImplementedError

  def handle_log(self, t, which, msg) -> None:
    if which == "carControl":
      self.lat_active.add(t, msg.latActive)
    elif which == "carState":
      self.steering_pressed.add(t, msg.steeringPressed)
      self.vego.add(t, msg.vEgo)
    elif which == "controlsState":
      curvature = msg.curvature
      desired_curvature = msg.desiredCurvature
      self.curvature.add(t, curvature)
      self.desired_curvature.add(t, desired_curvature)
    elif which == "livePose":
      device_pose = Pose.from_live_pose(msg)
      calibrated_pose = self.calibrator.build_calibrated_pose(device_pose)
      self.yaw_rate.add(t, calibrated_pose.angular_velocity.z)
    elif which == 'liveCalibration':
      self.calibrator.feed_live_calib(msg)

    self.t = t

  def get_msg(self, valid: bool, with_points: bool):
    if len(self.desired_curvature.x) >= self.window_len:
      times = np.arange(self.t - self.window_len * self.dt, self.t, self.dt)
      lat_active = np.interp(times, self.lat_active.t, self.lat_active.x).astype(bool)
      steering_pressed = np.interp(times, self.steering_pressed.t, self.steering_pressed.x).astype(bool)
      vego = np.interp(times, self.vego.t, self.vego.x)
      yaw_rate = np.interp(times, self.yaw_rate.t, self.yaw_rate.x)
      desired_curvature = np.interp(times, self.desired_curvature.t, self.desired_curvature.x)

      okay = lat_active & ~steering_pressed & (vego > MIN_LAG_VEL) & (np.abs(yaw_rate) > np.radians(MIN_ABS_YAW_RATE_DEG))
      if np.count_nonzero(okay) >= self.min_okay_window_len:
        lat_accel_desired = desired_curvature * vego * vego
        lat_accel_actual_loc = yaw_rate * vego

        delay, correlation = self.actuator_delay(lat_accel_desired, lat_accel_actual_loc, okay, self.dt, MAX_SANE_LAG)
        if correlation > MIN_NCC:
          self.lags.append(delay)

    if len(self.lags) > 0:
      steer_actuation_delay = np.mean(self.lags)
      is_estimated = True
    else:
      steer_actuation_delay = self.initial_lag + 0.2
      is_estimated = False

    msg = messaging.new_message('liveActuatorDelay')
    msg.valid = valid

    liveActuatorDelay = msg.liveActuatorDelay
    liveActuatorDelay.steerActuatorDelay = float(steer_actuation_delay)
    liveActuatorDelay.totalPoints = len(self.curvature.x)
    liveActuatorDelay.isEstimated = is_estimated

    return msg


class LagEstimator(BaseLagEstimator):
  def correlation_lags(self, sig_len, dt):
    return np.arange(0, sig_len) * dt

  def actuator_delay(self, expected_sig, actual_sig, is_okay, dt, max_lag):
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
      for which in sorted(sm.updated.keys(), key=lambda x: sm.logMonoTime[x]):
        if sm.updated[which]:
          t = sm.logMonoTime[which] * 1e-9
          estimator.handle_log(t, which, sm[which])

    if sm.frame % 25 == 0:
      msg = estimator.get_msg(sm.all_checks(), with_points=True)
      alert_msg = messaging.new_message('alertDebug')
      alert_msg.alertDebug.alertText1 = f"Lag estimate (fixed: {CP.steerActuatorDelay:.2f} s)"
      alert_msg.alertDebug.alertText2 = f"{msg.liveActuatorDelay.steerActuatorDelay:.2f} s ({msg.liveActuatorDelay.isEstimated})"

      pm.send('liveActuatorDelay', msg)
      pm.send('alertDebug', alert_msg)


if __name__ == "__main__":
  main()
