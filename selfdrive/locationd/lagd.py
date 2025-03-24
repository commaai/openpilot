#!/usr/bin/env python3
import numpy as np
from collections import deque

from skimage.registration._masked_phase_cross_correlation import cross_correlate_masked

import cereal.messaging as messaging
from cereal import car
from cereal.services import SERVICE_LIST
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process, DT_CTRL
from openpilot.selfdrive.locationd.helpers import PoseCalibrator, Pose

MIN_LAG_VEL = 20.0
MAX_SANE_LAG = 3.0
MIN_ABS_YAW_RATE_DEG = 1
MOVING_CORR_WINDOW = 300.0
MIN_OKAY_WINDOW = 25.0
MIN_NCC = 0.95


class Points:
  def __init__(self, num_points):
    self.times = deque(maxlen=num_points)
    self.okay = deque(maxlen=num_points)
    self.desired = deque(maxlen=num_points)
    self.actual = deque(maxlen=num_points)

  @property
  def num_points(self):
    return len(self.desired)

  @property
  def num_okay(self):
    return np.count_nonzero(self.okay)

  def update(self, t, desired, actual, okay):
    self.times.append(t)
    self.okay.append(okay)
    self.desired.append(desired)
    self.actual.append(actual)

  def get(self):
    return np.array(self.times), np.array(self.desired), np.array(self.actual), np.array(self.okay)


class BlockAverage:
  def __init__(self, num_blocks, block_size, initial_value):
    self.num_blocks = num_blocks
    self.block_size = block_size
    self.block_idx = 0
    self.idx = 0

    self.values = np.tile(initial_value, (num_blocks, 1))
    self.valid_blocks = 0

  def update(self, value):
    self.values[self.block_idx] = (self.idx * self.values[self.block_idx] + (self.block_size - self.idx) * value) / self.block_size
    self.idx = (self.idx + 1) % self.block_size
    if self.idx == 0:
      self.block_idx = (self.block_idx + 1) % self.num_blocks
      self.valid_blocks = min(self.valid_blocks + 1, self.num_blocks)

  def get(self):
    valid_block_idx = [i for i in range(self.valid_blocks) if i != self.block_idx]
    if not valid_block_idx:
      return None
    return np.mean(self.values[valid_block_idx], axis=0)


class LagEstimator:
  def __init__(self, CP, dt, block_num=5, block_size=100, window_sec=300.0, okay_window_sec=30.0,  min_vego=15, min_yr=np.radians(1), min_ncc=0.95):
    self.dt = dt
    self.window_sec = window_sec
    self.okay_window_sec = okay_window_sec
    self.initial_lag = CP.steerActuatorDelay
    self.min_vego = min_vego
    self.min_yr = min_yr
    self.min_ncc = min_ncc

    self.t = 0
    self.lat_active = False
    self.steering_pressed = False
    self.desired_curvature = 0
    self.v_ego = 0
    self.yaw_rate = 0

    window_len = int(window_sec / self.dt)
    self.points = Points(window_len)
    self.block_avg = BlockAverage(block_num, block_size, self.initial_lag)

    self.calibrator = PoseCalibrator()

    self.lag = self.initial_lag + 0.2

  def get_msg(self, valid):
    msg = messaging.new_message('liveActuatorDelay')

    msg.valid = valid

    liveDelay = msg.liveActuatorDelay
    liveDelay.steerActuatorDelay = self.lag
    liveDelay.isEstimated = self.block_avg.valid_blocks > 0

    return msg

  def handle_log(self, t, which, msg):
    if which == "carControl":
      self.lat_active = msg.latActive
    elif which == "carState":
      self.steering_pressed = msg.steeringPressed
      self.v_ego = msg.vEgo
    elif which == "controlsState":
      self.desired_curvature = msg.desiredCurvature
    elif which == "liveCalibration":
      self.calibrator.feed_live_calib(msg)
    elif which == "livePose":
      device_pose = Pose.from_live_pose(msg)
      calibrated_pose = self.calibrator.build_calibrated_pose(device_pose)
      self.yaw_rate = calibrated_pose.angular_velocity.z
    self.t = t

  def points_valid(self):
    return self.points.num_okay >= int(self.okay_window_sec / self.dt)

  def update_points(self):
    okay = self.lat_active and not self.steering_pressed and self.v_ego > self.min_vego and np.abs(self.yaw_rate) >= self.min_yr
    la_desired = self.desired_curvature * self.v_ego * self.v_ego
    la_actual_pose = self.yaw_rate * self.v_ego

    self.points.update(self.t, la_desired, la_actual_pose, okay)
    if not okay or not self.points_valid():
      return

    times, desired, actual, okay = self.points.get()
    times_interp = np.arange(times[-1] - self.window_sec, times[-1], DT_CTRL)
    desired_interp = np.interp(times_interp, times, desired)
    actual_interp = np.interp(times_interp, times, actual)
    okay_interp = np.interp(times_interp, times, okay).astype(bool)

    delay, corr = self.actuator_delay(desired_interp, actual_interp, okay_interp, DT_CTRL)
    if corr < self.min_ncc:
      return

    self.block_avg.update(delay)
    if (new_lag := self.block_avg.get()) is not None:
      self.lag = new_lag

  def correlation_lags(self, sig_len, dt):
    return np.arange(0, sig_len) * dt

  def actuator_delay(self, expected_sig, actual_sig, is_okay, dt, max_lag=1.):
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
  sm = messaging.SubMaster(['livePose', 'liveCalibration', 'carControl', 'carState', 'controlsState'], poll='livePose')

  params = Params()
  CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)
  estimator = LagEstimator(CP, 1. / SERVICE_LIST['livePose'].frequency)

  while True:
    sm.update()
    if sm.all_checks():
      for which in sorted(sm.updated.keys(), key=lambda x: sm.logMonoTime[x]):
        if sm.updated[which]:
          t = sm.logMonoTime[which] * 1e-9
          estimator.handle_log(t, which, sm[which])
      estimator.update_points()

    if sm.frame % 25 == 0:
      msg = estimator.get_msg(sm.all_checks())
      alert_msg = messaging.new_message('alertDebug')
      alert_msg.alertDebug.alertText1 = f"Lag estimate (fixed: {CP.steerActuatorDelay:.2f} s)"
      alert_msg.alertDebug.alertText2 = f"{msg.liveActuatorDelay.steerActuatorDelay:.2f} s ({msg.liveActuatorDelay.isEstimated})"

      pm.send('liveActuatorDelay', msg)
      pm.send('alertDebug', alert_msg)


if __name__ == "__main__":
  main()
