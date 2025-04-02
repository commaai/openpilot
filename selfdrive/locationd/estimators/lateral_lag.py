import numpy as np
from collections import deque
from functools import partial, cache

import cereal.messaging as messaging
from cereal import log
from openpilot.selfdrive.locationd.helpers import PoseCalibrator, Pose

BLOCK_SIZE = 100
BLOCK_NUM = 50
BLOCK_NUM_NEEDED = 5
MOVING_WINDOW_SEC = 300.0
MIN_OKAY_WINDOW_SEC = 30.0
MIN_RECOVERY_BUFFER_SEC = 2.0
MIN_VEGO = 15.0
MIN_ABS_YAW_RATE = np.radians(1.0)
MIN_NCC = 0.95


@cache
def fft_next_good_size(n):
    """
    smallest composite of 2, 3, 5, 7, 11 that is >= n
    inspired by pocketfft
    """
    if n <= 6: return n
    best = 2 * n; f2 = 1
    while f2 < best:
        f23 = f2
        while f23 < best:
            f235 = f23
            while f235 < best:
                f2357 = f235
                while f2357 < best:
                    f235711 = f2357
                    while f235711 < best:
                        best = f235711 if f235711 >= n else best; f235711 *= 11
                    f2357 *= 7
                f235 *= 5
            f23 *= 3
        f2 *= 2
    return best


def parabolic_peak_interp(R, max_index):
  if max_index == 0 or max_index == len(R) - 1:
    return max_index

  y_m1, y_0, y_p1 = R[max_index - 1], R[max_index], R[max_index + 1]
  offset = 0.5 * (y_p1 - y_m1) / (2 * y_0 - y_p1 - y_m1)

  return max_index + offset


def masked_normalized_cross_correlation(expected_sig, actual_sig, mask, n):
  """
  References:
    D. Padfield. "Masked FFT registration". In Proc. Computer Vision and
    Pattern Recognition, pp. 2918-2925 (2010).
    :DOI:`10.1109/CVPR.2010.5540032`
  """

  eps = np.finfo(np.float64).eps
  expected_sig = np.asarray(expected_sig, dtype=np.float64)
  actual_sig = np.asarray(actual_sig, dtype=np.float64)

  expected_sig[~mask] = 0.0
  actual_sig[~mask] = 0.0

  rotated_expected_sig = expected_sig[::-1]
  rotated_mask = mask[::-1]

  fft = partial(np.fft.fft, n=n)

  actual_sig_fft = fft(actual_sig)
  rotated_expected_sig_fft = fft(rotated_expected_sig)
  actual_mask_fft = fft(mask.astype(np.float64))
  rotated_mask_fft = fft(rotated_mask.astype(np.float64))

  number_overlap_masked_samples = np.fft.ifft(rotated_mask_fft * actual_mask_fft).real
  number_overlap_masked_samples[:] = np.round(number_overlap_masked_samples)
  number_overlap_masked_samples[:] = np.fmax(number_overlap_masked_samples, eps)
  masked_correlated_actual_fft = np.fft.ifft(rotated_mask_fft * actual_sig_fft).real
  masked_correlated_expected_fft = np.fft.ifft(actual_mask_fft * rotated_expected_sig_fft).real

  numerator = np.fft.ifft(rotated_expected_sig_fft * actual_sig_fft).real
  numerator -= masked_correlated_actual_fft * masked_correlated_expected_fft / number_overlap_masked_samples

  actual_squared_fft = fft(actual_sig ** 2)
  actual_sig_denom = np.fft.ifft(rotated_mask_fft * actual_squared_fft).real
  actual_sig_denom -= masked_correlated_actual_fft ** 2 / number_overlap_masked_samples
  actual_sig_denom[:] = np.fmax(actual_sig_denom, 0.0)

  rotated_expected_squared_fft = fft(rotated_expected_sig ** 2)
  expected_sig_denom = np.fft.ifft(actual_mask_fft * rotated_expected_squared_fft).real
  expected_sig_denom -= masked_correlated_expected_fft ** 2 / number_overlap_masked_samples
  expected_sig_denom[:] = np.fmax(expected_sig_denom, 0.0)

  denom = np.sqrt(actual_sig_denom * expected_sig_denom)

  # zero-out samples with very small denominators
  tol = 1e3 * eps * np.max(np.abs(denom), keepdims=True)
  nonzero_indices = denom > tol

  ncc = np.zeros_like(denom, dtype=np.float64)
  ncc[nonzero_indices] = numerator[nonzero_indices] / denom[nonzero_indices]
  np.clip(ncc, -1, 1, out=ncc)

  return ncc


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
  def __init__(self, num_blocks, block_size, valid_blocks, initial_value):
    self.num_blocks = num_blocks
    self.block_size = block_size
    self.block_idx = valid_blocks % block_size
    self.idx = 0

    self.values = np.tile(initial_value, (num_blocks, 1))
    self.valid_blocks = valid_blocks

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
    return float(np.mean(self.values[valid_block_idx], axis=0).item())


class LateralLagEstimator:
  inputs = {"carControl", "carState", "controlsState", "liveCalibration", "livePose"}

  def __init__(self, CP, dt,
               block_count=BLOCK_NUM, min_valid_block_count=BLOCK_NUM_NEEDED, block_size=BLOCK_SIZE,
               window_sec=MOVING_WINDOW_SEC, okay_window_sec=MIN_OKAY_WINDOW_SEC, min_recovery_buffer_sec=MIN_RECOVERY_BUFFER_SEC,
               min_vego=MIN_VEGO, min_yr=MIN_ABS_YAW_RATE, min_ncc=MIN_NCC):
    self.dt = dt
    self.window_sec = window_sec
    self.okay_window_sec = okay_window_sec
    self.min_recovery_buffer_sec = min_recovery_buffer_sec
    self.initial_lag = CP.steerActuatorDelay + 0.2
    self.block_size = block_size
    self.block_count = block_count
    self.min_valid_block_count = min_valid_block_count
    self.min_vego = min_vego
    self.min_yr = min_yr
    self.min_ncc = min_ncc

    self.t = 0
    self.lat_active = False
    self.steering_pressed = False
    self.steering_saturated = False
    self.desired_curvature = 0
    self.v_ego = 0
    self.yaw_rate = 0

    self.last_lat_inactive_t = 0
    self.last_steering_pressed_t = 0
    self.last_steering_saturated_t = 0
    self.last_estimate_t = 0

    self.calibrator = PoseCalibrator()

    self.reset(self.initial_lag, 0)

  def reset(self, initial_lag, valid_blocks):
    window_len = int(self.window_sec / self.dt)
    self.points = Points(window_len)
    self.block_avg = BlockAverage(self.block_count, self.block_size, valid_blocks, initial_lag)

  def get_msg(self, valid, debug=False):
    msg = messaging.new_message('liveDelay')

    msg.valid = valid

    liveDelay = msg.liveDelay

    estimated_lag = self.block_avg.get()
    liveDelay.lateralDelayEstimate = estimated_lag or self.initial_lag
    if self.block_avg.valid_blocks >= self.min_valid_block_count and estimated_lag is not None:
      liveDelay.status = log.LiveDelayData.Status.estimated
      liveDelay.lateralDelay = estimated_lag
    else:
      liveDelay.status = log.LiveDelayData.Status.unestimated
      liveDelay.lateralDelay = self.initial_lag
    liveDelay.validBlocks = self.block_avg.valid_blocks
    if debug:
      liveDelay.points = self.block_avg.values.flatten().tolist()

    return msg

  def handle_log(self, t, which, msg):
    if which == "carControl":
      self.lat_active = msg.latActive
    elif which == "carState":
      self.steering_pressed = msg.steeringPressed
      self.v_ego = msg.vEgo
    elif which == "controlsState":
      self.steering_saturated = getattr(msg.lateralControlState, msg.lateralControlState.which()).saturated
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
    if not self.lat_active:
      self.last_lat_inactive_t = self.t
    if self.steering_pressed:
      self.last_steering_pressed_t = self.t
    if self.steering_saturated:
      self.last_steering_saturated_t = self.t

    la_desired = self.desired_curvature * self.v_ego * self.v_ego
    la_actual_pose = self.yaw_rate * self.v_ego

    fast = self.v_ego > self.min_vego
    turning = np.abs(self.yaw_rate) >= self.min_yr
    has_recovered = all( # wait for recovery after !lat_active, steering_pressed, steering_saturated
      self.t - last_t >= self.min_recovery_buffer_sec
      for last_t in [self.last_lat_inactive_t, self.last_steering_pressed_t, self.last_steering_saturated_t]
    )
    okay = self.lat_active and not self.steering_pressed and not self.steering_saturated and fast and turning and has_recovered

    self.points.update(self.t, la_desired, la_actual_pose, okay)

  def update_estimate(self):
    # check if the points are valid overall
    if not self.points_valid():
      return

    times, desired, actual, okay = self.points.get()
    # check if there are any new valid data points since the last update
    if self.last_estimate_t != 0 and times[0] <= self.last_estimate_t:
      new_values_start_idx = next(-i for i, t in enumerate(reversed(times)) if t <= self.last_estimate_t)
      if (new_values_start_idx == 0 or not np.any(okay[new_values_start_idx:])):
        return

    delay, corr = self.actuator_delay(desired, actual, okay, self.dt)
    if corr < self.min_ncc:
      return

    self.block_avg.update(delay)
    self.last_estimate_t = self.t

  def correlation_lags(self, sig_len, dt):
    return np.arange(0, sig_len) * dt

  def actuator_delay(self, expected_sig, actual_sig, mask, dt, max_lag=1.):
    assert len(expected_sig) == len(actual_sig)
    max_lag_samples = int(max_lag / dt)
    padded_size = fft_next_good_size(len(expected_sig) + max_lag_samples)

    ncc = masked_normalized_cross_correlation(expected_sig, actual_sig, mask, padded_size)

    # only consider lags from 0 to max_lag
    roi_ncc = ncc[len(expected_sig) - 1: len(expected_sig) - 1 + max_lag_samples]

    max_corr_index = np.argmax(roi_ncc)
    corr = roi_ncc[max_corr_index]
    lag = parabolic_peak_interp(roi_ncc, max_corr_index) * dt

    return lag, corr
