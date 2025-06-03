#!/usr/bin/env python3
import os
import time
import statistics
import numpy as np
import capnp
from collections import deque
from functools import partial, wraps

import cereal.messaging as messaging
from cereal import car, log
from cereal.services import SERVICE_LIST
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.locationd.helpers import (
  PoseCalibrator, Pose,
  fft_next_good_size, parabolic_peak_interp,
)

# ----------------------------------------------------------------------
#                       inexpensive opt-in profiler
# ----------------------------------------------------------------------
def _profile_time(tag: str):
  """
  Decorator for super-cheap, per-call timing.  Activate by running with
  `PROFILE_FUNCS=1` in the environment.  Every 1 000 calls the average and
  max wall-time are logged via cloudlog (fallback: print).  Adds <80 ns
  overhead when disabled.
  """
  def decorator(fn):
    if os.getenv("PROFILE_FUNCS", "0") != "1":
      return fn                            # fast path – no profiling

    _hist: list[float] = []

    @wraps(fn)
    def wrapper(*args, **kwargs):
      t0 = time.perf_counter()
      out = fn(*args, **kwargs)
      _hist.append(time.perf_counter() - t0)
      if len(_hist) >= 1000:
        avg, mx = statistics.mean(_hist), max(_hist)
        try:
          cloudlog.info(f"{tag}: avg {avg*1e3:.3f} ms, max {mx*1e3:.3f} ms "
                        f"over {len(_hist)} calls")
        except Exception:
          print(f"{tag}: avg {avg*1e3:.3f} ms, max {mx*1e3:.3f} ms "
                f"over {len(_hist)} calls")
        _hist.clear()
      return out
    return wrapper
  return decorator
# ----------------------------------------------------------------------

BLOCK_SIZE = 100
BLOCK_NUM = 50
BLOCK_NUM_NEEDED = 5
MOVING_WINDOW_SEC = 60.0
MIN_OKAY_WINDOW_SEC = 25.0
MIN_RECOVERY_BUFFER_SEC = 2.0
MIN_VEGO = 15.0
MIN_ABS_YAW_RATE = 0.0
MAX_YAW_RATE_SANITY_CHECK = 1.0
MIN_NCC = 0.95
MAX_LAG = 1.0
MAX_LAG_STD = 0.1
MAX_LAT_ACCEL = 2.0
MAX_LAT_ACCEL_DIFF = 0.6
MIN_CONFIDENCE = 0.7
CORR_BORDER_OFFSET = 5
LAG_CANDIDATE_CORR_THRESHOLD = 0.9


def masked_normalized_cross_correlation(expected_sig: np.ndarray,
                                        actual_sig: np.ndarray,
                                        mask: np.ndarray,
                                        n: int):
  """
  Masked NCC implementation (Padfield 2010).
  """
  eps = np.finfo(np.float64).eps
  expected_sig = np.asarray(expected_sig, dtype=np.float64)
  actual_sig   = np.asarray(actual_sig,   dtype=np.float64)

  expected_sig[~mask] = 0.0
  actual_sig[~mask]   = 0.0

  rotated_expected_sig = expected_sig[::-1]
  rotated_mask         = mask[::-1]
  fft = partial(np.fft.fft, n=n)

  actual_sig_fft         = fft(actual_sig)
  rotated_expected_fft   = fft(rotated_expected_sig)
  actual_mask_fft        = fft(mask.astype(np.float64))
  rotated_mask_fft       = fft(rotated_mask.astype(np.float64))

  overlap_mask = np.fft.ifft(rotated_mask_fft * actual_mask_fft).real
  np.round(overlap_mask, out=overlap_mask)
  np.fmax(overlap_mask, eps, out=overlap_mask)

  corr_actual  = np.fft.ifft(rotated_mask_fft * actual_sig_fft).real
  corr_expect  = np.fft.ifft(actual_mask_fft  * rotated_expected_fft).real

  num = np.fft.ifft(rotated_expected_fft * actual_sig_fft).real
  num -= corr_actual * corr_expect / overlap_mask

  actual_sq_fft   = fft(actual_sig ** 2)
  denom_actual    = np.fft.ifft(rotated_mask_fft * actual_sq_fft).real
  denom_actual   -= corr_actual ** 2 / overlap_mask
  np.fmax(denom_actual, 0.0, out=denom_actual)

  expect_sq_fft   = fft(rotated_expected_sig ** 2)
  denom_expect    = np.fft.ifft(actual_mask_fft * expect_sq_fft).real
  denom_expect   -= corr_expect ** 2 / overlap_mask
  np.fmax(denom_expect, 0.0, out=denom_expect)

  denom = np.sqrt(denom_actual * denom_expect)

  tol = 1e3 * eps * np.max(np.abs(denom), keepdims=True)
  nz  = denom > tol
  ncc = np.zeros_like(denom)
  ncc[nz] = num[nz] / denom[nz]
  np.clip(ncc, -1, 1, out=ncc)
  return ncc


class Points:
  def __init__(self, num_points: int):
    self.times   = deque([0.0]   * num_points, maxlen=num_points)
    self.okay    = deque([False] * num_points, maxlen=num_points)
    self.desired = deque([0.0]   * num_points, maxlen=num_points)
    self.actual  = deque([0.0]   * num_points, maxlen=num_points)

  @property
  def num_points(self): return len(self.desired)
  @property
  def num_okay(self):   return np.count_nonzero(self.okay)

  def update(self, t: float, desired: float, actual: float, okay: bool):
    self.times.append(t)
    self.okay.append(okay)
    self.desired.append(desired)
    self.actual.append(actual)

  def get(self):
    return (np.array(self.times), np.array(self.desired),
            np.array(self.actual), np.array(self.okay))


class BlockAverage:
  def __init__(self, num_blocks: int, block_size: int,
               valid_blocks: int, initial_value: float):
    self.num_blocks = num_blocks
    self.block_size = block_size
    self.block_idx  = valid_blocks % num_blocks
    self.idx        = 0

    self.values       = np.tile(initial_value, (num_blocks, 1))
    self.valid_blocks = valid_blocks

  def update(self, value: float):
    self.values[self.block_idx] = (self.idx * self.values[self.block_idx] + value) / (self.idx + 1)
    self.idx = (self.idx + 1) % self.block_size
    if self.idx == 0:
      self.block_idx = (self.block_idx + 1) % self.num_blocks
      self.valid_blocks = min(self.valid_blocks + 1, self.num_blocks)

  def get(self):
    valid_idx          = [i for i in range(self.valid_blocks) if i != self.block_idx]
    valid_and_current  = valid_idx + ([self.block_idx] if self.idx > 0 else [])

    if valid_idx:
      valid_mean = float(np.mean(self.values[valid_idx]).item())
      valid_std  = float(np.std(self.values[valid_idx]).item())
    else:
      valid_mean = valid_std = float('nan')

    if valid_and_current:
      cur_mean = float(np.mean(self.values[valid_and_current]).item())
      cur_std  = float(np.std(self.values[valid_and_current]).item())
    else:
      cur_mean = cur_std = float('nan')

    return valid_mean, valid_std, cur_mean, cur_std


class LateralLagEstimator:
  inputs = {"carControl", "carState", "controlsState", "liveCalibration",
            "livePose"}

  def __init__(self, CP: car.CarParams, dt: float,
               block_count: int = BLOCK_NUM,
               min_valid_block_count: int = BLOCK_NUM_NEEDED,
               block_size: int = BLOCK_SIZE,
               window_sec: float = MOVING_WINDOW_SEC,
               okay_window_sec: float = MIN_OKAY_WINDOW_SEC,
               min_recovery_buffer_sec: float = MIN_RECOVERY_BUFFER_SEC,
               min_vego: float = MIN_VEGO,
               min_yr: float = MIN_ABS_YAW_RATE,
               min_ncc: float = MIN_NCC,
               max_lat_accel: float = MAX_LAT_ACCEL,
               max_lat_accel_diff: float = MAX_LAT_ACCEL_DIFF,
               min_confidence: float = MIN_CONFIDENCE,
               enabled: bool = True):
    self.dt                     = dt
    self.window_sec             = window_sec
    self.okay_window_sec        = okay_window_sec
    self.min_recovery_buffer_sec= min_recovery_buffer_sec
    self.initial_lag            = CP.steerActuatorDelay + 0.2
    self.block_size             = block_size
    self.block_count            = block_count
    self.min_valid_block_count  = min_valid_block_count
    self.min_vego               = min_vego
    self.min_yr                 = min_yr
    self.min_ncc                = min_ncc
    self.min_confidence         = min_confidence
    self.max_lat_accel          = max_lat_accel
    self.max_lat_accel_diff     = max_lat_accel_diff
    self.enabled                = enabled

    self.t = 0.0
    self.lat_active  = False
    self.steering_pressed   = False
    self.steering_saturated = False
    self.desired_curvature  = 0.0
    self.v_ego = 0.0
    self.yaw_rate = 0.0
    self.yaw_rate_std = 0.0
    self.pose_valid = False

    self.last_lat_inactive_t      = 0.0
    self.last_steering_pressed_t  = 0.0
    self.last_steering_saturated_t= 0.0
    self.last_pose_invalid_t      = 0.0
    self.last_estimate_t          = 0.0

    self.calibrator = PoseCalibrator()

    # constant FFT size (window + max-lag) – cached once
    win_len          = int(self.window_sec / self.dt)
    max_lag_samples  = int(MAX_LAG / self.dt)
    self._fft_size   = fft_next_good_size(win_len + max_lag_samples)

    self.reset(self.initial_lag, 0)

  # -------------------------------------------------------------

  def reset(self, initial_lag: float, valid_blocks: int):
    window_len   = int(self.window_sec / self.dt)
    self.points  = Points(window_len)
    self.block_avg = BlockAverage(self.block_count, self.block_size,
                                  valid_blocks, initial_lag)

  # -------------------------------------------------------------
  #                 message handling & point updates
  # -------------------------------------------------------------
  def get_msg(self, valid: bool, debug: bool = False):
    msg = messaging.new_message('liveDelay')
    msg.valid = valid
    ld = msg.liveDelay

    v_mean, v_std, cur_mean, cur_std = self.block_avg.get()
    if (self.enabled and
        self.block_avg.valid_blocks >= self.min_valid_block_count and
        not np.isnan(v_mean) and
        not np.isnan(v_std)):
      ld.status = (log.LiveDelayData.Status.invalid
                   if v_std > MAX_LAG_STD
                   else log.LiveDelayData.Status.estimated)
    else:
      ld.status = log.LiveDelayData.Status.unestimated

    ld.lateralDelay = v_mean if ld.status == log.LiveDelayData.Status.estimated \
                      else self.initial_lag
    ld.lateralDelayEstimate     = cur_mean if not np.isnan(cur_mean) else self.initial_lag
    ld.lateralDelayEstimateStd  = cur_std  if not np.isnan(cur_std)  else 0.0
    ld.validBlocks              = self.block_avg.valid_blocks
    if debug:
      ld.points = self.block_avg.values.flatten().tolist()
    return msg

  def handle_log(self, t: float, which: str, msg: capnp._DynamicStructReader):
    if which == "carControl":
      self.lat_active = msg.latActive
    elif which == "carState":
      self.steering_pressed = msg.steeringPressed
      self.v_ego = msg.vEgo
    elif which == "controlsState":
      self.steering_saturated = getattr(msg.lateralControlState,
                                        msg.lateralControlState.which()).saturated
      self.desired_curvature  = msg.desiredCurvature
    elif which == "liveCalibration":
      self.calibrator.feed_live_calib(msg)
    elif which == "livePose":
      pose = Pose.from_live_pose(msg)
      cpose = self.calibrator.build_calibrated_pose(pose)
      self.yaw_rate      = cpose.angular_velocity.yaw
      self.yaw_rate_std  = cpose.angular_velocity.yaw_std
      self.pose_valid    = (msg.angularVelocityDevice.valid and
                            msg.posenetOK and msg.inputsOK)
    self.t = t

  # -------------------------------------------------------------
  #                  point management helpers
  # -------------------------------------------------------------
  def points_enough(self):
    return self.points.num_points >= int(self.okay_window_sec / self.dt)

  def points_valid(self):
    return self.points.num_okay >= int(self.okay_window_sec / self.dt)

  def update_points(self):
    la_desired      = self.desired_curvature * self.v_ego ** 2
    la_actual_pose  = self.yaw_rate * self.v_ego

    fast            = self.v_ego > self.min_vego
    turning         = abs(self.yaw_rate) >= self.min_yr
    sensors_valid   = (self.pose_valid and
                       abs(self.yaw_rate) < MAX_YAW_RATE_SANITY_CHECK and
                       self.yaw_rate_std  < MAX_YAW_RATE_SANITY_CHECK)
    la_valid        = (abs(la_actual_pose) <= self.max_lat_accel and
                       abs(la_desired - la_actual_pose) <= self.max_lat_accel_diff)
    calib_ok        = self.calibrator.calib_valid

    if not self.lat_active:
      self.last_lat_inactive_t = self.t
    if self.steering_pressed:
      self.last_steering_pressed_t = self.t
    if self.steering_saturated:
      self.last_steering_saturated_t = self.t
    if not sensors_valid or not la_valid:
      self.last_pose_invalid_t = self.t

    recovered = all(self.t - last >= self.min_recovery_buffer_sec
                    for last in (self.last_lat_inactive_t,
                                 self.last_steering_pressed_t,
                                 self.last_steering_saturated_t,
                                 self.last_pose_invalid_t))

    okay = (self.lat_active and
            not self.steering_pressed and
            not self.steering_saturated and
            fast and turning and recovered and
            calib_ok and sensors_valid and la_valid)

    self.points.update(self.t, la_desired, la_actual_pose, okay)

  # -------------------------------------------------------------
  #                    main estimation routine
  # -------------------------------------------------------------
  @_profile_time("update_estimate")
  def update_estimate(self):
    if not self.points_enough() or not self.points_valid():
      return

    # --- cheap early-outs ----------------------------------------------------
    times = self.points.times
    if self.last_estimate_t and times[-1] <= self.last_estimate_t:
      return

    if self.last_estimate_t:
      # any new 'okay' points since last estimate?
      for i in range(len(times) - 1, -1, -1):
        if times[i] <= self.last_estimate_t:
          break
        if self.points.okay[i]:
          break
      else:
        return

    # --- heavy work only now -------------------------------------------------
    _t, desired, actual, okay = self.points.get()
    delay, corr, conf = self.actuator_delay(desired, actual, okay,
                                            self.dt, MAX_LAG)
    if corr < self.min_ncc or conf < self.min_confidence:
      return

    self.block_avg.update(delay)
    self.last_estimate_t = self.t

  # -------------------------------------------------------------
  @_profile_time("actuator_delay")
  def actuator_delay(self, expected_sig: np.ndarray, actual_sig: np.ndarray,
                     mask: np.ndarray, dt: float, max_lag: float):
    assert len(expected_sig) == len(actual_sig)
    max_lag_samples = int(max_lag / dt)
    # cached size computed once in __init__
    padded_size = self._fft_size

    ncc = masked_normalized_cross_correlation(expected_sig, actual_sig,
                                              mask, padded_size)

    roi         = np.s_[len(expected_sig) - 1: len(expected_sig) - 1 + max_lag_samples]
    ext_roi     = np.s_[roi.start - CORR_BORDER_OFFSET: roi.stop + CORR_BORDER_OFFSET]
    roi_ncc     = ncc[roi]
    ext_ncc     = ncc[ext_roi]

    max_idx = int(np.argmax(roi_ncc))
    corr    = float(roi_ncc[max_idx])
    lag     = parabolic_peak_interp(roi_ncc, max_idx) * dt

    ncc_thr   = ((roi_ncc.max() - roi_ncc.min()) * LAG_CANDIDATE_CORR_THRESHOLD
                 + roi_ncc.min())
    cand_mask = ext_ncc >= ncc_thr
    edges     = np.diff(cand_mask.astype(int), prepend=0, append=0)
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0] - 1
    run_idx   = np.searchsorted(starts,
                                max_idx + CORR_BORDER_OFFSET,
                                side='right') - 1
    width     = ends[run_idx] - starts[run_idx] + 1
    conf      = float(np.clip(1 - width * dt, 0, 1))

    return lag, corr, conf


# -------------------------------------------------------------
#                     housekeeping helpers
# -------------------------------------------------------------
def retrieve_initial_lag(params: Params, CP: car.CarParams):
  last_lag = params.get("LiveDelay")
  last_cp  = params.get("CarParamsPrevRoute")

  if last_lag is not None:
    try:
      with log.Event.from_bytes(last_lag) as lag_msg, \
           car.CarParams.from_bytes(last_cp) as prev_cp:
        if prev_cp.carFingerprint != CP.carFingerprint:
          raise RuntimeError("Car model mismatch")

        ld = lag_msg.liveDelay
        if ld.status == log.LiveDelayData.Status.invalid:
          raise RuntimeError("Last lag estimate marked invalid")
        if ld.validBlocks > BLOCK_NUM:
          raise RuntimeError("Invalid number of valid blocks")

        return ld.lateralDelayEstimate, ld.validBlocks
    except Exception as e:
      cloudlog.error(f"Failed to retrieve initial lag: {e}")
      params.remove("LiveDelay")
  return None


# -------------------------------------------------------------
def main():
  config_realtime_process([0, 1, 2, 3], 5)
  DEBUG = bool(int(os.getenv("DEBUG", "0")))

  pm = messaging.PubMaster(['liveDelay'])
  sm = messaging.SubMaster(['livePose', 'liveCalibration', 'carState',
                            'controlsState', 'carControl'],
                           poll='livePose')

  params = Params()
  CP = messaging.log_from_bytes(params.get("CarParams", block=True),
                                car.CarParams)

  is_release = params.get_bool("IsReleaseBranch")

  lag     = LateralLagEstimator(CP, 1.0 / SERVICE_LIST['livePose'].frequency,
                                enabled=not is_release)
  initial = retrieve_initial_lag(params, CP)
  if initial is not None:
    init_lag, valid_blocks = initial
    lag.reset(init_lag, valid_blocks)

  while True:
    sm.update()
    if sm.all_checks():
      for which in sorted(sm.updated.keys(),
                          key=lambda k: sm.logMonoTime[k]):
        if sm.updated[which]:
          t = sm.logMonoTime[which] * 1e-9
          lag.handle_log(t, which, sm[which])
      lag.update_points()

    # 4 Hz (driven by 20 Hz livePose) → send every 5 frames
    if sm.frame % 5 == 0:
      lag.update_estimate()
      msg = lag.get_msg(sm.all_checks(), DEBUG)
      pkt = msg.to_bytes()
      pm.send('liveDelay', pkt)

      # cache estimate every 60 s (20 Hz * 3000 frames)
      if sm.frame % 1200 == 0:
        params.put_nonblocking("LiveDelay", pkt)


if __name__ == "__main__":
  main()
