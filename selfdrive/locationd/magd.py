#!/usr/bin/env python3
import os
import sys
import signal
import numpy as np
from typing import Any, Tuple, Union

import cereal.messaging as messaging
from cereal import car, log
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.system.swaglog import cloudlog
from openpilot.selfdrive.locationd.helpers import PointBuckets

POINTS_PER_BUCKET = 3000
MIN_POINTS_TOTAL = 4000
FIT_POINTS_TOTAL = 4000
MIN_VEL = 5  # m/s
FILTER_DECAY = 0.1
FILTER_DT = 0.015
BUCKET_KEYS = [(-np.pi, -np.pi / 2), (-np.pi / 2, 0), (0, np.pi / 2), (np.pi / 2, np.pi)]
MIN_BUCKET_POINTS = 1000
CALIB_DEFAULTS = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
VERSION = 1  # bump this to invalidate old parameter caches


class MagBuckets(PointBuckets):
  def add_point(self, x: float, y: float, yaw: float) -> None:
    for bound_min, bound_max in self.x_bounds:
      if (yaw >= bound_min) and (yaw < bound_max):
        self.buckets[(bound_min, bound_max)].append([x, y, yaw])
        break


class MagCalibrator:
  def __init__(self, CP: car.CarParams) -> None:
    self.reset()

    # try to restore cached params
    params = Params()
    params_cache = params.get("MagnetometerCarParams")
    magnetometer_cache = params.get("MagnetometerCalibration")
    if params_cache is not None and magnetometer_cache is not None:
      try:
        with log.Event.from_bytes(magnetometer_cache) as log_evt:
          cache_mc = log_evt.magnetometerCalibration
        with car.CarParams.from_bytes(params_cache) as msg:
          cache_CP = msg
        if self.get_restore_key(cache_CP, cache_mc.version) == self.get_restore_key(CP, VERSION):
          if cache_mc.calibrated:
            self.calibrationParams = np.array(cache_mc.calibrationParams)
          self.point_buckets.load_points(cache_mc.points)
          cloudlog.info("restored magnetometer calibration params from cache")
      except Exception:
        cloudlog.exception("failed to restore cached magnetometer calibration params")
        params.remove("MagnetometerCarParams")
        params.remove("MagnetometerCalibration")

  def get_restore_key(self, CP: car.CarParams, version: int) -> Tuple[str, int]:
    return (CP.carFingerprint, version)

  def reset(self) -> None:
    self.calibrationParams = CALIB_DEFAULTS
    self.calibrated = False
    self.bearingValid = False
    self.filtered_magnetic = [FirstOrderFilter(0, FILTER_DECAY, FILTER_DT) for _ in range(3)]
    self.filtered_vals = np.zeros(3)
    self.vego = 0
    self.yaw = 0
    self.yaw_valid = False
    self.point_buckets = MagBuckets(x_bounds=BUCKET_KEYS,
                                        min_points=[MIN_BUCKET_POINTS] * len(BUCKET_KEYS),
                                        min_points_total=MIN_POINTS_TOTAL,
                                        points_per_bucket=POINTS_PER_BUCKET,
                                        rowsize=3)
    self.past_raw_vals = np.zeros(3)

  def get_ellipsoid_rotation(self, coeffs: np.ndarray) -> Tuple[np.ndarray, float]:
    a, b, c = coeffs[:3]
    M = np.array([
        [2 * a, c],
        [c, 2 * b]
    ])
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    axes_lengths = np.sqrt(1 / eigenvalues)
    rotation_angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    return axes_lengths, rotation_angle

  def get_ellipsoid_center(self, coeffs: np.ndarray) -> np.ndarray:
    a, b, c, d, e = coeffs[:5]
    num = (c**2 / 4) - (a * b)
    x0 = ((b * d / 2) - (c / 2 * e / 2)) / num
    y0 = ((a * e / 2) - (c / 2 * d / 2)) / num
    return np.array([x0, y0])

  def estimate_calibration_params(self) -> np.ndarray:
    points = self.point_buckets.get_points(FIT_POINTS_TOTAL)
    calibration_params = np.ones(5) * np.nan
    try:
      x, y, yaw = points[:, 0], points[:, 1], points[:, 2]
      D = np.vstack((x * x, y * y, x * y, x, y, np.ones_like(x))).T
      _, _, V = np.linalg.svd(D, full_matrices=False)
      coeffs = V[-1] / V[-1, -1]
      x0, y0 = self.get_ellipsoid_center(coeffs)
      (l1, l2), theta = self.get_ellipsoid_rotation(coeffs)
      bearing = self.get_calibrated_bearing(x, y, np.array([x0, y0, l1, l2, theta, 0.0]))
      offset = np.median(self.reset_angle_range(bearing - yaw))
      calibration_params = np.array([x0, y0, l1, l2, theta, offset])
    except np.linalg.LinAlgError:
      cloudlog.exception("LinAlgError computing magnetometer calibration params")
    except Exception:
      cloudlog.exception("Error computing magnetometer calibration params")
    return calibration_params

  def reset_angle_range(self, angle: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    # reset angle range to [-pi, pi]
    return (angle + np.pi) % (2 * np.pi) - np.pi

  def get_calibrated_bearing(self, x: Union[float, np.ndarray], y: Union[float, np.ndarray], calibration_params: np.ndarray) -> Any:
    x0, y0, l1, l2, theta, offset = calibration_params
    x = x - x0
    y = y - y0
    x_new = x * np.cos(theta) + y * np.sin(theta)
    y_new = -x * np.sin(theta) + y * np.cos(theta)
    x_new, y_new = x_new / l1, y_new / l2
    bearing = np.arctan2(y_new, x_new)
    bearing = self.reset_angle_range(bearing - offset)
    return bearing

  def handle_log(self, which: str, msg: log.Event) -> None:
    if which == "carState":
      self.vego = msg.vEgo
    elif which == "liveLocationKalman":
      self.yaw = msg.orientationNED.value[2]
      self.yaw_valid = msg.orientationNED.valid
    elif which == "magnetometer":
      if msg.source == log.SensorEventData.SensorSource.mmc5603nj:
        raw_vals = np.array(msg.magneticUncalibrated.v)
        self.raw_vals = (raw_vals[: 3] - raw_vals[3:]) / 2
        if np.all(np.abs(self.raw_vals - self.past_raw_vals) < 2.0) and msg.magneticUncalibrated.status:
          self.filtered_vals = np.array([f.update(v) for f, v in zip(self.filtered_magnetic, self.raw_vals, strict=True)])
        if self.vego > MIN_VEL and self.yaw_valid:
          self.point_buckets.add_point(self.filtered_vals[0], self.filtered_vals[2], self.yaw)
        self.past_raw_vals = self.raw_vals

  def get_msg(self, valid: bool, with_points: bool = False) -> log.Event:
    msg = messaging.new_message('magnetometerCalibration')
    msg.valid = valid
    magnetometerCalibration = msg.magnetometerCalibration
    magnetometerCalibration.version = VERSION

    bearing, bearingValid = 0.0, False
    if self.calibrated:
      bearing = self.get_calibrated_bearing(self.filtered_vals[0], self.filtered_vals[2], self.calibrationParams)
      bearingValid = True

    magnetometerCalibration.calibrated = self.calibrated
    magnetometerCalibration.calibrationParams = self.calibrationParams.tolist()
    magnetometerCalibration.bearing = float(bearing)
    magnetometerCalibration.bearingValid = bearingValid
    magnetometerCalibration.totalBucketPoints = len(self.point_buckets)
    if with_points:
      magnetometerCalibration.points = self.point_buckets.get_points().tolist()
    return msg

  def compute_calibration_params(self) -> None:
    if self.point_buckets.is_valid():
      calibration_params = self.estimate_calibration_params()
      if any(np.isnan(calibration_params)):
        self.calibrationParams = CALIB_DEFAULTS
        self.calibrated = False
      else:
        self.calibrationParams = calibration_params
        self.calibrated = True


def main():
  config_realtime_process([0, 1, 2, 3], 5)

  sm = messaging.SubMaster(['magnetometer', 'carState', 'liveLocationKalman'], poll=['liveLocationKalman'])
  pm = messaging.PubMaster(['magnetometerCalibration'])

  params = Params()
  with car.CarParams.from_bytes(params.get("CarParams", block=True)) as CP:
    calibrator = MagCalibrator(CP)

  def cache_params(sig, frame):
    signal.signal(sig, signal.SIG_DFL)
    cloudlog.warning("caching mag calib params")

    params = Params()
    params.put("MagnetometerCarParams", CP.as_builder().to_bytes())

    msg = calibrator.get_msg(valid=True, with_points=True)
    params.put("MagnetometerCalibration", msg.to_bytes())

    sys.exit(0)
  if "REPLAY" not in os.environ:
    signal.signal(signal.SIGINT, cache_params)

  while True:
    sm.update()
    if sm.all_checks():
      for which in sm.updated.keys():
        if sm.updated[which]:
          calibrator.handle_log(which, sm[which])

    if sm.updated['magnetometer']:
      pm.send('magnetometerCalibration', calibrator.get_msg(valid=sm.all_checks()))

    # 1Hz
    if sm.frame % 20 == 0:
      calibrator.compute_calibration_params()


if __name__ == "__main__":
  main()
