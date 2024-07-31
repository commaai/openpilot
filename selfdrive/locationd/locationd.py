#!/usr/bin/env python3
import os
import capnp
import numpy as np

from cereal import log, messaging
from openpilot.common.transformations.orientation import rot_from_euler
from openpilot.common.realtime import config_realtime_process
from openpilot.selfdrive.locationd.helpers import rotate_std
from openpilot.selfdrive.locationd.models.pose_kf import PoseKalman, States
from openpilot.selfdrive.locationd.models.constants import ObservationKind, GENERATED_DIR

ACCEL_SANITY_CHECK = 100.0  # m/s^2
ROTATION_SANITY_CHECK = 10.0  # rad/s
TRANS_SANITY_CHECK = 200.0  # m/s
CALIB_RPY_SANITY_CHECK = 0.5  # rad (+- 30 deg)
MIN_STD_SANITY_CHECK = 1e-5  # m or rad
MAX_FILTER_REWIND_TIME = 0.8  # s
MAX_SENSOR_TIME_DIFF = 0.1  # s
YAWRATE_CROSS_ERR_CHECK_FACTOR = 30


def init_xyz_measurement(measurement: capnp._DynamicStructBuilder, values: np.ndarray, stds: np.ndarray, valid: bool):
  assert len(values) == len(stds) == 3
  for i, d in enumerate(("x", "y", "z")):
    setattr(measurement, d, values[i])
    setattr(measurement, f"{d}Std", stds[i])
  measurement.valid = valid


class LocationEstimator:
  def __init__(self, debug: bool):
    self.kf = PoseKalman(GENERATED_DIR, MAX_FILTER_REWIND_TIME)

    self.debug = debug

    self.posenet_stds = []
    self.camodo_yawrate_distribution = np.array([0.0, 10.0])  # mean, std
    self.device_from_calib = np.eye(3)

  def reset(self, t: float):
    self.kf.reset(t)

  def _validate_sensor_time(self, sensor_time: float, t: float):
    # ingore empty readings
    if sensor_time == 0:
      return False

    # sensor time and log time should be close
    sensor_time_invalid = abs(sensor_time - t) > MAX_SENSOR_TIME_DIFF
    if sensor_time_invalid:
      print("Sensor reading ignored, sensor timestamp more than 100ms off from log time")
    return not sensor_time_invalid

  def _validate_timestamp(self, t: float):
    invalid = not np.isnan(self.kf.t) and (self.kf.t - t) > MAX_FILTER_REWIND_TIME
    if invalid:
      print("Observation timestamp is older than the max rewind threshold of the filter")
    return not invalid

  def _finite_check(self, t: float):
    all_finite = np.all(np.isfinite(self.kf.x)) and np.all(np.isfinite(self.kf.P))
    if not all_finite:
      print("Non-finite values detected, kalman reset")
      self.reset(t)

  def handle_log(self, t: float, which: str, msg: capnp._DynamicStructReader):
    if which == "accelerometer" and msg.which() == "accelerometer":
      sensor_time = msg.timestamp * 1e-9

      if not self._validate_sensor_time(sensor_time, t) or not self._validate_timestamp(sensor_time):
        return

      v = msg.accelerometer.v
      meas = np.array([-v[2], -v[1], -v[0]])
      if np.linalg.norm(meas) < ACCEL_SANITY_CHECK:
        self.kf.predict_and_observe(sensor_time, ObservationKind.PHONE_ACCEL, meas)
      else:
        pass
    elif which == "gyroscope" and msg.which() == "gyroUncalibrated":
      sensor_time = msg.timestamp * 1e-9

      if not self._validate_sensor_time(sensor_time, t) or not self._validate_timestamp(sensor_time):
        return

      v = msg.gyroUncalibrated.v
      meas = np.array([-v[2], -v[1], -v[0]])

      gyro_bias = self.kf.x[States.GYRO_BIAS]
      gyro_camodo_yawrate_err = np.abs((meas[2] - gyro_bias[2]) - self.camodo_yawrate_distribution[0])
      gyro_camodo_yawrate_err_threshold = YAWRATE_CROSS_ERR_CHECK_FACTOR * self.camodo_yawrate_distribution[1]
      gyro_valid = gyro_camodo_yawrate_err < gyro_camodo_yawrate_err_threshold

      if np.linalg.norm(meas) < ROTATION_SANITY_CHECK and gyro_valid:
        self.kf.predict_and_observe(sensor_time, ObservationKind.PHONE_GYRO, meas)
      else:
        pass
    elif which == "liveCalibration":
      if not len(msg.rpyCalib):
        return
      if not self._validate_timestamp(t):
        return

      calib = np.array(msg.rpyCalib)
      if calib.min() < -CALIB_RPY_SANITY_CHECK or calib.max() > CALIB_RPY_SANITY_CHECK:
        return

      self.device_from_calib = rot_from_euler(calib)
      self.calibrated = msg.calStatus == log.LiveCalibrationData.Status.calibrated
    elif which == "cameraOdometry":
      rot_device = np.matmul(self.device_from_calib, np.array(msg.rot))
      trans_device = np.matmul(self.device_from_calib, np.array(msg.trans))

      if not self._validate_timestamp(t):
        return

      if np.linalg.norm(rot_device) > ROTATION_SANITY_CHECK or np.linalg.norm(trans_device) > TRANS_SANITY_CHECK:
        return

      rot_calib_std = np.array(msg.rotStd)
      trans_calib_std = np.array(msg.transStd)

      if rot_calib_std.min() <= MIN_STD_SANITY_CHECK or trans_calib_std.min() <= MIN_STD_SANITY_CHECK:
        return

      if np.linalg.norm(rot_calib_std) > 10 * ROTATION_SANITY_CHECK or np.linalg.norm(trans_calib_std) > 10 * TRANS_SANITY_CHECK:
        return

      self.posenet_stds.pop(0)
      self.posenet_stds.append(trans_calib_std[0])

      # Multiply by 10 to avoid to high certainty in kalman filter because of temporally correlated noise
      rot_calib_std *= 10
      trans_calib_std *= 10

      rot_device_std = rotate_std(self.device_from_calib, rot_calib_std)
      trans_device_std = rotate_std(self.device_from_calib, trans_calib_std)
      rot_device_noise = rot_device_std ** 2
      trans_device_noise = trans_device_std ** 2

      self.kf.predict_and_observe(t, ObservationKind.CAMERA_ODO_ROTATION, rot_device, rot_device_noise)
      self.kf.predict_and_observe(t, ObservationKind.CAMERA_ODO_TRANSLATION, trans_device, trans_device_noise)
      self.camodo_yawrate_distribution =  np.array([rot_device[2], rot_device_std[2]])
    self._finite_check(t)

  def get_msg(self, sensors_valid: bool, inputs_valid: bool, msg_valid: bool):
    state, cov = self.kf.x, self.kf.P
    std = np.sqrt(np.diag(cov))

    orientation_ned, orientation_ned_std = state[States.NED_ORIENTATION], std[States.NED_ORIENTATION]
    velocity_device, velocity_device_std = state[States.DEVICE_VELOCITY], std[States.DEVICE_VELOCITY]
    angular_velocity_device, angular_velocity_device_std = state[States.ANGULAR_VELOCITY], std[States.ANGULAR_VELOCITY]
    acceleration_device, acceleration_device_std = state[States.ACCELERATION], std[States.ACCELERATION]

    msg = messaging.new_message("livePose")
    msg.valid = msg_valid

    livePose = msg.livePose
    init_xyz_measurement(livePose.orientationNED, orientation_ned, orientation_ned_std, True)
    init_xyz_measurement(livePose.velocityDevice, velocity_device, velocity_device_std, True)
    init_xyz_measurement(livePose.angularVelocityDevice, angular_velocity_device, angular_velocity_device_std, True)
    init_xyz_measurement(livePose.accelerationDevice, acceleration_device, acceleration_device_std, True)
    if self.debug:
      livePose.filterState.value = state.tolist()
      livePose.filterState.std = std.tolist()
      livePose.filterState.valid = True
    livePose.inputsOK = inputs_valid
    livePose.posenetOK = True
    livePose.sensorsOK = sensors_valid

    return msg


def main():
  config_realtime_process([0, 1, 2, 3], 5)

  DEBUG = bool(int(os.getenv("DEBUG", "0")))

  pm = messaging.PubMaster(['livePose'])
  sm = messaging.SubMaster(['accelerometer', 'gyroscope', 'liveCalibration', 'cameraOdometry'])

  estimator = LocationEstimator(DEBUG)

  filter_initialized = False

  while True:
    sm.update()

    filter_initialized = sm.all_checks()
    if filter_initialized:
      for which in sorted(sm.updated.keys(), key=lambda x: sm.logMonoTime[x]):
        if sm.valid[which]:
          t = sm.logMonoTime[which] * 1e-9
          estimator.handle_log(t, which, sm[which])

    if sm.updated["cameraOdometry"]:
      inputsOK = sm.all_valid()
      sensorsOK = sm.all_checks(["accelerometer", "gyroscope"])

      msg = estimator.get_msg(sensorsOK, inputsOK, filter_initialized)
      pm.send("livePose", msg)


if __name__ == "__main__":
  main()
