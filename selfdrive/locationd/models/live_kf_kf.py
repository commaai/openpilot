#!/usr/bin/env python3

import numpy as np

from selfdrive.swaglog import cloudlog
from selfdrive.locationd.models.constants import ObservationKind
from rednose.helpers.ekf_sym_pyx import EKF_sym  # pylint: disable=no-name-in-module

EARTH_GM = 3.986005e14  # m^3/s^2 (gravitational constant * mass of earth)


class States():
  ECEF_POS = slice(0, 3)  # x, y and z in ECEF in meters
  ECEF_ORIENTATION = slice(3, 7)  # quat for pose of phone in ecef
  ECEF_VELOCITY = slice(7, 10)  # ecef velocity in m/s
  ANGULAR_VELOCITY = slice(10, 13)  # roll, pitch and yaw rates in device frame in radians/s
  GYRO_BIAS = slice(13, 16)  # roll, pitch and yaw biases
  ODO_SCALE = slice(16, 17)  # odometer scale
  ACCELERATION = slice(17, 20)  # Acceleration in device frame in m/s**2
  IMU_OFFSET = slice(20, 23)  # imu offset angles in radians

  # Error-state has different slices because it is an ESKF
  ECEF_POS_ERR = slice(0, 3)
  ECEF_ORIENTATION_ERR = slice(3, 6)  # euler angles for orientation error
  ECEF_VELOCITY_ERR = slice(6, 9)
  ANGULAR_VELOCITY_ERR = slice(9, 12)
  GYRO_BIAS_ERR = slice(12, 15)
  ODO_SCALE_ERR = slice(15, 16)
  ACCELERATION_ERR = slice(16, 19)
  IMU_OFFSET_ERR = slice(19, 22)


class LiveKalman():
  name = 'live'

  initial_x = np.array([-2.7e6, 4.2e6, 3.8e6,
                        1, 0, 0, 0,
                        0, 0, 0,
                        0, 0, 0,
                        0, 0, 0,
                        1,
                        0, 0, 0,
                        0, 0, 0])

  # state covariance
  initial_P_diag = np.array([1e16, 1e16, 1e16,
                             1e6, 1e6, 1e6,
                             1e4, 1e4, 1e4,
                             1**2, 1**2, 1**2,
                             0.05**2, 0.05**2, 0.05**2,
                             0.02**2,
                             1**2, 1**2, 1**2,
                             (0.01)**2, (0.01)**2, (0.01)**2])

  # process noise
  Q = np.diag([0.03**2, 0.03**2, 0.03**2,
               0.001**2, 0.001**2, 0.001**2,
               0.01**2, 0.01**2, 0.01**2,
               0.1**2, 0.1**2, 0.1**2,
               (0.005 / 100)**2, (0.005 / 100)**2, (0.005 / 100)**2,
               (0.02 / 100)**2,
               3**2, 3**2, 3**2,
               (0.05 / 60)**2, (0.05 / 60)**2, (0.05 / 60)**2])

  def __init__(self, generated_dir):
    self.dim_state = self.initial_x.shape[0]
    self.dim_state_err = self.initial_P_diag.shape[0]

    self.obs_noise = {ObservationKind.ODOMETRIC_SPEED: np.atleast_2d(0.2**2),
                      ObservationKind.PHONE_GYRO: np.diag([0.025**2, 0.025**2, 0.025**2]),
                      ObservationKind.PHONE_ACCEL: np.diag([.5**2, .5**2, .5**2]),
                      ObservationKind.CAMERA_ODO_ROTATION: np.diag([0.05**2, 0.05**2, 0.05**2]),
                      ObservationKind.IMU_FRAME: np.diag([0.05**2, 0.05**2, 0.05**2]),
                      ObservationKind.NO_ROT: np.diag([0.00025**2, 0.00025**2, 0.00025**2]),
                      ObservationKind.ECEF_POS: np.diag([5**2, 5**2, 5**2]),
                      ObservationKind.ECEF_VEL: np.diag([.5**2, .5**2, .5**2]),
                      ObservationKind.ECEF_ORIENTATION_FROM_GPS: np.diag([.2**2, .2**2, .2**2, .2**2])}

    # init filter
    self.filter = EKF_sym(generated_dir, self.name, self.Q, self.initial_x, np.diag(self.initial_P_diag), self.dim_state, self.dim_state_err, max_rewind_age=0.2, logger=cloudlog)

  @property
  def x(self):
    return self.filter.get_state()

  @property
  def t(self):
    return self.filter.get_filter_time

  @property
  def P(self):
    return self.filter.get_covs()

  def rts_smooth(self, estimates):
    return self.filter.rts_smooth(estimates, norm_quats=True)

  def init_state(self, state, covs_diag=None, covs=None, filter_time=None):
    if covs_diag is not None:
      P = np.diag(covs_diag)
    elif covs is not None:
      P = covs
    else:
      P = self.filter.covs()
    self.filter.init_state(state, P, filter_time)

  def predict_and_observe(self, t, kind, meas, R=None):
    if len(meas) > 0:
      meas = np.atleast_2d(meas)
    if kind == ObservationKind.CAMERA_ODO_TRANSLATION:
      r = self.predict_and_update_odo_trans(meas, t, kind)
    elif kind == ObservationKind.CAMERA_ODO_ROTATION:
      r = self.predict_and_update_odo_rot(meas, t, kind)
    elif kind == ObservationKind.ODOMETRIC_SPEED:
      r = self.predict_and_update_odo_speed(meas, t, kind)
    else:
      if R is None:
        R = self.get_R(kind, len(meas))
      elif len(R.shape) == 2:
        R = R[None]
      r = self.filter.predict_and_update_batch(t, kind, meas, R)

    self.filter.normalize_state(States.ECEF_ORIENTATION.start, States.ECEF_ORIENTATION.stop)
    return r

  def get_R(self, kind, n):
    obs_noise = self.obs_noise[kind]
    dim = obs_noise.shape[0]
    R = np.zeros((n, dim, dim))
    for i in range(n):
      R[i, :, :] = obs_noise
    return R

  def predict_and_update_odo_speed(self, speed, t, kind):
    z = np.array(speed)
    R = np.zeros((len(speed), 1, 1))
    for i, _ in enumerate(z):
      R[i, :, :] = np.diag([0.2**2])
    return self.filter.predict_and_update_batch(t, kind, z, R)

  def predict_and_update_odo_trans(self, trans, t, kind):
    z = trans[:, :3]
    R = np.zeros((len(trans), 3, 3))
    for i, _ in enumerate(z):
        R[i, :, :] = np.diag(trans[i, 3:]**2)
    return self.filter.predict_and_update_batch(t, kind, z, R)

  def predict_and_update_odo_rot(self, rot, t, kind):
    z = rot[:, :3]
    R = np.zeros((len(rot), 3, 3))
    for i, _ in enumerate(z):
        R[i, :, :] = np.diag(rot[i, 3:]**2)
    return self.filter.predict_and_update_batch(t, kind, z, R)
