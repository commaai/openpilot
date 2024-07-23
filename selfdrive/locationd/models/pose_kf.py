#!/usr/bin/env python

import sys
import numpy as np

from openpilot.selfdrive.locationd.models.constants import ObservationKind

if __name__=="__main__":
  import sympy as sp
  from rednose.helpers.ekf_sym import gen_code
else:
  from rednose.helpers.ekf_sym_pyx import EKF_sym_pyx

EARTH_G = 9.81


class States:
  NED_ORIENTATION = slice(0, 3)  # roll, pitch, yaw in rad
  NED_VELOCITY = slice(3, 6)  # ned velocity in m/s
  ANGULAR_VELOCITY = slice(6, 9)  # roll, pitch and yaw rates in rad/s
  GYRO_BIAS = slice(9, 12)  # roll, pitch and yaw gyroscope biases in rad/s
  ACCELERATION = slice(12, 15)  # acceleration in device frame in m/s**2
  ACCEL_BIAS = slice(15, 18)  # Acceletometer bias in m/s**2


class PoseKalman:
  name = "pose"

  # state
  initial_x = np.array([0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0])
  # state covariance
  initial_P = np.diag([0.01**2, 0.01**2, 0.01**2,
                       10**2, 10**2, 10**2,
                       1**2, 1**2, 1**2,
                       1**2, 1**2, 1**2,
                       100**2, 100**2, 100**2,
                       0.01**2, 0.01**2, 0.01**2])

  # process noise
  Q = np.diag([0.001**2, 0.001**2, 0.001**2,
               0.01**2, 0.01**2, 0.01**2,
               0.1**2, 0.1**2, 0.1**2,
               (0.005 / 100)**2, (0.005 / 100)**2, (0.005 / 100)**2,
               3**2, 3**2, 3**2,
               0.005**2, 0.005**2, 0.005**2])

  obs_noise = {ObservationKind.PHONE_GYRO: np.array([0.025**2, 0.025**2, 0.025**2]),
               ObservationKind.PHONE_ACCEL: np.array([.5**2, .5**2, .5**2]),
               ObservationKind.CAMERA_ODO_TRANSLATION: np.array([0.5**2, 0.5**2, 0.5**2]),
               ObservationKind.CAMERA_ODO_ROTATION: np.array([0.05**2, 0.05**2, 0.05**2])}

  @staticmethod
  def generate_code(generated_dir):
    name = PoseKalman.name
    dim_state = PoseKalman.initial_x.shape[0]
    dim_state_err = PoseKalman.initial_P.shape[0]

    state_sym = sp.MatrixSymbol('state', dim_state, 1)
    state = sp.Matrix(state_sym)
    velocity = state[States.NED_VELOCITY, :]
    angular_velocity = state[States.ANGULAR_VELOCITY, :]
    gyro_bias = state[States.GYRO_BIAS, :]
    acceleration = state[States.ACCELERATION, :]
    acc_bias = state[States.ACCEL_BIAS, :]

    dt = sp.Symbol('dt')

    state_dot = sp.Matrix(np.zeros((dim_state, 1)))
    state_dot[States.NED_ORIENTATION, :] = state[States.ANGULAR_VELOCITY]
    state_dot[States.NED_VELOCITY, :] = state[States.ACCELERATION]

    f_sym = state + dt * state_dot

    gravity = sp.Matrix([0, 0, -EARTH_G])
    h_gyro_sym = angular_velocity + gyro_bias
    h_acc_sym = gravity + acceleration + acc_bias
    h_phone_rot_sym = angular_velocity
    h_relative_motion = velocity
    obs_eqs = [
      [h_gyro_sym, ObservationKind.PHONE_GYRO, None],
      [h_acc_sym, ObservationKind.PHONE_ACCEL, None],
      [h_relative_motion, ObservationKind.CAMERA_ODO_TRANSLATION, None],
      [h_phone_rot_sym, ObservationKind.CAMERA_ODO_ROTATION, None],
    ]
    gen_code(generated_dir, name, f_sym, dt, state_sym, obs_eqs, dim_state, dim_state_err)

  def __init__(self, generated_dir):
    dim_state, dim_state_err = PoseKalman.initial_x.shape[0], PoseKalman.initial_P.shape[0]
    self.filter = EKF_sym_pyx(generated_dir, self.name, PoseKalman.Q, PoseKalman.initial_x, PoseKalman.initial_P, dim_state, dim_state_err)

  @property
  def x(self):
    return self.filter.state()

  @property
  def P(self):
    return self.filter.covs()

  def predict_and_observe(self, t, kind, data, obs_noise=None):
    if obs_noise is None:
      obs_noise = self.obs_noise[kind]
    R = self._get_R(len(data), obs_noise)
    data = np.atleast_2d(data)
    return self.filter.predict_and_update_batch(t, kind, data, R)

  def reset(self, t, x_init=initial_x, P_init=initial_P):
    self.filter.init_state(x_init, P_init, t)

  def _get_R(self, n, obs_noise):
    dim = obs_noise.shape[0]
    R = np.zeros((n, dim, dim))
    for i in range(n):
      R[i, :, :] = obs_noise
    return R


if __name__ == "__main__":
  generated_dir = sys.argv[2]
  PoseKalman.generate_code(generated_dir)
