#!/usr/bin/env python3

import sys
import numpy as np

from openpilot.selfdrive.locationd.models.constants import ObservationKind

from rednose.helpers.kalmanfilter import KalmanFilter

if __name__=="__main__":
  import sympy as sp
  from rednose.helpers.ekf_sym import gen_code
  from rednose.helpers.sympy_helpers import euler_rotate, rot_to_euler
else:
  from rednose.helpers.ekf_sym_pyx import EKF_sym_pyx

EARTH_G = 9.81


class States:
  NED_ORIENTATION = slice(0, 3)  # roll, pitch, yaw in rad
  DEVICE_VELOCITY = slice(3, 6)  # ned velocity in m/s
  ANGULAR_VELOCITY = slice(6, 9)  # roll, pitch and yaw rates in rad/s
  GYRO_BIAS = slice(9, 12)  # roll, pitch and yaw gyroscope biases in rad/s
  ACCELERATION = slice(12, 15)  # acceleration in device frame in m/s**2
  ACCEL_BIAS = slice(15, 18)  # Accelerometer bias in m/s**2
  ACCEL_TEMP_COEFF = slice(18, 21)  # Accelerometer bias temperature coefficient in m/s**2/C


class PoseKalman(KalmanFilter):
  name = "pose"

  # state
  # Fleet stationary warm-up data gives a 0.01 m/s^2/C median bias on the pitch-sensitive axis.
  # The filter learns a per-device correction around this prior as temperature changes.
  initial_x = np.array([0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0,
                        0.01, 0.0, 0.0])
  # state covariance
  initial_P = np.diag([0.01**2, 0.01**2, 0.01**2,
                       10**2, 10**2, 10**2,
                       1**2, 1**2, 1**2,
                       1**2, 1**2, 1**2,
                       100**2, 100**2, 100**2,
                       0.01**2, 0.01**2, 0.01**2,
                       0.02**2, 0.02**2, 0.02**2])

  # process noise
  Q = np.diag([0.001**2, 0.001**2, 0.001**2,
               0.01**2, 0.01**2, 0.01**2,
               0.085**2, 0.085**2, 0.085**2,
               (0.005 / 100)**2, (0.005 / 100)**2, (0.005 / 100)**2,
               3**2, 3**2, 3**2,
               0.005**2, 0.005**2, 0.005**2,
               0.0001**2, 0.0001**2, 0.0001**2])

  obs_noise = {ObservationKind.PHONE_GYRO: np.diag([0.025**2, 0.025**2, 0.025**2]),
               ObservationKind.PHONE_ACCEL: np.diag([0.75**2, 0.75**2, 0.75**2]),
               ObservationKind.CAMERA_ODO_TRANSLATION: np.diag([0.5**2, 0.5**2, 0.5**2]),
               ObservationKind.CAMERA_ODO_ROTATION: np.diag([0.05**2, 0.05**2, 0.05**2])}

  @staticmethod
  def generate_code(generated_dir):
    name = PoseKalman.name
    dim_state = PoseKalman.initial_x.shape[0]
    dim_state_err = PoseKalman.initial_P.shape[0]

    state_sym = sp.MatrixSymbol('state', dim_state, 1)
    state = sp.Matrix(state_sym)
    roll, pitch, yaw = state[States.NED_ORIENTATION, :]
    velocity = state[States.DEVICE_VELOCITY, :]
    angular_velocity = state[States.ANGULAR_VELOCITY, :]
    vroll, vpitch, vyaw = angular_velocity
    gyro_bias = state[States.GYRO_BIAS, :]
    acceleration = state[States.ACCELERATION, :]
    acc_bias = state[States.ACCEL_BIAS, :]
    acc_temp_coeff = state[States.ACCEL_TEMP_COEFF, :]
    temperature_delta = sp.Symbol('temperature_delta')

    dt = sp.Symbol('dt')

    ned_from_device = euler_rotate(roll, pitch, yaw)
    device_from_ned = ned_from_device.T

    state_dot = sp.Matrix(np.zeros((dim_state, 1)))
    state_dot[States.DEVICE_VELOCITY, :] = acceleration

    f_sym = state + dt * state_dot
    device_from_device_t1 = euler_rotate(dt*vroll, dt*vpitch, dt*vyaw)
    ned_from_device_t1 = ned_from_device * device_from_device_t1
    f_sym[States.NED_ORIENTATION, :] = rot_to_euler(ned_from_device_t1)

    centripetal_acceleration = angular_velocity.cross(velocity)
    gravity = sp.Matrix([0, 0, -EARTH_G])
    h_gyro_sym = angular_velocity + gyro_bias
    h_acc_sym = device_from_ned * gravity + acceleration + centripetal_acceleration + acc_bias + acc_temp_coeff * temperature_delta
    h_phone_rot_sym = angular_velocity
    h_relative_motion_sym = velocity
    obs_eqs = [
      [h_gyro_sym, ObservationKind.PHONE_GYRO, None],
      [h_acc_sym, ObservationKind.PHONE_ACCEL, None],
      [h_relative_motion_sym, ObservationKind.CAMERA_ODO_TRANSLATION, None],
      [h_phone_rot_sym, ObservationKind.CAMERA_ODO_ROTATION, None],
    ]
    gen_code(generated_dir, name, f_sym, dt, state_sym, obs_eqs, dim_state, dim_state_err, global_vars=[temperature_delta])

  def __init__(self, generated_dir, max_rewind_age):
    dim_state, dim_state_err = PoseKalman.initial_x.shape[0], PoseKalman.initial_P.shape[0]
    self.filter = EKF_sym_pyx(generated_dir, self.name, PoseKalman.Q, PoseKalman.initial_x, PoseKalman.initial_P,
                              dim_state, dim_state_err, global_vars=['temperature_delta'], max_rewind_age=max_rewind_age)
    self.set_temperature_delta(0.0)

  def set_temperature_delta(self, temperature_delta: float) -> None:
    self.filter.set_global('temperature_delta', temperature_delta)


if __name__ == "__main__":
  generated_dir = sys.argv[2]
  PoseKalman.generate_code(generated_dir)
