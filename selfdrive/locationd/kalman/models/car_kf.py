#!/usr/bin/env python3

import math
import numpy as np
import sympy as sp

from selfdrive.locationd.kalman.helpers import ObservationKind
from selfdrive.locationd.kalman.helpers.ekf_sym import EKF_sym, gen_code
from selfdrive.locationd.kalman.models.loc_kf import parse_pr, parse_prr

from selfdrive.car import CivicParams

i = 0

def _slice(n):
  global i
  s = slice(i, i + n)
  i += n

  return s


class States():
  # Vehicle model params
  STIFFNESS = _slice(1)  # [-]
  STEER_RATIO = _slice(1)  # [-]
  ANGLE_OFFSET = _slice(1)  # [rad]
  ANGLE_OFFSET_FAST = _slice(1)  # [rad]

  VELOCITY = _slice(2)  # (x, y) [m/s]
  YAW_RATE = _slice(1)  # [rad/s]
  STEER_ANGLE = _slice(1)  # [rad]


class CarKalman():
  name = 'car'

  x_initial = np.array([
    0.5,
    12.0,
    0.0,
    0.0,

    10.0, 0.0,
    0.0,
    0.0,
  ])

  # state covariance
  P_initial = np.diag([
    5**2,
    10**2,
    math.radians(5.0)**2,
    math.radians(5.0)**2,

    10**2, 10**2,
    1**2,
    1**2,
  ])

  # process noise
  Q = np.diag([
    (.05/100)**2,
    .01**2,
    math.radians(0.01)**2,
    math.radians(0.2)**2,

    .1**2, .1**2,
    math.radians(0.1)**2,
    math.radians(0.1)**2,
  ])

  obs_noise = {
    ObservationKind.CAL_DEVICE_FRAME_XY_SPEED: np.diag([0.1**2, 0.1**2]),
    ObservationKind.CAL_DEVICE_FRAME_YAW_RATE: np.atleast_2d(math.radians(0.1)**2),
    ObservationKind.STEER_ANGLE: np.atleast_2d(math.radians(0.1)**2),
    ObservationKind.ANGLE_OFFSET_FAST: np.atleast_2d(math.radians(5.0)**2),
  }

  maha_test_kinds = []

  @staticmethod
  def generate_code():
    dim_state = CarKalman.x_initial.shape[0]
    name = CarKalman.name
    maha_test_kinds = CarKalman.maha_test_kinds

    # make functions and jacobians with sympy
    # state variables
    state_sym = sp.MatrixSymbol('state', dim_state, 1)
    state = sp.Matrix(state_sym)

    # Vehicle model
    m = CivicParams.MASS
    j = CivicParams.ROTATIONAL_INERTIA
    aF = CivicParams.CENTER_TO_FRONT
    aR = CivicParams.CENTER_TO_REAR

    x = state[States.STIFFNESS, :][0, 0]

    cF, cR = x * CivicParams.TIRE_STIFFNESS_FRONT, x * CivicParams.TIRE_STIFFNESS_REAR
    angle_offset = state[States.ANGLE_OFFSET, :][0, 0]
    angle_offset_fast = state[States.ANGLE_OFFSET_FAST, :][0, 0]
    sa = state[States.STEER_ANGLE, :][0, 0]

    sR = state[States.STEER_RATIO, :][0, 0]
    u, v = state[States.VELOCITY, :]
    r = state[States.YAW_RATE, :][0, 0]

    A = sp.Matrix(np.zeros((2, 2)))
    A[0, 0] = -(cF + cR) / (m * u)
    A[0, 1] = -(cF * aF - cR * aR) / (m * u) - u
    A[1, 0] = -(cF * aF - cR * aR) / (j * u)
    A[1, 1] = -(cF * aF**2 + cR * aR**2) / (j * u)

    B = sp.Matrix(np.zeros((2, 1)))
    B[0, 0] = cF / m / sR
    B[1, 0] = (cF * aF) / j / sR

    x = sp.Matrix([v, r])  # lateral velocity, yaw rate
    x_dot = A * x + B * (sa + angle_offset + angle_offset_fast)

    dt = sp.Symbol('dt')
    state_dot = sp.Matrix(np.zeros((dim_state, 1)))
    state_dot[States.VELOCITY.start + 1, 0] = x_dot[0]
    state_dot[States.YAW_RATE.start, 0] = x_dot[1]

    # Basic descretization, 1st order integrator
    # Can be pretty bad if dt is big
    f_sym = state + dt * state_dot

    #
    # Observation functions
    #
    h_yaw_rate = sp.Matrix([r])
    # h_velocity = sp.Matrix([u, v])
    h_velocity = sp.Matrix([u, v])
    h_steer_angle = sp.Matrix([sa])
    h_fast_angle_offset = sp.Matrix([angle_offset_fast])

    obs_eqs = [
      [h_yaw_rate, ObservationKind.CAL_DEVICE_FRAME_YAW_RATE, None],
      [h_velocity, ObservationKind.CAL_DEVICE_FRAME_XY_SPEED, None],
      [h_steer_angle, ObservationKind.STEER_ANGLE, None],
      [h_fast_angle_offset, ObservationKind.ANGLE_OFFSET_FAST, None],
    ]

    gen_code(name, f_sym, dt, state_sym, obs_eqs, dim_state, dim_state, maha_test_kinds=maha_test_kinds)

  def __init__(self):
    self.dim_state = self.x_initial.shape[0]


    # init filter
    self.filter = EKF_sym(self.name, self.Q, self.x_initial, self.P_initial, self.dim_state, self.dim_state, maha_test_kinds=self.maha_test_kinds)

  @property
  def x(self):
    return self.filter.state()

  @property
  def P(self):
    return self.filter.covs()

  def predict(self, t):
    return self.filter.predict(t)

  def rts_smooth(self, estimates):
    return self.filter.rts_smooth(estimates, norm_quats=False)

  def get_R(self, kind, n):
    obs_noise = self.obs_noise[kind]
    dim = obs_noise.shape[0]
    R = np.zeros((n, dim, dim))
    for i in range(n):
      R[i, :, :] = obs_noise
    return R

  def init_state(self, state, covs_diag=None, covs=None, filter_time=None):
    if covs_diag is not None:
      P = np.diag(covs_diag)
    elif covs is not None:
      P = covs
    else:
      P = self.filter.covs()
    self.filter.init_state(state, P, filter_time)

  def predict_and_observe(self, t, kind, data):
    if len(data) > 0:
      data = np.atleast_2d(data)
    return self.filter.predict_and_update_batch(t, kind, data, self.get_R(kind, len(data)))


if __name__ == "__main__":
  CarKalman.generate_code()
