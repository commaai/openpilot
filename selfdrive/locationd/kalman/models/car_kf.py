#!/usr/bin/env python3

import math
import numpy as np
import sympy as sp

from selfdrive.locationd.kalman.helpers import ObservationKind
from selfdrive.locationd.kalman.helpers.ekf_sym import EKF_sym, gen_code

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
    1.0,
    15.0,
    0.0,
    0.0,

    10.0, 0.0,
    0.0,
    0.0,
  ])

  # state covariance
  P_initial = np.diag([
    .1**2,
    .1**2,
    math.radians(0.1)**2,
    math.radians(0.1)**2,

    10**2, 10**2,
    1**2,
    1**2,
  ])

  # process noise
  Q = np.diag([
    (.05/10)**2,
    .0001**2,
    math.radians(0.01)**2,
    math.radians(0.2)**2,

    .1**2, .1**2,
    math.radians(0.1)**2,
    math.radians(0.1)**2,
  ])

  obs_noise = {
    ObservationKind.STEER_ANGLE: np.atleast_2d(math.radians(0.1)**2),
    ObservationKind.ANGLE_OFFSET_FAST: np.atleast_2d(math.radians(5.0)**2),
    ObservationKind.STEER_RATIO: np.atleast_2d(50.0**2),
    ObservationKind.STIFFNESS: np.atleast_2d(50.0**2),
  }

  maha_test_kinds = []  # [ObservationKind.ROAD_FRAME_YAW_RATE, ObservationKind.ROAD_FRAME_XY_SPEED]
  global_vars = [
    sp.Symbol('mass'),
    sp.Symbol('rotational_inertia'),
    sp.Symbol('center_to_front'),
    sp.Symbol('center_to_rear'),
    sp.Symbol('stiffness_front'),
    sp.Symbol('stiffness_rear'),
  ]

  @staticmethod
  def generate_code():
    dim_state = CarKalman.x_initial.shape[0]
    name = CarKalman.name
    maha_test_kinds = CarKalman.maha_test_kinds

    # globals
    m, j, aF, aR, cF_orig, cR_orig = CarKalman.global_vars

    # make functions and jacobians with sympy
    # state variables
    state_sym = sp.MatrixSymbol('state', dim_state, 1)
    state = sp.Matrix(state_sym)

    # Vehicle model constants
    x = state[States.STIFFNESS, :][0, 0]

    cF, cR = x * cF_orig, x * cR_orig
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
    x_dot = A * x + B * (sa - angle_offset - angle_offset_fast)

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
    obs_eqs = [
      [sp.Matrix([r]), ObservationKind.ROAD_FRAME_YAW_RATE, None],
      [sp.Matrix([u, v]), ObservationKind.ROAD_FRAME_XY_SPEED, None],
      [sp.Matrix([sa]), ObservationKind.STEER_ANGLE, None],
      [sp.Matrix([angle_offset_fast]), ObservationKind.ANGLE_OFFSET_FAST, None],
      [sp.Matrix([sR]), ObservationKind.STEER_RATIO, None],
      [sp.Matrix([x]), ObservationKind.STIFFNESS, None],
    ]

    gen_code(name, f_sym, dt, state_sym, obs_eqs, dim_state, dim_state, maha_test_kinds=maha_test_kinds, global_vars=CarKalman.global_vars)

  def __init__(self, steer_ratio=15, stiffness_factor=1, angle_offset=0):
    self.dim_state = self.x_initial.shape[0]
    x_init = self.x_initial
    x_init[States.STEER_RATIO] = steer_ratio
    x_init[States.STIFFNESS] = stiffness_factor
    x_init[States.ANGLE_OFFSET] = angle_offset

    # init filter
    self.filter = EKF_sym(self.name, self.Q, self.x_initial, self.P_initial, self.dim_state, self.dim_state, maha_test_kinds=self.maha_test_kinds, global_vars=self.global_vars)

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

  def predict_and_observe(self, t, kind, data, R=None):
    if len(data) > 0:
      data = np.atleast_2d(data)

    if R is None:
      R = self.get_R(kind, len(data))

    self.filter.predict_and_update_batch(t, kind, data, R)


if __name__ == "__main__":
  CarKalman.generate_code()
