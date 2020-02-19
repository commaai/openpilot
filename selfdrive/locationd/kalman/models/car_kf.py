#!/usr/bin/env python3

import numpy as np
import sympy as sp

from selfdrive.locationd.kalman.helpers import ObservationKind
from selfdrive.locationd.kalman.helpers.ekf_sym import EKF_sym, gen_code
from selfdrive.locationd.kalman.models.loc_kf import parse_pr, parse_prr

from selfdrive.car import CivicParams


class States():

  # Vehicle model params
  STIFFNES = slice(0, 2)  # N/rad
  STEER_RATIO = slice(2, 3)  # N/rad

  VELOCITY = slice(3, 5)  # x, y m/s
  YAW_RATE = slice(5, 6)  # [rad/s]
  STEER_ANGLE = slice(6, 7)  # [rad]


class CarKalman():
  name = 'car'

  x_initial = np.array([
    CivicParams.TIRE_STIFFNESS_FRONT, CivicParams.TIRE_STIFFNESS_REAR,
    15.0,
    0.0, 0.0,
    0.0,
    0.0,
  ])

  # state covariance
  P_initial = np.diag([
    10000**2, 10000**2,
    10**2,
    10**2, 10**2,
    10**2,
    10**2,
  ])

  # process noise
  Q = np.diag([
    1, 1,
    1,
    1, 1,
    1,
    1,
  ])

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

    cF, cR = state[States.STIFFNES, :]
    sR = state[States.STEER_RATIO, :][0, 0]
    sa = state[States.STEER_ANGLE, :][0, 0]
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

    x = sp.Matrix([v, r])
    x_dot = A * x + B * sa

    dt = sp.Symbol('dt')
    state_dot = sp.Matrix(np.zeros((dim_state, 1)))
    state_dot[3, 0] = x_dot[0]
    state_dot[5, 0] = x_dot[1]

    # Basic descretization, 1st order integrator
    # Can be pretty bad if dt is big
    f_sym = state + dt * state_dot

    #
    # Observation functions
    #
    h_yaw_rate = sp.Matrix([r])
    h_velocity = sp.Matrix([u, v])
    h_steer_angle = sp.Matrix([sa])

    obs_eqs = [
      [h_yaw_rate, ObservationKind.CAL_DEVICE_FRAME_YAW_RATE, None],
      [h_velocity, ObservationKind.CAL_DEVICE_FRAME_XY_SPEED, None],
      [h_steer_angle, ObservationKind.STEER_ANGLE, None],
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

    # TODO: Predict


if __name__ == "__main__":
  CarKalman.generate_code()
