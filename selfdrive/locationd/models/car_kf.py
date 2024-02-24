#!/usr/bin/env python3
import math
import sys
from typing import Any

import numpy as np

from openpilot.selfdrive.controls.lib.vehicle_model import ACCELERATION_DUE_TO_GRAVITY
from openpilot.selfdrive.locationd.models.constants import ObservationKind
from openpilot.common.swaglog import cloudlog

from rednose.helpers.kalmanfilter import KalmanFilter

if __name__ == '__main__':  # Generating sympy
  import sympy as sp
  from rednose.helpers.ekf_sym import gen_code
else:
  from rednose.helpers.ekf_sym_pyx import EKF_sym_pyx


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
  ROAD_ROLL = _slice(1)  # [rad]


class CarKalman(KalmanFilter):
  name = 'car'

  initial_x = np.array([
    1.0,
    15.0,
    0.0,
    0.0,

    10.0, 0.0,
    0.0,
    0.0,
    0.0
  ])

  # process noise
  Q = np.diag([
    (.05 / 100)**2,
    .01**2,
    math.radians(0.02)**2,
    math.radians(0.25)**2,

    .1**2, .01**2,
    math.radians(0.1)**2,
    math.radians(0.1)**2,
    math.radians(1)**2,
  ])
  P_initial = Q.copy()

  obs_noise: dict[int, Any] = {
    ObservationKind.STEER_ANGLE: np.atleast_2d(math.radians(0.05)**2),
    ObservationKind.ANGLE_OFFSET_FAST: np.atleast_2d(math.radians(10.0)**2),
    ObservationKind.ROAD_ROLL: np.atleast_2d(math.radians(1.0)**2),
    ObservationKind.STEER_RATIO: np.atleast_2d(5.0**2),
    ObservationKind.STIFFNESS: np.atleast_2d(0.5**2),
    ObservationKind.ROAD_FRAME_X_SPEED: np.atleast_2d(0.1**2),
  }

  global_vars = [
    'mass',
    'rotational_inertia',
    'center_to_front',
    'center_to_rear',
    'stiffness_front',
    'stiffness_rear',
  ]

  @staticmethod
  def generate_code(generated_dir):
    dim_state = CarKalman.initial_x.shape[0]
    name = CarKalman.name

    # vehicle models comes from The Science of Vehicle Dynamics: Handling, Braking, and Ride of Road and Race Cars
    # Model used is in 6.15 with formula from 6.198

    # globals
    global_vars = [sp.Symbol(name) for name in CarKalman.global_vars]
    m, j, aF, aR, cF_orig, cR_orig = global_vars

    # make functions and jacobians with sympy
    # state variables
    state_sym = sp.MatrixSymbol('state', dim_state, 1)
    state = sp.Matrix(state_sym)

    # Vehicle model constants
    sf = state[States.STIFFNESS, :][0, 0]

    cF, cR = sf * cF_orig, sf * cR_orig
    angle_offset = state[States.ANGLE_OFFSET, :][0, 0]
    angle_offset_fast = state[States.ANGLE_OFFSET_FAST, :][0, 0]
    theta = state[States.ROAD_ROLL, :][0, 0]
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

    C = sp.Matrix(np.zeros((2, 1)))
    C[0, 0] = ACCELERATION_DUE_TO_GRAVITY
    C[1, 0] = 0

    x = sp.Matrix([v, r])  # lateral velocity, yaw rate
    x_dot = A * x + B * (sa - angle_offset - angle_offset_fast) - C * theta

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
      [sp.Matrix([u]), ObservationKind.ROAD_FRAME_X_SPEED, None],
      [sp.Matrix([sa]), ObservationKind.STEER_ANGLE, None],
      [sp.Matrix([angle_offset_fast]), ObservationKind.ANGLE_OFFSET_FAST, None],
      [sp.Matrix([sR]), ObservationKind.STEER_RATIO, None],
      [sp.Matrix([sf]), ObservationKind.STIFFNESS, None],
      [sp.Matrix([theta]), ObservationKind.ROAD_ROLL, None],
    ]

    gen_code(generated_dir, name, f_sym, dt, state_sym, obs_eqs, dim_state, dim_state, global_vars=global_vars)

  def __init__(self, generated_dir, steer_ratio=15, stiffness_factor=1, angle_offset=0, P_initial=None):
    dim_state = self.initial_x.shape[0]
    dim_state_err = self.P_initial.shape[0]
    x_init = self.initial_x
    x_init[States.STEER_RATIO] = steer_ratio
    x_init[States.STIFFNESS] = stiffness_factor
    x_init[States.ANGLE_OFFSET] = angle_offset

    if P_initial is not None:
      self.P_initial = P_initial
    # init filter
    self.filter = EKF_sym_pyx(generated_dir, self.name, self.Q, self.initial_x, self.P_initial,
                              dim_state, dim_state_err, global_vars=self.global_vars, logger=cloudlog)


if __name__ == "__main__":
  generated_dir = sys.argv[2]
  CarKalman.generate_code(generated_dir)
