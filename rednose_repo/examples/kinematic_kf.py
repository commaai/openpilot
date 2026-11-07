#!/usr/bin/env python3
import sys

import numpy as np
import sympy as sp

from rednose.helpers.kalmanfilter import KalmanFilter

if __name__ == '__main__':  # generating sympy code
  from rednose.helpers.ekf_sym import gen_code
else:
  from rednose.helpers.ekf_sym_pyx import EKF_sym_pyx # pylint: disable=no-name-in-module


class ObservationKind():
  UNKNOWN = 0
  NO_OBSERVATION = 1
  POSITION = 1

  names = [
    'Unknown',
    'No observation',
    'Position'
  ]

  @classmethod
  def to_string(cls, kind):
    return cls.names[kind]


class States():
  POSITION = slice(0, 1)
  VELOCITY = slice(1, 2)


class KinematicKalman(KalmanFilter):
  name = 'kinematic'

  initial_x = np.array([0.5, 0.0])

  # state covariance
  initial_P_diag = np.array([1.0**2, 1.0**2])

  # process noise
  Q = np.diag([0.1**2, 2.0**2])

  obs_noise = {ObservationKind.POSITION: np.atleast_2d(0.1**2)}

  @staticmethod
  def generate_code(generated_dir):
    name = KinematicKalman.name
    dim_state = KinematicKalman.initial_x.shape[0]

    state_sym = sp.MatrixSymbol('state', dim_state, 1)
    state = sp.Matrix(state_sym)

    position = state[States.POSITION, :][0,:]
    velocity = state[States.VELOCITY, :][0,:]

    dt = sp.Symbol('dt')
    state_dot = sp.Matrix(np.zeros((dim_state, 1)))
    state_dot[States.POSITION.start, 0] = velocity
    f_sym = state + dt * state_dot

    obs_eqs = [
      [sp.Matrix([position]), ObservationKind.POSITION, None],
    ]

    gen_code(generated_dir, name, f_sym, dt, state_sym, obs_eqs, dim_state, dim_state)

  def __init__(self, generated_dir):
    dim_state = self.initial_x.shape[0]
    dim_state_err = self.initial_P_diag.shape[0]

    # init filter
    self.filter = EKF_sym_pyx(generated_dir, self.name, self.Q, self.initial_x, np.diag(self.initial_P_diag), dim_state, dim_state_err)


if __name__ == "__main__":
  generated_dir = sys.argv[2]
  KinematicKalman.generate_code(generated_dir)
