#!/usr/bin/env python3
import os
import sys
import sympy as sp
import numpy as np
import unittest

if __name__ == '__main__':  # generating sympy code
  from rednose.helpers.ekf_sym import gen_code
else:
  from rednose.helpers.ekf_sym_pyx import EKF_sym_pyx # pylint: disable=no-name-in-module
  from rednose.helpers.ekf_sym import EKF_sym as EKF_sym2


GENERATED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'generated'))


class ObservationKind:
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


class States:
  POSITION = slice(0, 1)
  VELOCITY = slice(1, 2)


class CompareFilter:
  name = "compare"

  initial_x = np.array([0.5, 0.0])
  initial_P_diag = np.array([1.0**2, 1.0**2])
  Q = np.diag([0.1**2, 2.0**2])
  obs_noise = {ObservationKind.POSITION: np.atleast_2d(0.1**2)}

  @staticmethod
  def generate_code(generated_dir):
    name = CompareFilter.name
    dim_state = CompareFilter.initial_x.shape[0]

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
    self.filter_py = EKF_sym_pyx(generated_dir, self.name, self.Q, self.initial_x, np.diag(self.initial_P_diag), dim_state, dim_state_err)
    self.filter_pyx = EKF_sym2(generated_dir, self.name, self.Q, self.initial_x, np.diag(self.initial_P_diag), dim_state, dim_state_err)

  def get_R(self, kind, n):
    obs_noise = self.obs_noise[kind]
    dim = obs_noise.shape[0]
    R = np.zeros((n, dim, dim))
    for i in range(n):
      R[i, :, :] = obs_noise
    return R


class TestCompare(unittest.TestCase):
  def test_compare(self):
    np.random.seed(0)

    kf = CompareFilter(GENERATED_DIR)

    # Simple simulation
    dt = 0.01
    ts = np.arange(0, 5, step=dt)
    xs = np.empty(ts.shape)

    # Simulate
    x = 0.0
    for i, v in enumerate(np.sin(ts * 5)):
      xs[i] = x
      x += v * dt

    # insert late observation
    switch = (20, 40)
    ts[switch[0]], ts[switch[1]] = ts[switch[1]], ts[switch[0]]
    xs[switch[0]], xs[switch[1]] = xs[switch[1]], xs[switch[0]]

    for t, x in zip(ts, xs):
      # get measurement
      meas = np.random.normal(x, 0.1)
      z = np.array([[meas]])
      R = kf.get_R(ObservationKind.POSITION, 1)

      # Update kf
      kf.filter_py.predict_and_update_batch(t, ObservationKind.POSITION, z, R)
      kf.filter_pyx.predict_and_update_batch(t, ObservationKind.POSITION, z, R)

      self.assertAlmostEqual(kf.filter_py.get_filter_time(), kf.filter_pyx.get_filter_time())
      self.assertTrue(np.allclose(kf.filter_py.state(), kf.filter_pyx.state()))
      self.assertTrue(np.allclose(kf.filter_py.covs(), kf.filter_pyx.covs()))


if __name__ == "__main__":
  generated_dir = sys.argv[2]
  CompareFilter.generate_code(generated_dir)
