#!/usr/bin/env python3
import sys
import numpy as np
import sympy as sp

from selfdrive.locationd.models.constants import ObservationKind
from rednose.helpers.ekf_sym import gen_code, EKF_sym


class LaneKalman():
  name = 'lane'

  @staticmethod
  def generate_code(generated_dir):
    # make functions and jacobians with sympy
    #  state variables
    dim = 6
    state = sp.MatrixSymbol('state', dim, 1)

    dd = sp.Symbol('dd')  # WARNING: NOT TIME

    # Time derivative of the state as a function of state
    state_dot = sp.Matrix(np.zeros((dim, 1)))
    state_dot[:3,0] = sp.Matrix(state[3:6,0])

    # Basic descretization, 1st order intergrator
    # Can be pretty bad if dt is big
    f_sym = sp.Matrix(state) + dd*state_dot

    #
    # Observation functions
    #
    h_lane_sym = sp.Matrix(state[:3,0])
    obs_eqs = [[h_lane_sym, ObservationKind.LANE_PT, None]]
    gen_code(generated_dir, LaneKalman.name, f_sym, dd, state, obs_eqs, dim, dim)

  def __init__(self, generated_dir, pt_std=5):
    # state
    # left and right lane centers in ecef
    # WARNING: this is not a temporal model
    # the 'time' in this kalman filter is
    # the distance traveled by the vehicle,
    # which should approximately be the
    # distance along the lane path
    # a more logical parametrization
    # states 0-2 are ecef coordinates distance d
    # states 3-5  is the 3d "velocity" of the
    # lane in ecef (m/m).
    x_initial = np.array([0,0,0,
                          0,0,0])

    # state covariance
    P_initial = np.diag([1e16, 1e16, 1e16,
                         1**2, 1**2, 1**2])

    # process noise
    Q = np.diag([0.1**2, 0.1**2, 0.1**2,
                 0.1**2, 0.1**2, 0.1*2])

    self.dim_state = len(x_initial)

    # init filter
    self.filter = EKF_sym(generated_dir, self.name, Q, x_initial, P_initial, x_initial.shape[0], P_initial.shape[0])
    self.obs_noise = {ObservationKind.LANE_PT: np.diag([pt_std**2]*3)}

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
    data = np.atleast_2d(data)
    return self.filter.predict_and_update_batch(t, kind, data, self.get_R(kind, len(data)))

  def get_R(self, kind, n):
    obs_noise = self.obs_noise[kind]
    dim = obs_noise.shape[0]
    R = np.zeros((n, dim, dim))
    for i in range(n):
      R[i,:,:] = obs_noise
    return R


if __name__ == "__main__":
  generated_dir = sys.argv[2]
  LaneKalman.generate_code(generated_dir)
