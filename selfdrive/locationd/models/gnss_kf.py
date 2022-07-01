#!/usr/bin/env python3
import sys
from typing import List

import numpy as np

from selfdrive.locationd.models.constants import ObservationKind
from selfdrive.locationd.models.gnss_helpers import parse_pr, parse_prr

if __name__ == '__main__':  # Generating sympy
  import sympy as sp
  from rednose.helpers.ekf_sym import gen_code
else:
  from rednose.helpers.ekf_sym_pyx import EKF_sym_pyx  # pylint: disable=no-name-in-module,import-error
  from rednose.helpers.ekf_sym import EKF_sym  # pylint: disable=no-name-in-module,import-error


class States():
  ECEF_POS = slice(0, 3)  # x, y and z in ECEF in meters
  ECEF_VELOCITY = slice(3, 6)
  CLOCK_BIAS = slice(6, 7)  # clock bias in light-meters,
  CLOCK_DRIFT = slice(7, 8)  # clock drift in light-meters/s,
  CLOCK_ACCELERATION = slice(8, 9)  # clock acceleration in light-meters/s**2
  GLONASS_BIAS = slice(9, 10)  # clock drift in light-meters/s,
  GLONASS_FREQ_SLOPE = slice(10, 11)  # GLONASS bias in m expressed as bias + freq_num*freq_slope


class GNSSKalman():
  name = 'gnss'

  x_initial = np.array([-2712700.6008, -4281600.6679, 3859300.1830,
                        0, 0, 0,
                        0, 0, 0,
                        0, 0])

  # state covariance
  P_initial = np.diag([1e16, 1e16, 1e16,
                       10**2, 10**2, 10**2,
                       1e14, (100)**2, (0.2)**2,
                       (10)**2, (1)**2])

  # process noise
  Q = np.diag([0.03**2, 0.03**2, 0.03**2,
               3**2, 3**2, 3**2,
               (.1)**2, (0)**2, (0.005)**2,
               .1**2, (.01)**2])

  maha_test_kinds: List[int] = []  # ObservationKind.PSEUDORANGE_RATE, ObservationKind.PSEUDORANGE, ObservationKind.PSEUDORANGE_GLONASS]

  @staticmethod
  def generate_code(generated_dir):
    dim_state = GNSSKalman.x_initial.shape[0]
    name = GNSSKalman.name
    maha_test_kinds = GNSSKalman.maha_test_kinds

    # make functions and jacobians with sympy
    # state variables
    state_sym = sp.MatrixSymbol('state', dim_state, 1)
    state = sp.Matrix(state_sym)
    x, y, z = state[0:3, :]
    v = state[3:6, :]
    vx, vy, vz = v
    cb, cd, ca = state[6:9, :]
    glonass_bias, glonass_freq_slope = state[9:11, :]

    dt = sp.Symbol('dt')

    state_dot = sp.Matrix(np.zeros((dim_state, 1)))
    state_dot[:3, :] = v
    state_dot[6, 0] = cd
    state_dot[7, 0] = ca

    # Basic descretization, 1st order integrator
    # Can be pretty bad if dt is big
    f_sym = state + dt * state_dot

    #
    # Observation functions
    #

    # extra args
    sat_pos_freq_sym = sp.MatrixSymbol('sat_pos', 4, 1)
    sat_pos_vel_sym = sp.MatrixSymbol('sat_pos_vel', 6, 1)
    # sat_los_sym = sp.MatrixSymbol('sat_los', 3, 1)
    # orb_epos_sym = sp.MatrixSymbol('orb_epos_sym', 3, 1)

    # expand extra args
    sat_x, sat_y, sat_z, glonass_freq = sat_pos_freq_sym
    sat_vx, sat_vy, sat_vz = sat_pos_vel_sym[3:]
    # los_x, los_y, los_z = sat_los_sym
    # orb_x, orb_y, orb_z = orb_epos_sym

    h_pseudorange_sym = sp.Matrix([
      sp.sqrt(
        (x - sat_x)**2 +
        (y - sat_y)**2 +
        (z - sat_z)**2
      ) + cb
    ])

    h_pseudorange_glonass_sym = sp.Matrix([
      sp.sqrt(
        (x - sat_x)**2 +
        (y - sat_y)**2 +
        (z - sat_z)**2
      ) + cb + glonass_bias + glonass_freq_slope * glonass_freq
    ])

    los_vector = (sp.Matrix(sat_pos_vel_sym[0:3]) - sp.Matrix([x, y, z]))
    los_vector = los_vector / sp.sqrt(los_vector[0]**2 + los_vector[1]**2 + los_vector[2]**2)
    h_pseudorange_rate_sym = sp.Matrix([los_vector[0] * (sat_vx - vx) +
                                        los_vector[1] * (sat_vy - vy) +
                                        los_vector[2] * (sat_vz - vz) +
                                        cd])

    obs_eqs = [[h_pseudorange_sym, ObservationKind.PSEUDORANGE_GPS, sat_pos_freq_sym],
               [h_pseudorange_glonass_sym, ObservationKind.PSEUDORANGE_GLONASS, sat_pos_freq_sym],
               [h_pseudorange_rate_sym, ObservationKind.PSEUDORANGE_RATE_GPS, sat_pos_vel_sym],
               [h_pseudorange_rate_sym, ObservationKind.PSEUDORANGE_RATE_GLONASS, sat_pos_vel_sym]]

    gen_code(generated_dir, name, f_sym, dt, state_sym, obs_eqs, dim_state, dim_state, maha_test_kinds=maha_test_kinds)

  def __init__(self, generated_dir, cython=False):
    self.dim_state = self.x_initial.shape[0]

    # init filter
    filter_cls = EKF_sym_pyx if cython else EKF_sym
    self.filter = filter_cls(generated_dir, self.name, self.Q, self.x_initial, self.P_initial, self.dim_state,
                             self.dim_state, maha_test_kinds=self.maha_test_kinds)
    self.init_state(GNSSKalman.x_initial, covs=GNSSKalman.P_initial)

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
    if kind == ObservationKind.PSEUDORANGE_GPS or kind == ObservationKind.PSEUDORANGE_GLONASS:
      r = self.predict_and_update_pseudorange(data, t, kind)
    elif kind == ObservationKind.PSEUDORANGE_RATE_GPS or kind == ObservationKind.PSEUDORANGE_RATE_GLONASS:
      r = self.predict_and_update_pseudorange_rate(data, t, kind)
    return r

  def predict_and_update_pseudorange(self, meas, t, kind):
    R = np.zeros((len(meas), 1, 1))
    sat_pos_freq = np.zeros((len(meas), 4))
    z = np.zeros((len(meas), 1))
    for i, m in enumerate(meas):
      z_i, R_i, sat_pos_freq_i = parse_pr(m)
      sat_pos_freq[i, :] = sat_pos_freq_i
      z[i, :] = z_i
      R[i, :, :] = R_i
    return self.filter.predict_and_update_batch(t, kind, z, R, sat_pos_freq)

  def predict_and_update_pseudorange_rate(self, meas, t, kind):
    R = np.zeros((len(meas), 1, 1))
    z = np.zeros((len(meas), 1))
    sat_pos_vel = np.zeros((len(meas), 6))
    for i, m in enumerate(meas):
      z_i, R_i, sat_pos_vel_i = parse_prr(m)
      sat_pos_vel[i] = sat_pos_vel_i
      R[i, :, :] = R_i
      z[i, :] = z_i
    return self.filter.predict_and_update_batch(t, kind, z, R, sat_pos_vel)


if __name__ == "__main__":
  generated_dir = sys.argv[2]
  GNSSKalman.generate_code(generated_dir)
