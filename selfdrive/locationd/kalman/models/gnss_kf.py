#!/usr/bin/env python3
import numpy as np
import selfdrive.locationd.kalman.models.gnss_model as gnss_model

from selfdrive.locationd.kalman.helpers import ObservationKind
from selfdrive.locationd.kalman.helpers.ekf_sym import EKF_sym
from selfdrive.locationd.kalman.models.loc_kf import parse_pr, parse_prr

class States():
  ECEF_POS = slice(0,3) # x, y and z in ECEF in meters
  ECEF_VELOCITY = slice(3,6)
  CLOCK_BIAS = slice(6, 7) # clock bias in light-meters,
  CLOCK_DRIFT = slice(7, 8) # clock drift in light-meters/s,
  CLOCK_ACCELERATION = slice(8, 9) # clock acceleration in light-meters/s**2
  GLONASS_BIAS = slice(9, 10) # clock drift in light-meters/s,
  GLONASS_FREQ_SLOPE = slice(10, 11) # GLONASS bias in m expressed as bias + freq_num*freq_slope


class GNSSKalman():
  def __init__(self, N=0, max_tracks=3000):
    x_initial = np.array([-2712700.6008, -4281600.6679, 3859300.1830,
                          0, 0, 0,
                          0, 0, 0,
                          0, 0])

    # state covariance
    P_initial = np.diag([10000**2, 10000**2, 10000**2,
                         10**2, 10**2, 10**2,
                         (2000000)**2, (100)**2, (0.5)**2,
                         (10)**2, (1)**2])

    # process noise
    Q = np.diag([0.3**2, 0.3**2, 0.3**2,
                 3**2, 3**2, 3**2,
                 (.1)**2, (0)**2, (0.01)**2,
                 .1**2, (.01)**2])

    self.dim_state = x_initial.shape[0]

    # mahalanobis outlier rejection
    maha_test_kinds = []#ObservationKind.PSEUDORANGE_RATE, ObservationKind.PSEUDORANGE, ObservationKind.PSEUDORANGE_GLONASS]

    name = 'gnss'
    gnss_model.gen_model(name, self.dim_state, maha_test_kinds)

    # init filter
    self.filter = EKF_sym(name, Q, x_initial, P_initial, self.dim_state, self.dim_state, maha_test_kinds=maha_test_kinds)

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
      sat_pos_freq[i,:] = sat_pos_freq_i
      z[i,:] = z_i
      R[i,:,:] = R_i
    return self.filter.predict_and_update_batch(t, kind, z, R, sat_pos_freq)

  def predict_and_update_pseudorange_rate(self, meas, t, kind):
    R = np.zeros((len(meas), 1, 1))
    z = np.zeros((len(meas), 1))
    sat_pos_vel = np.zeros((len(meas), 6))
    for i, m in enumerate(meas):
      z_i, R_i, sat_pos_vel_i = parse_prr(m)
      sat_pos_vel[i] = sat_pos_vel_i
      R[i,:,:] = R_i
      z[i, :] = z_i
    return self.filter.predict_and_update_batch(t, kind, z, R, sat_pos_vel)


if __name__ == "__main__":
  GNSSKalman()
