#!/usr/bin/env python3
import numpy as np
from selfdrive.locationd.kalman.models.loc_model import gen_model

from selfdrive.locationd.kalman.helpers import ObservationKind
from selfdrive.locationd.kalman.helpers.lst_sq_computer import LstSqComputer, unroll_shutter
from selfdrive.locationd.kalman.helpers.ekf_sym import EKF_sym
from laika.raw_gnss import GNSSMeasurement


def parse_prr(m):
  sat_pos_vel_i = np.concatenate((m[GNSSMeasurement.SAT_POS],
                                  m[GNSSMeasurement.SAT_VEL]))
  R_i = np.atleast_2d(m[GNSSMeasurement.PRR_STD]**2)
  z_i = m[GNSSMeasurement.PRR]
  return z_i, R_i, sat_pos_vel_i


def parse_pr(m):
  pseudorange = m[GNSSMeasurement.PR]
  pseudorange_stdev = m[GNSSMeasurement.PR_STD]
  sat_pos_freq_i = np.concatenate((m[GNSSMeasurement.SAT_POS],
                                   np.array([m[GNSSMeasurement.GLONASS_FREQ]])))
  z_i = np.atleast_1d(pseudorange)
  R_i = np.atleast_2d(pseudorange_stdev**2)
  return z_i, R_i, sat_pos_freq_i


class States():
  ECEF_POS = slice(0,3) # x, y and z in ECEF in meters
  ECEF_ORIENTATION = slice(3,7) # quat for pose of phone in ecef
  ECEF_VELOCITY = slice(7,10) # ecef velocity in m/s
  ANGULAR_VELOCITY = slice(10, 13) # roll, pitch and yaw rates in device frame in radians/s
  CLOCK_BIAS = slice(13, 14) # clock bias in light-meters,
  CLOCK_DRIFT = slice(14, 15) # clock drift in light-meters/s,
  GYRO_BIAS = slice(15, 18) # roll, pitch and yaw biases
  ODO_SCALE = slice(18, 19) # odometer scale
  ACCELERATION = slice(19, 22) # Acceleration in device frame in m/s**2
  FOCAL_SCALE = slice(22, 23) # focal length scale
  IMU_OFFSET = slice(23,26) # imu offset angles in radians
  GLONASS_BIAS = slice(26,27) # GLONASS bias in m expressed as bias + freq_num*freq_slope
  GLONASS_FREQ_SLOPE = slice(27, 28) # GLONASS bias in m expressed as bias + freq_num*freq_slope
  CLOCK_ACCELERATION = slice(28, 29) # clock acceleration in light-meters/s**2,


class LocKalman():
  def __init__(self, N=0, max_tracks=3000):
    x_initial = np.array([-2.7e6, 4.2e6, 3.8e6,
                          1, 0, 0, 0,
                          0, 0, 0,
                          0, 0, 0,
                          0, 0,
                          0, 0, 0,
                          1,
                          0, 0, 0,
                          1,
                          0, 0, 0,
                          0, 0,
                          0])

    # state covariance
    P_initial = np.diag([10000**2, 10000**2, 10000**2,
                         10**2, 10**2, 10**2,
                         10**2, 10**2, 10**2,
                         1**2, 1**2, 1**2,
                         (200000)**2, (100)**2,
                         0.05**2, 0.05**2, 0.05**2,
                         0.02**2,
                         1**2, 1**2, 1**2,
                         0.01**2,
                         (0.01)**2, (0.01)**2, (0.01)**2,
                         10**2, 1**2,
                         0.05**2])

    # process noise
    Q = np.diag([0.03**2, 0.03**2, 0.03**2,
                 0.0**2, 0.0**2, 0.0**2,
                 0.0**2, 0.0**2, 0.0**2,
                 0.1**2, 0.1**2, 0.1**2,
                 (.1)**2, (0.0)**2,
                 (0.005/100)**2, (0.005/100)**2, (0.005/100)**2,
                 (0.02/100)**2,
                 3**2, 3**2, 3**2,
                 0.001**2,
                 (0.05/60)**2, (0.05/60)**2, (0.05/60)**2,
                 (.1)**2, (.01)**2,
                 0.005**2])

    self.obs_noise = {ObservationKind.ODOMETRIC_SPEED: np.atleast_2d(0.2**2),
                      ObservationKind.PHONE_GYRO: np.diag([0.025**2, 0.025**2, 0.025**2]),
                      ObservationKind.PHONE_ACCEL:  np.diag([.5**2, .5**2, .5*2]),
                      ObservationKind.CAMERA_ODO_ROTATION:  np.diag([0.05**2, 0.05**2, 0.05**2]),
                      ObservationKind.IMU_FRAME:  np.diag([0.05**2, 0.05**2, 0.05**2]),
                      ObservationKind.NO_ROT: np.diag([0.00025**2, 0.00025**2, 0.00025**2]),
                      ObservationKind.ECEF_POS: np.diag([5**2, 5**2, 5**2])}

    # MSCKF stuff
    self.N = N
    self.dim_main = x_initial.shape[0]
    self.dim_augment = 7
    self.dim_main_err = P_initial.shape[0]
    self.dim_augment_err = 6
    self.dim_state = self.dim_main + self.dim_augment*self.N
    self.dim_state_err = self.dim_main_err + self.dim_augment_err*self.N

    # mahalanobis outlier rejection
    maha_test_kinds = [ObservationKind.ORB_FEATURES] #, ObservationKind.PSEUDORANGE, ObservationKind.PSEUDORANGE_RATE]

    name = 'loc_%d' % N
    gen_model(name, N, self.dim_main, self.dim_main_err,
                        self.dim_augment, self.dim_augment_err,
                        self.dim_state, self.dim_state_err,
                        maha_test_kinds)

    if self.N > 0:
      x_initial, P_initial, Q = self.pad_augmented(x_initial, P_initial, Q)
      self.computer = LstSqComputer(N)
      self.max_tracks = max_tracks

    # init filter
    self.filter = EKF_sym(name, Q, x_initial, P_initial, self.dim_main, self.dim_main_err,
                          N, self.dim_augment, self.dim_augment_err, maha_test_kinds)

  @property
  def x(self):
    return self.filter.state()

  @property
  def t(self):
    return self.filter.filter_time

  @property
  def P(self):
    return self.filter.covs()

  def predict(self, t):
    return self.filter.predict(t)

  def rts_smooth(self, estimates):
    return self.filter.rts_smooth(estimates, norm_quats=True)

  def pad_augmented(self, x, P, Q=None):
    if x.shape[0] == self.dim_main and self.N > 0:
      x = np.pad(x, (0, self.N*self.dim_augment), mode='constant')
      x[self.dim_main+3::7] = 1
    if P.shape[0] == self.dim_main_err and self.N > 0:
      P = np.pad(P, [(0, self.N*self.dim_augment_err), (0, self.N*self.dim_augment_err)], mode='constant')
      P[self.dim_main_err:, self.dim_main_err:] = 10e20*np.eye(self.dim_augment_err *self.N)
    if Q is None:
      return x, P
    else:
      Q = np.pad(Q, [(0, self.N*self.dim_augment_err), (0, self.N*self.dim_augment_err)], mode='constant')
      return x, P, Q

  def init_state(self, state, covs_diag=None, covs=None, filter_time=None):
    if covs_diag is not None:
      P = np.diag(covs_diag)
    elif covs is not None:
      P = covs
    else:
      P = self.filter.covs()
    state, P = self.pad_augmented(state, P)
    self.filter.init_state(state, P, filter_time)

  def predict_and_observe(self, t, kind, data):
    if len(data) > 0:
      data = np.atleast_2d(data)
    if kind == ObservationKind.CAMERA_ODO_TRANSLATION:
      r = self.predict_and_update_odo_trans(data, t, kind)
    elif kind == ObservationKind.CAMERA_ODO_ROTATION:
      r = self.predict_and_update_odo_rot(data, t, kind)
    elif kind == ObservationKind.PSEUDORANGE_GPS or kind == ObservationKind.PSEUDORANGE_GLONASS:
      r = self.predict_and_update_pseudorange(data, t, kind)
    elif kind == ObservationKind.PSEUDORANGE_RATE_GPS or kind == ObservationKind.PSEUDORANGE_RATE_GLONASS:
      r = self.predict_and_update_pseudorange_rate(data, t, kind)
    elif kind == ObservationKind.ORB_POINT:
      r = self.predict_and_update_orb(data, t, kind)
    elif kind == ObservationKind.ORB_FEATURES:
      r = self.predict_and_update_orb_features(data, t, kind)
    elif kind == ObservationKind.MSCKF_TEST:
      r = self.predict_and_update_msckf_test(data, t, kind)
    elif kind == ObservationKind.FEATURE_TRACK_TEST:
      r = self.predict_and_update_feature_track_test(data, t, kind)
    elif kind == ObservationKind.ODOMETRIC_SPEED:
      r = self.predict_and_update_odo_speed(data, t, kind)
    else:
      r = self.filter.predict_and_update_batch(t, kind, data, self.get_R(kind, len(data)))
    # Normalize quats
    quat_norm = np.linalg.norm(self.filter.x[3:7,0])
    # Should not continue if the quats behave this weirdly
    if not 0.1 < quat_norm < 10:
      raise RuntimeError("Sir! The filter's gone all wobbly!")
    self.filter.x[3:7,0] = self.filter.x[3:7,0]/quat_norm
    for i in range(self.N):
      d1 = self.dim_main
      d3 = self.dim_augment
      self.filter.x[d1+d3*i+3:d1+d3*i+7] /= np.linalg.norm(self.filter.x[d1+i*d3 + 3:d1+i*d3 + 7,0])
    return r

  def get_R(self, kind, n):
    obs_noise = self.obs_noise[kind]
    dim = obs_noise.shape[0]
    R = np.zeros((n, dim, dim))
    for i in range(n):
      R[i,:,:] = obs_noise
    return R

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

  def predict_and_update_orb(self, orb, t, kind):
    true_pos = orb[:,2:]
    z = orb[:,:2]
    R = np.zeros((len(orb), 2, 2))
    for i, _ in enumerate(z):
      R[i,:,:] = np.diag([10**2, 10**2])
    return self.filter.predict_and_update_batch(t, kind, z, R, true_pos)

  def predict_and_update_odo_speed(self, speed, t, kind):
    z = np.array(speed)
    R = np.zeros((len(speed), 1, 1))
    for i, _ in enumerate(z):
      R[i,:,:] = np.diag([0.2**2])
    return self.filter.predict_and_update_batch(t, kind, z, R)

  def predict_and_update_odo_trans(self, trans, t, kind):
    z = trans[:,:3]
    R = np.zeros((len(trans), 3, 3))
    for i, _ in enumerate(z):
        R[i,:,:] = np.diag(trans[i,3:]**2)
    return self.filter.predict_and_update_batch(t, kind, z, R)

  def predict_and_update_odo_rot(self, rot, t, kind):
    z = rot[:,:3]
    R = np.zeros((len(rot), 3, 3))
    for i, _ in enumerate(z):
        R[i,:,:] = np.diag(rot[i,3:]**2)
    return self.filter.predict_and_update_batch(t, kind, z, R)

  def predict_and_update_orb_features(self, tracks, t, kind):
    k = 2*(self.N+1)
    R = np.zeros((len(tracks), k, k))
    z = np.zeros((len(tracks), k))
    ecef_pos = np.zeros((len(tracks), 3))
    ecef_pos[:] = np.nan
    poses = self.x[self.dim_main:].reshape((-1,7))
    times = tracks.reshape((len(tracks),self.N+1, 4))[:,:,0]
    good_counter = 0
    if times.any() and np.allclose(times[0,:-1], self.filter.augment_times, rtol=1e-6):
      for i, track in enumerate(tracks):
        img_positions = track.reshape((self.N+1, 4))[:,2:]
        # TODO not perfect as last pose not used
        #img_positions = unroll_shutter(img_positions, poses, self.filter.state()[7:10], self.filter.state()[10:13], ecef_pos[i])
        ecef_pos[i] = self.computer.compute_pos(poses, img_positions[:-1])
        z[i] = img_positions.flatten()
        R[i,:,:] = np.diag([0.005**2]*(k))
        if np.isfinite(ecef_pos[i][0]):
          good_counter += 1
          if good_counter > self.max_tracks:
            break
    good_idxs = np.all(np.isfinite(ecef_pos),axis=1)
    # have to do some weird stuff here to keep
    # to have the observations input from mesh3d
    # consistent with the outputs of the filter
    # Probably should be replaced, not sure how.
    ret = self.filter.predict_and_update_batch(t, kind, z[good_idxs], R[good_idxs], ecef_pos[good_idxs], augment=True)
    if ret is None:
      return
    y_full = np.zeros((z.shape[0], z.shape[1] - 3))
    #print sum(good_idxs), len(tracks)
    if sum(good_idxs) > 0:
        y_full[good_idxs] = np.array(ret[6])
    ret = ret[:6] + (y_full, z, ecef_pos)
    return ret

  def predict_and_update_msckf_test(self, test_data, t, kind):
    assert self.N > 0
    z = test_data
    R = np.zeros((len(test_data), len(z[0]), len(z[0])))
    ecef_pos = [self.x[:3]]
    for i, _ in enumerate(z):
      R[i,:,:] = np.diag([0.1**2]*len(z[0]))
    ret = self.filter.predict_and_update_batch(t, kind, z, R, ecef_pos)
    self.filter.augment()
    return ret

  def maha_test_pseudorange(self, x, P, meas, kind, maha_thresh=.3):
    bools = []
    for i, m in enumerate(meas):
      z, R, sat_pos_freq = parse_pr(m)
      bools.append(self.filter.maha_test(x, P, kind, z, R, extra_args=sat_pos_freq, maha_thresh=maha_thresh))
    return np.array(bools)

  def maha_test_pseudorange_rate(self, x, P, meas, kind, maha_thresh=.999):
    bools = []
    for i, m in enumerate(meas):
      z, R, sat_pos_vel = parse_prr(m)
      bools.append(self.filter.maha_test(x, P, kind, z, R, extra_args=sat_pos_vel, maha_thresh=maha_thresh))
    return np.array(bools)


if __name__ == "__main__":
  LocKalman(N=4)
