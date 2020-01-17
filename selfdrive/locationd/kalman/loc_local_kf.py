#!/usr/bin/env python3
import numpy as np
from selfdrive.locationd.kalman import loc_local_model

from selfdrive.locationd.kalman.kalman_helpers import ObservationKind
from selfdrive.locationd.kalman.ekf_sym import EKF_sym



class States():
  VELOCITY = slice(0,3) # device frame velocity in m/s
  ANGULAR_VELOCITY = slice(3, 6) # roll, pitch and yaw rates in device frame in radians/s
  GYRO_BIAS = slice(6, 9) # roll, pitch and yaw biases
  ODO_SCALE = slice(9, 10) # odometer scale
  ACCELERATION = slice(10, 13) # Acceleration in device frame in m/s**2


class LocLocalKalman():
  def __init__(self):
    x_initial = np.array([0, 0, 0,
                          0, 0, 0,
                          0, 0, 0,
                          1,
                          0, 0, 0])

    # state covariance
    P_initial = np.diag([10**2, 10**2, 10**2,
                         1**2, 1**2, 1**2,
                         0.05**2, 0.05**2, 0.05**2,
                         0.02**2,
                         1**2, 1**2, 1**2])

    # process noise
    Q = np.diag([0.0**2, 0.0**2, 0.0**2,
                 .01**2, .01**2, .01**2,
                 (0.005/100)**2, (0.005/100)**2, (0.005/100)**2,
                 (0.02/100)**2,
                 3**2, 3**2, 3**2])

    self.obs_noise = {ObservationKind.ODOMETRIC_SPEED: np.atleast_2d(0.2**2),
                      ObservationKind.PHONE_GYRO: np.diag([0.025**2, 0.025**2, 0.025**2])}

    # MSCKF stuff
    self.dim_state = len(x_initial)
    self.dim_main = self.dim_state

    name = 'loc_local'
    loc_local_model.gen_model(name, self.dim_state)

    # init filter
    self.filter = EKF_sym(name, Q, x_initial, P_initial, self.dim_main, self.dim_main)

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
    if self.t:
      # Does NOT modify filter state
      return self.filter._predict(self.x, self.P, t - self.t)[0]
    else:
      raise RuntimeError("Request predict on filter with uninitialized time")

  def rts_smooth(self, estimates):
    return self.filter.rts_smooth(estimates, norm_quats=True)


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
    if kind == ObservationKind.CAMERA_ODO_TRANSLATION:
      r = self.predict_and_update_odo_trans(data, t, kind)
    elif kind == ObservationKind.CAMERA_ODO_ROTATION:
      r = self.predict_and_update_odo_rot(data, t, kind)
    elif kind == ObservationKind.ODOMETRIC_SPEED:
      r = self.predict_and_update_odo_speed(data, t, kind)
    else:
      r = self.filter.predict_and_update_batch(t, kind, data, self.get_R(kind, len(data)))
    return r

  def get_R(self, kind, n):
    obs_noise = self.obs_noise[kind]
    dim = obs_noise.shape[0]
    R = np.zeros((n, dim, dim))
    for i in range(n):
      R[i,:,:] = obs_noise
    return R

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

if __name__ == "__main__":
  LocLocalKalman()
