#!/usr/bin/env python3
import numpy as np

from selfdrive.swaglog import cloudlog
from selfdrive.locationd.kalman.live_model import gen_model, States
from selfdrive.locationd.kalman.kalman_helpers import ObservationKind, KalmanError
from selfdrive.locationd.kalman.ekf_sym import EKF_sym


initial_x = np.array([-2.7e6, 4.2e6, 3.8e6,
                       1, 0, 0, 0,
                       0, 0, 0,
                       0, 0, 0,
                       0, 0, 0,
                       1,
                       0, 0, 0,
                       0, 0, 0])


# state covariance
initial_P_diag = np.array([10000**2, 10000**2, 10000**2,
                         10**2, 10**2, 10**2,
                         10**2, 10**2, 10**2,
                         1**2, 1**2, 1**2,
                         0.05**2, 0.05**2, 0.05**2,
                         0.02**2,
                         1**2, 1**2, 1**2,
                         (0.01)**2, (0.01)**2, (0.01)**2])


class LiveKalman():
  def __init__(self):
    # process noise
    Q = np.diag([0.03**2, 0.03**2, 0.03**2,
                 0.0**2, 0.0**2, 0.0**2,
                 0.0**2, 0.0**2, 0.0**2,
                 0.1**2, 0.1**2, 0.1**2,
                 (0.005/100)**2, (0.005/100)**2, (0.005/100)**2,
                 (0.02/100)**2,
                 3**2, 3**2, 3**2,
                 (0.05/60)**2, (0.05/60)**2, (0.05/60)**2])

    self.dim_state = initial_x.shape[0]
    self.dim_state_err = initial_P_diag.shape[0]

    self.obs_noise = {ObservationKind.ODOMETRIC_SPEED: np.atleast_2d(0.2**2),
                      ObservationKind.PHONE_GYRO: np.diag([0.025**2, 0.025**2, 0.025**2]),
                      ObservationKind.PHONE_ACCEL: np.diag([.5**2, .5**2, .5*2]),
                      ObservationKind.CAMERA_ODO_ROTATION: np.diag([0.05**2, 0.05**2, 0.05**2]),
                      ObservationKind.IMU_FRAME: np.diag([0.05**2, 0.05**2, 0.05**2]),
                      ObservationKind.NO_ROT: np.diag([0.00025**2, 0.00025**2, 0.00025**2]),
                      ObservationKind.ECEF_POS: np.diag([5**2, 5**2, 5**2])}

    name = 'live'
    gen_model(name, self.dim_state, self.dim_state_err, [])

    # init filter
    self.filter = EKF_sym(name, Q, initial_x, np.diag(initial_P_diag), self.dim_state, self.dim_state_err)

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

    # Normalize quats
    quat_norm = np.linalg.norm(self.filter.x[3:7, 0])

    # Should not continue if the quats behave this weirdly
    if not (0.1 < quat_norm < 10):
      cloudlog.error("Kalman filter quaternions unstable")
      raise KalmanError

    self.filter.x[States.ECEF_ORIENTATION, 0] = self.filter.x[States.ECEF_ORIENTATION, 0] / quat_norm

    return r

  def get_R(self, kind, n):
    obs_noise = self.obs_noise[kind]
    dim = obs_noise.shape[0]
    R = np.zeros((n, dim, dim))
    for i in range(n):
      R[i, :, :] = obs_noise
    return R

  def predict_and_update_odo_speed(self, speed, t, kind):
    z = np.array(speed)
    R = np.zeros((len(speed), 1, 1))
    for i, _ in enumerate(z):
      R[i, :, :] = np.diag([0.2**2])
    return self.filter.predict_and_update_batch(t, kind, z, R)

  def predict_and_update_odo_trans(self, trans, t, kind):
    z = trans[:, :3]
    R = np.zeros((len(trans), 3, 3))
    for i, _ in enumerate(z):
        R[i, :, :] = np.diag(trans[i, 3:]**2)
    return self.filter.predict_and_update_batch(t, kind, z, R)

  def predict_and_update_odo_rot(self, rot, t, kind):
    z = rot[:, :3]
    R = np.zeros((len(rot), 3, 3))
    for i, _ in enumerate(z):
        R[i, :, :] = np.diag(rot[i, 3:]**2)
    return self.filter.predict_and_update_batch(t, kind, z, R)


if __name__ == "__main__":
  LiveKalman()
