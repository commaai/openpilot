from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
  from rednose.helpers.ekf_sym import EKF_sym
  from rednose.helpers.ekf_sym_pyx import EKF_sym_pyx


class KalmanFilter:
  name = "<name>"
  initial_x: np.ndarray = np.zeros(0)
  initial_P_diag: np.ndarray = np.zeros(0)
  Q: np.ndarray = np.zeros((0, 0))
  obs_noise: dict[int, Any] = {}

  # Initialized by each KalmanFilter implementation.
  filter: "EKF_sym | EKF_sym_pyx"

  @property
  def x(self):
    return self.filter.state()

  @property
  def t(self):
    return self.filter.get_filter_time()

  @property
  def P(self):
    return self.filter.covs()

  def init_state(self, state, covs_diag=None, covs=None, filter_time=None):
    if covs_diag is not None:
      P = np.diag(covs_diag)
    elif covs is not None:
      P = covs
    else:
      P = self.filter.covs()
    self.filter.init_state(state, P, filter_time)

  def get_R(self, kind, n):
    obs_noise = self.obs_noise[kind]
    dim = obs_noise.shape[0]
    R = np.zeros((n, dim, dim))
    for i in range(n):
      R[i, :, :] = obs_noise
    return R

  def predict_and_observe(self, t, kind, data, R=None):
    if len(data) > 0:
      data = np.atleast_2d(data)

    if R is None:
      R = self.get_R(kind, len(data))

    return self.filter.predict_and_update_batch(t, kind, data, R)
