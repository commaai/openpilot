import numpy as np


def get_kalman_gain(dt, A, C, Q, R, iterations=100):
  P = np.zeros_like(Q)
  for _ in range(iterations):
    P = A.dot(P).dot(A.T) + dt * Q
    S = C.dot(P).dot(C.T) + R
    K = P.dot(C.T).dot(np.linalg.inv(S))
    P = (np.eye(len(P)) - K.dot(C)).dot(P)
  return K


class KF1D:
  # this EKF assumes constant covariance matrix, so calculations are much simpler
  # the Kalman gain also needs to be precomputed using the control module

  def __init__(self, x0, A, C, K):
    self.x0_0 = x0[0][0]
    self.x1_0 = x0[1][0]
    self.A0_0 = A[0][0]
    self.A0_1 = A[0][1]
    self.A1_0 = A[1][0]
    self.A1_1 = A[1][1]
    self.C0_0 = C[0]
    self.C0_1 = C[1]
    self.K0_0 = K[0][0]
    self.K1_0 = K[1][0]

    self.A_K_0 = self.A0_0 - self.K0_0 * self.C0_0
    self.A_K_1 = self.A0_1 - self.K0_0 * self.C0_1
    self.A_K_2 = self.A1_0 - self.K1_0 * self.C0_0
    self.A_K_3 = self.A1_1 - self.K1_0 * self.C0_1

    # K matrix needs to  be pre-computed as follow:
    # import control
    # (x, l, K) = control.dare(np.transpose(self.A), np.transpose(self.C), Q, R)
    # self.K = np.transpose(K)

  def update(self, meas):
    #self.x = np.dot(self.A_K, self.x) + np.dot(self.K, meas)
    x0_0 = self.A_K_0 * self.x0_0 + self.A_K_1 * self.x1_0 + self.K0_0 * meas
    x1_0 = self.A_K_2 * self.x0_0 + self.A_K_3 * self.x1_0 + self.K1_0 * meas
    self.x0_0 = x0_0
    self.x1_0 = x1_0
    return [self.x0_0, self.x1_0]

  @property
  def x(self):
    return [[self.x0_0], [self.x1_0]]

  def set_x(self, x):
    self.x0_0 = x[0][0]
    self.x1_0 = x[1][0]
