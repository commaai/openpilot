# pylint: skip-file
from openpilot.common.kalman.simple_kalman_impl import KF1D as KF1D
assert KF1D
import numpy as np

def get_kalman_gain(dt, A, C, Q, R, iterations=100):
  P = np.zeros_like(Q)
  for _ in range(iterations):
    P = A.dot(P).dot(A.T) + dt * Q
    S = C.dot(P).dot(C.T) + R
    K = P.dot(C.T).dot(np.linalg.inv(S))
    P = (np.eye(len(P)) - K.dot(C)).dot(P)
  return K