import timeit

import numpy as np
import numpy.linalg
from scipy.linalg import cho_factor, cho_solve

# We are trying to solve the following system
# (A.T * A) * x = A.T * b
# Where x are the polynomial coefficients and b is are the input points

# First we build A
deg = 3
x = np.arange(50 * 1.0)
A = np.vstack(tuple(x**n for n in range(deg, -1, -1))).T

# The first way to solve this is using the pseudoinverse, which can be precomputed
# x = (A.T * A)^-1 * A^T * b = PINV b
PINV = np.linalg.pinv(A)

# Another way is using the Cholesky decomposition
# We can note that at (A.T * A) is always positive definite
# By precomputing the Cholesky decomposition we can efficiently solve
# systems of the form (A.T * A) x = c
CHO = cho_factor(np.dot(A.T, A))


def model_polyfit_old(points, deg=3):
  A = np.vstack(tuple(x**n for n in range(deg, -1, -1))).T
  pinv = np.linalg.pinv(A)
  return np.dot(pinv, map(float, points))


def model_polyfit(points, deg=3):
  A = np.vander(x, deg + 1)
  pinv = np.linalg.pinv(A)
  return np.dot(pinv, map(float, points))


def model_polyfit_cho(points, deg=3):
  A = np.vander(x, deg + 1)
  cho = cho_factor(np.dot(A.T, A))
  c = np.dot(A.T, points)
  return cho_solve(cho, c, check_finite=False)


def model_polyfit_np(points, deg=3):
  return np.polyfit(x, points, deg)


def model_polyfit_lstsq(points, deg=3):
  A = np.vander(x, deg + 1)
  return np.linalg.lstsq(A, points, rcond=None)[0]


TEST_DATA = np.linspace(0, 5, num=50) + 1.


def time_pinv_old():
  model_polyfit_old(TEST_DATA)


def time_pinv():
  model_polyfit(TEST_DATA)


def time_cho():
  model_polyfit_cho(TEST_DATA)


def time_np():
  model_polyfit_np(TEST_DATA)


def time_lstsq():
  model_polyfit_lstsq(TEST_DATA)


if __name__ == "__main__":
  # Verify correct results
  pinv_old = model_polyfit_old(TEST_DATA)
  pinv = model_polyfit(TEST_DATA)
  cho = model_polyfit_cho(TEST_DATA)
  numpy = model_polyfit_np(TEST_DATA)
  lstsq = model_polyfit_lstsq(TEST_DATA)

  assert all(np.isclose(pinv, pinv_old))
  assert all(np.isclose(pinv, cho))
  assert all(np.isclose(pinv, numpy))
  assert all(np.isclose(pinv, lstsq))

  # Run benchmark
  print("Pseudo inverse (old)", timeit.timeit("time_pinv_old()", setup="from __main__ import time_pinv_old", number=10000))
  print("Pseudo inverse", timeit.timeit("time_pinv()", setup="from __main__ import time_pinv", number=10000))
  print("Cholesky", timeit.timeit("time_cho()", setup="from __main__ import time_cho", number=10000))
  print("Numpy leastsq", timeit.timeit("time_lstsq()", setup="from __main__ import time_lstsq", number=10000))
  print("Numpy polyfit", timeit.timeit("time_np()", setup="from __main__ import time_np", number=10000))
