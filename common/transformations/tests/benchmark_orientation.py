#!/usr/bin/env python3
# pylint: skip-file
import time

from common.transformations.tests.test_orientation_cython import ecef_positions, ned_eulers, eulers
from common.transformations.transformations import ecef_euler_from_ned_single, ned_euler_from_ecef_single
from common.transformations.orientation import ecef_euler_from_ned as ecef_euler_from_ned_py
from common.transformations.orientation import ned_euler_from_ecef as ned_euler_from_ecef_py

N = 1000

if __name__ == "__main__":
  print(f"N = {N}")
  t = time.time()
  for _ in range(N):
    ecef_euler_from_ned_py(ecef_positions[0], ned_eulers[0])
  dt = time.time() - t
  print(f"Python ecef_euler_from_ned {dt:.5f}")

  t = time.time()
  for _ in range(N):
    ned_euler_from_ecef_py(ecef_positions[0], ned_eulers[0])
  dt = time.time() - t
  print(f"Python ned_euler_from_ecef {dt:.5f}")

  t = time.time()
  for _ in range(N):
    ecef_euler_from_ned_single(ecef_positions[0], ned_eulers[0])
  dt = time.time() - t
  print(f"C++ ecef_euler_from_ned {dt:.5f}")

  t = time.time()
  for _ in range(N):
    ned_euler_from_ecef_single(ecef_positions[0], ned_eulers[0])
  dt = time.time() - t
  print(f"C++ ned_euler_from_ecef {dt:.5f}")
