import math

import pytest

from openpilot.tools.sim.lib.common import compute_imu_from_vehicle_state, vec3


def test_compute_imu_first_sample_is_zero():
  accel, gyro = compute_imu_from_vehicle_state(
    current_velocity=vec3(1.0, 2.0, 0.0),
    current_bearing=10.0,
    previous_velocity=None,
    previous_bearing=None,
    dt=None,
  )

  assert accel == vec3(0.0, 0.0, 0.0)
  assert gyro == vec3(0.0, 0.0, 0.0)


def test_compute_imu_accel_and_yaw_rate():
  accel, gyro = compute_imu_from_vehicle_state(
    current_velocity=vec3(10.0, 4.0, 0.0),
    current_bearing=40.0,
    previous_velocity=vec3(0.0, 0.0, 0.0),
    previous_bearing=10.0,
    dt=2.0,
  )

  assert accel == vec3(5.0, 2.0, 0.0)
  assert gyro.x == 0.0
  assert gyro.y == 0.0
  assert gyro.z == pytest.approx(math.radians(15.0))


def test_compute_imu_wraps_bearing_delta():
  _, gyro = compute_imu_from_vehicle_state(
    current_velocity=vec3(0.0, 0.0, 0.0),
    current_bearing=10.0,
    previous_velocity=vec3(0.0, 0.0, 0.0),
    previous_bearing=350.0,
    dt=1.0,
  )

  assert gyro.x == 0.0
  assert gyro.y == 0.0
  assert gyro.z == pytest.approx(math.radians(20.0))
