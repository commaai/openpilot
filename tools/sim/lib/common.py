import math
import multiprocessing
import numpy as np

from abc import ABC, abstractmethod
from collections import namedtuple

W, H = 1928, 1208


vec3 = namedtuple("vec3", ["x", "y", "z"])

class GPSState:
  def __init__(self):
    self.latitude = 0
    self.longitude = 0
    self.altitude = 0

  def from_xy(self, xy):
    """Simulates a lat/lon from an xy coordinate on a plane, for simple simulation. TODO: proper global projection?"""
    BASE_LAT = 32.75308505188913
    BASE_LON = -117.2095393365393
    DEG_TO_METERS = 100000

    self.latitude = float(BASE_LAT + xy[0] / DEG_TO_METERS)
    self.longitude = float(BASE_LON + xy[1] / DEG_TO_METERS)
    self.altitude = 0


class IMUState:
  def __init__(self):
    self.accelerometer: vec3 = vec3(0,0,0)
    self.gyroscope: vec3 = vec3(0,0,0)
    self.bearing: float = 0


class SimulatorState:
  def __init__(self):
    self.valid = False
    self.is_engaged = False
    self.ignition = True

    self.velocity: vec3 = None
    self.bearing: float = 0
    self.gps = GPSState()
    self.imu = IMUState()

    self.steering_angle: float = 0

    self.user_gas: float = 0
    self.user_brake: float = 0
    self.user_torque: float = 0

    self.cruise_button = 0

    self.left_blinker = False
    self.right_blinker = False

  @property
  def speed(self):
    return math.sqrt(self.velocity.x**2 + self.velocity.y**2 + self.velocity.z**2)

  def set_accelerometer(self, prev_velocity: vec3, dt: float) -> vec3:
    if prev_velocity is None:  # First time
      prev_velocity = vec3(0, 0, 0)

    if dt <= 0:
      print("Warning: Time difference (dt) must be greater than zero. Skipping IMU update.")
      return vec3(0, 0, 0)  # Return a default acceleration value

    # Compute acceleration for each component
    acc_x = (self.velocity.x - prev_velocity.x) / dt
    acc_y = (self.velocity.y - prev_velocity.y) / dt
    acc_z = (self.velocity.z - prev_velocity.z) / dt

    self.imu.accelerometer = vec3(acc_x, acc_y, acc_z)
    # Return acceleration as a vec3
    return vec3(acc_x, acc_y, acc_z)

  def set_gyroscope(self, prev_bearing: float, dt: float) -> vec3:
    """
    Calculate a simple gyroscope reading based on the rate of change of bearing (yaw).
    For more advanced scenarios, you could also estimate pitch and roll rates.
    """
    if prev_bearing is None:  # First time
      prev_bearing = 0

    if dt <= 0:
      print("Warning: Time difference (dt) must be greater than zero. Skipping IMU update.")
      return vec3(0, 0, 0)

    # Calculate change in bearing (yaw)
    delta_bearing = self.bearing - prev_bearing
    yaw_rate = delta_bearing / dt

    self.imu.gyroscope = vec3(0, 0, yaw_rate)

    # For now, assume no pitch or roll rate 0 car probably not rotating along its axis and not tilting its nose most of the time
    return vec3(0, 0, yaw_rate)


class World(ABC):
  def __init__(self, dual_camera):
    self.dual_camera = dual_camera

    self.image_lock = multiprocessing.Semaphore(value=0)
    self.road_image = np.zeros((H, W, 3), dtype=np.uint8)
    self.wide_road_image = np.zeros((H, W, 3), dtype=np.uint8)

    self.exit_event = multiprocessing.Event()

  @abstractmethod
  def apply_controls(self, steer_sim, throttle_out, brake_out):
    pass

  @abstractmethod
  def tick(self):
    pass

  @abstractmethod
  def read_state(self):
    pass

  @abstractmethod
  def read_sensors(self, simulator_state: SimulatorState):
    pass

  @abstractmethod
  def read_cameras(self):
    pass

  @abstractmethod
  def close(self, reason: str):
    pass

  @abstractmethod
  def reset(self):
    pass
