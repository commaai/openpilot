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
        self.accelerometer = vec3(0, 0, 0)
        self.gyroscope = vec3(0, 0, 0)
        self.bearing = 0.0  # in degrees

def update_imu_state(imu_state, current_vel, last_vel, current_bearing_deg, last_bearing_deg, dt):
    # Calculate linear acceleration vector
    accel_x = (current_vel.x - last_vel.x) / dt
    accel_y = (current_vel.y - last_vel.y) / dt
    accel_z = 0.0

    imu_state.accelerometer = vec3(accel_x, accel_y, accel_z)

    # Calculate gyro_z as rate of change of bearing
    # Make sure to wrap angle difference properly between -180 and 180 degrees to avoid jumps
    delta_bearing = current_bearing_deg - last_bearing_deg
    if delta_bearing > 180:
        delta_bearing -= 360
    elif delta_bearing < -180:
        delta_bearing += 360

    gyro_z = delta_bearing / dt  # degrees per second

    imu_state.gyroscope = vec3(0.0, 0.0, gyro_z)

    # Update bearing in IMUState
    imu_state.bearing = current_bearing_deg



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
    return math.sqrt(self.velocity.x ** 2 + self.velocity.y ** 2 + self.velocity.z ** 2)


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
