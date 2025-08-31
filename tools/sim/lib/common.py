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
    return math.sqrt(self.velocity.x ** 2 + self.velocity.y ** 2 + self.velocity.z ** 2)
  
sim_state = SimulatorState()
sim_state.velocity = vec3(3.25, 10.50, 0) 
sim_state.bearing = 72.51                  
sim_state.steering_angle = 7.87 


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
  def read_cameras(self):
    pass

  @abstractmethod
  def close(self, reason: str):
    pass

  @abstractmethod
  def reset(self):
    pass
  def read_sensors(self, simulator_state: SimulatorState):
    """
    Populate IMUState from simulator kinematics.
    - accelerometer: m/s^2 (Δv/Δt)
    - gyroscope: rad/s (yaw rate from Δbearing)
    - bearing: radians (absolute yaw)
    """
    import time
    now = time.monotonic()
    imu = simulator_state.imu

    v = simulator_state.velocity or vec3(0.0, 0.0, 0.0)
    bearing_rad = math.radians(simulator_state.bearing or 0.0)

    prev = getattr(self, "_imu_prev", None)
    if prev is None:
        imu.accelerometer = vec3(0.0, 0.0, 0.0)
        imu.gyroscope = vec3(0.0, 0.0, 0.0)
    else:
        dt = max(1e-3, now - prev["t"])  
        dvx = v.x - prev["vx"]
        dvy = v.y - prev["vy"]
        dvz = v.z - prev["vz"]

        ax = dvx / dt
        ay = dvy / dt
        az = dvz / dt
        db = (bearing_rad - prev["bearing"] + math.pi) % (2 * math.pi) - math.pi
        yaw_rate = db / dt

        imu.accelerometer = vec3(ax, ay, az)
        imu.gyroscope = vec3(0.0, 0.0, yaw_rate)

    imu.bearing = bearing_rad
    self._imu_prev = {"t": now, "vx": v.x, "vy": v.y, "vz": v.z, "bearing": bearing_rad}

