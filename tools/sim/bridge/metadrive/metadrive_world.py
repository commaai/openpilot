import ctypes
import functools
import multiprocessing
import numpy as np
import time

from multiprocessing import Pipe, Array

from openpilot.tools.sim.bridge.common import QueueMessage, QueueMessageType
from openpilot.tools.sim.bridge.metadrive.metadrive_process import (metadrive_process, metadrive_simulation_state,
                                                                    metadrive_vehicle_state)
from openpilot.tools.sim.lib.common import SimulatorState, World , vec3
from openpilot.tools.sim.lib.camerad import W, H
from dataclasses import dataclass
import math


@dataclass
class PreviousState:
  timestamp: float = 0.0
  velocity: vec3 = None
  bearing: float = 0.0
  position: tuple[float, float] = (0.0, 0.0)

class MetaDriveWorld(World):
  def __init__(self, status_q, config, test_duration, test_run, dual_camera=False):
    super().__init__(dual_camera)
    self.status_q = status_q
    self.camera_array = Array(ctypes.c_uint8, W*H*3)
    self.road_image = np.frombuffer(self.camera_array.get_obj(), dtype=np.uint8).reshape((H, W, 3))
    self.wide_camera_array = None
    if dual_camera:
      self.wide_camera_array = Array(ctypes.c_uint8, W*H*3)
      self.wide_road_image = np.frombuffer(self.wide_camera_array.get_obj(), dtype=np.uint8).reshape((H, W, 3))

    self.controls_send, self.controls_recv = Pipe()
    self.simulation_state_send, self.simulation_state_recv = Pipe()
    self.vehicle_state_send, self.vehicle_state_recv = Pipe()

    self.exit_event = multiprocessing.Event()
    self.op_engaged = multiprocessing.Event()

    self.test_run = test_run

    self.first_engage = None
    self.last_check_timestamp = 0
    self.distance_moved = 0

    self.metadrive_process = multiprocessing.Process(
      name="metadrive process",
      target=functools.partial(
        metadrive_process,
        dual_camera,
        config,
        self.camera_array,
        self.wide_camera_array,
        self.image_lock,
        self.controls_recv,
        self.simulation_state_send,
        self.vehicle_state_send,
        self.exit_event,
        self.op_engaged,
        test_duration,
        self.test_run
      )
    )

    self.metadrive_process.start()
    self.status_q.put(QueueMessage(QueueMessageType.START_STATUS, "starting"))

    print("----------------------------------------------------------")
    print("---- Spawning Metadrive world, this might take awhile ----")
    print("----------------------------------------------------------")

    # Wait for initial state message to ensure metadrive is launched
    self.vehicle_last_pos = self.vehicle_state_recv.recv().position
    self.status_q.put(QueueMessage(QueueMessageType.START_STATUS, "started"))

    self.steer_ratio = 15
    self.vc = [0.0, 0.0]
    self.reset_time = 0
    self.should_reset = False

    # Initialize state tracking for IMU calculations
    self.prev_state = PreviousState()

  def calculate_imu_values(self, curr_velocity: vec3, curr_bearing: float,
                         curr_pos: tuple[float, float], curr_time: float) -> tuple[vec3, vec3]:
    """
    Calculate IMU accelerometer and gyroscope values from vehicle state.
    Returns (accelerometer_vec3, gyroscope_vec3)
    """
    dt = curr_time - self.prev_state.timestamp
    if dt == 0 or self.prev_state.velocity is None:
      return vec3(0, 0, 0), vec3(0, 0, 0)

    # Calculate acceleration in vehicle's local frame
    accel_x = (curr_velocity.x - self.prev_state.velocity.x) / dt
    accel_y = (curr_velocity.y - self.prev_state.velocity.y) / dt
    accel_z = (curr_velocity.z - self.prev_state.velocity.z) / dt

    # Add gravitational acceleration
    accel_z += 9.81

    # Calculate angular velocity (gyroscope)
    bearing_diff = (curr_bearing - self.prev_state.bearing)
    # Normalize bearing difference to [-180, 180]
    if bearing_diff > 180:
      bearing_diff -= 360
    elif bearing_diff < -180:
      bearing_diff += 360

    angular_velocity_z = math.radians(bearing_diff) / dt

    # Calculate lateral acceleration from turning
    speed = math.sqrt(curr_velocity.x**2 + curr_velocity.y**2)
    centripetal_accel = speed * angular_velocity_z  # v * ω

    # Adjust accelerations based on vehicle orientation
    bearing_rad = math.radians(curr_bearing)
    cos_bearing = math.cos(bearing_rad)
    sin_bearing = math.sin(bearing_rad)

    # Combine linear and centripetal accelerations
    adjusted_accel_x = accel_x * cos_bearing - accel_y * sin_bearing - centripetal_accel * sin_bearing
    adjusted_accel_y = accel_x * sin_bearing + accel_y * cos_bearing + centripetal_accel * cos_bearing

    return vec3(adjusted_accel_x, adjusted_accel_y, accel_z), vec3(0, 0, angular_velocity_z)

  def apply_controls(self, steer_angle, throttle_out, brake_out):
    if (time.monotonic() - self.reset_time) > 2:
      self.vc[0] = steer_angle

      if throttle_out:
        self.vc[1] = throttle_out
      else:
        self.vc[1] = -brake_out
    else:
      self.vc[0] = 0
      self.vc[1] = 0

    self.controls_send.send([*self.vc, self.should_reset])
    self.should_reset = False

  def read_state(self):
    while self.simulation_state_recv.poll(0):
      md_state: metadrive_simulation_state = self.simulation_state_recv.recv()
      if md_state.done:
        self.status_q.put(QueueMessage(QueueMessageType.TERMINATION_INFO, md_state.done_info))
        self.exit_event.set()

  def read_sensors(self, state: SimulatorState):
    while self.vehicle_state_recv.poll(0):
      md_vehicle: metadrive_vehicle_state = self.vehicle_state_recv.recv()
      curr_pos = md_vehicle.position
      curr_time = time.monotonic()

      # Calculate IMU values
      accelerometer, gyroscope = self.calculate_imu_values(
        md_vehicle.velocity,
        md_vehicle.bearing,
        curr_pos,
        curr_time
      )

      # Update simulator state
      state.velocity = md_vehicle.velocity
      state.bearing = md_vehicle.bearing
      state.steering_angle = md_vehicle.steering_angle
      state.gps.from_xy(curr_pos)

      # Update IMU state
      state.imu.accelerometer = accelerometer
      state.imu.gyroscope = gyroscope
      state.imu.bearing = md_vehicle.bearing

      state.valid = True

      # Store current state for next iteration
      self.prev_state.timestamp = curr_time
      self.prev_state.velocity = md_vehicle.velocity
      self.prev_state.bearing = md_vehicle.bearing
      self.prev_state.position = curr_pos

      # Engagement and movement checks
      is_engaged = state.is_engaged
      if is_engaged and self.first_engage is None:
        self.first_engage = curr_time
        self.op_engaged.set()

      after_engaged_check = is_engaged and curr_time - self.first_engage >= 5 and self.test_run

      x_dist = abs(curr_pos[0] - self.vehicle_last_pos[0])
      y_dist = abs(curr_pos[1] - self.vehicle_last_pos[1])
      dist_threshold = 1
      if x_dist >= dist_threshold or y_dist >= dist_threshold:
        self.distance_moved += x_dist + y_dist

      time_check_threshold = 30
      since_last_check = curr_time - self.last_check_timestamp
      if since_last_check >= time_check_threshold:
        if after_engaged_check and self.distance_moved == 0:
          self.status_q.put(QueueMessage(QueueMessageType.TERMINATION_INFO, {"vehicle_not_moving": True}))
          self.exit_event.set()

        self.last_check_timestamp = curr_time
        self.distance_moved = 0
        self.vehicle_last_pos = curr_pos

  def read_cameras(self):
    pass

  def tick(self):
    pass

  def reset(self):
    self.should_reset = True
    self.prev_state = PreviousState()  # Reset IMU state tracking on vehicle reset

  def close(self, reason: str):
    self.status_q.put(QueueMessage(QueueMessageType.CLOSE_STATUS, reason))
    self.exit_event.set()
    self.metadrive_process.join()
