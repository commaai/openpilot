import functools
import math
import multiprocessing
import numpy as np
import time

from multiprocessing import Pipe
from openpilot.tools.sim.bridge.metadrive.metadrive_process import metadrive_process
from openpilot.tools.sim.lib.common import SimulatorState, World, vec3


class MetaDriveWorld(World):
  def __init__(self, config, dual_camera = False):
    super().__init__(dual_camera)
    self.camera_recv, self.camera_send = Pipe()
    self.controls_send, self.controls_recv = Pipe()
    self.state_send, self.state_recv = Pipe()

    self.metadrive_process = multiprocessing.Process(target=
                              functools.partial(metadrive_process, dual_camera, config, self.camera_send, self.controls_recv, self.state_send))
    self.steer_ratio = 15
    self.vc = [0.0,0.0]
    self.reset_time = 0

  def apply_controls(self, steer_angle, throttle_out, brake_out):
    steer_metadrive = steer_angle * 1 / (self.env.vehicle.MAX_STEERING * self.steer_ratio)
    steer_metadrive = np.clip(steer_metadrive, -1, 1)

    if (time.monotonic() - self.reset_time) > 5:
      self.vc[0] = steer_metadrive

      if throttle_out:
        self.vc[1] = throttle_out/10
      else:
        self.vc[1] = -brake_out
    else:
      self.vc[0] = 0
      self.vc[1] = 0

  def read_sensors(self, state: SimulatorState):
    state.velocity = vec3(x=float(self.env.vehicle.velocity[0]), y=float(self.env.vehicle.velocity[1]), z=0)
    state.gps.from_xy(self.env.vehicle.position)
    state.bearing = float(math.degrees(self.env.vehicle.heading_theta))
    state.steering_angle = self.env.vehicle.steering * self.env.vehicle.MAX_STEERING
    state.valid = True

  def read_cameras(self):
    if self.dual_camera:
     self.wide_road_image = self.get_cam_as_rgb("rgb_wide")
    self.road_image = self.get_cam_as_rgb("rgb_road")

  def tick(self):
    obs, _, terminated, _, info = self.env.step(self.vc)

    if terminated:
      self.reset()

  def reset(self):
    self.env.reset()
    self.reset_time = time.monotonic()

  def close(self):
    pass