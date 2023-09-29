import math
import numpy as np

from collections import namedtuple
from multiprocessing.connection import Connection

from metadrive.engine.core.engine_core import EngineCore
from metadrive.engine.core.image_buffer import ImageBuffer
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.obs.image_obs import ImageObservation

from openpilot.common.realtime import Ratekeeper
from openpilot.tools.sim.lib.common import vec3
from openpilot.tools.sim.lib.camerad import W, H


metadrive_state = namedtuple("metadrive_state", ["velocity", "position", "bearing", "steering_angle"])

def apply_metadrive_patches():
  # By default, metadrive won't try to use cuda images unless it's used as a sensor for vehicles, so patch that in
  def add_image_sensor_patched(self, name: str, cls, args):
    if self.global_config["image_on_cuda"]:# and name == self.global_config["vehicle_config"]["image_source"]:
        sensor = cls(*args, self, cuda=True)
    else:
        sensor = cls(*args, self, cuda=False)
    assert isinstance(sensor, ImageBuffer), "This API is for adding image sensor"
    self.sensors[name] = sensor

  EngineCore.add_image_sensor = add_image_sensor_patched

  # we aren't going to use the built-in observation stack, so disable it to save time
  def observe_patched(self, vehicle):
    return self.state

  ImageObservation.observe = observe_patched

  def arrive_destination_patch(self, vehicle):
    return False

  MetaDriveEnv._is_arrive_destination = arrive_destination_patch

def metadrive_process(dual_camera: bool, config: dict, camera_array, controls_recv: Connection, state_send: Connection, exit_event):
  apply_metadrive_patches()

  road_image = np.frombuffer(camera_array.get_obj(), dtype=np.uint8).reshape((H, W, 3))

  env = MetaDriveEnv(config)
  env.reset()

  def get_cam_as_rgb(cam):
    cam = env.engine.sensors[cam]
    img = cam.perceive(env.vehicle, clip=False)
    if type(img) != np.ndarray:
      img = img.get() # convert cupy array to numpy
    return img

  rk = Ratekeeper(100, None)

  steer_ratio = 15
  vc = [0,0]

  while not exit_event.is_set():
    state = metadrive_state(
      velocity=vec3(x=float(env.vehicle.velocity[0]), y=float(env.vehicle.velocity[1]), z=0),
      position=env.vehicle.position,
      bearing=float(math.degrees(env.vehicle.heading_theta)),
      steering_angle=env.vehicle.steering * env.vehicle.MAX_STEERING
    )

    state_send.send(state)

    if controls_recv.poll(0):
      while controls_recv.poll(0):
        steer_angle, gas, reset = controls_recv.recv()

      steer_metadrive = steer_angle * 1 / (env.vehicle.MAX_STEERING * steer_ratio)
      steer_metadrive = np.clip(steer_metadrive, -1, 1)

      vc = [steer_metadrive, gas]

      if reset:
        env.reset()

    if rk.frame % 5 == 0:
      obs, _, terminated, _, info = env.step(vc)

      if terminated:
        env.reset()

      #if dual_camera:
      #  wide_road_image = get_cam_as_rgb("rgb_wide")
      road_image[...] = get_cam_as_rgb("rgb_road")

    rk.keep_time()