import math
import numpy as np

from collections import namedtuple

from multiprocessing.connection import Connection
from metadrive.envs.metadrive_env import MetaDriveEnv

from openpilot.common.realtime import Ratekeeper
from openpilot.tools.sim.lib.common import vec3

metadrive_state = namedtuple("metadrive_state", ["velocity", "position", "bearing", "steering_angle"])


def metadrive_process(dual_camera: bool, config: dict, camera_send: Connection, controls_recv: Connection, state_send: Connection):
  env = MetaDriveEnv(config)
  env.reset()

  def get_cam_as_rgb(cam):
    cam = env.engine.sensors[cam]
    img = cam.perceive(env.vehicle, clip=False)
    if type(img) != np.ndarray:
      img = img.get() # convert cupy array to numpy
    return img

  rk = Ratekeeper(100)

  steer_ratio = 15
  vc = [0,0]

  while True:
    state = metadrive_state(
      velocity=vec3(x=float(env.vehicle.velocity[0]), y=float(env.vehicle.velocity[1]), z=0),
      position=env.vehicle.position,
      bearing=float(math.degrees(env.vehicle.heading_theta)),
      steering_angle=env.vehicle.steering * env.vehicle.MAX_STEERING
    )

    state_send.send(state)

    while controls_recv.poll(0):
      steer_angle, gas = controls_recv.recv()

      steer_metadrive = steer_angle * 1 / (env.vehicle.MAX_STEERING * steer_ratio)
      steer_metadrive = np.clip(steer_metadrive, -1, 1)

      vc = [steer_metadrive, gas]

    env.step(vc)

    #if dual_camera:
    #  wide_road_image = get_cam_as_rgb("rgb_wide")
    road_image = get_cam_as_rgb("rgb_road")

    camera_send.send(road_image)

    rk.keep_time()