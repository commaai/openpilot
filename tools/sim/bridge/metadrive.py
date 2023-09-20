import math
import numpy as np
import time

from openpilot.tools.sim.bridge.common import World, SimulatorBridge
from openpilot.tools.sim.lib.common import vec3, SimulatorState
from openpilot.tools.sim.lib.camerad import W, H


def apply_metadrive_patches():
  from metadrive.engine.core.engine_core import EngineCore
  from metadrive.engine.core.image_buffer import ImageBuffer
  from metadrive.obs.image_obs import ImageObservation

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


class MetaDriveWorld(World):
  def __init__(self, env, ticks_per_frame: float, dual_camera = False):
    super().__init__(dual_camera)
    self.env = env
    self.ticks_per_frame = ticks_per_frame
    self.dual_camera = dual_camera

    self.steer_ratio = 15

    self.vc = [0.0,0.0]

    self.reset_time = 0

  def get_cam_as_rgb(self, cam):
    cam = self.env.engine.sensors[cam]
    img = cam.perceive(self.env.vehicle, clip=False)
    if type(img) != np.ndarray:
      img = img.get() # convert cupy array to numpy
    return img

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
      self.env.reset()
      self.reset_time = time.monotonic()

  def close(self):
    pass


class MetaDriveBridge(SimulatorBridge):
  TICKS_PER_FRAME = 2

  def __init__(self, args):
    self.should_render = False

    super(MetaDriveBridge, self).__init__(args)

  def spawn_world(self):
    print("----------------------------------------------------------")
    print("---- Spawning Metadrive world, this might take awhile ----")
    print("----------------------------------------------------------")
    from metadrive.component.sensors.rgb_camera import RGBCamera
    from metadrive.component.sensors.base_camera import _cuda_enable
    from metadrive.envs.metadrive_env import MetaDriveEnv
    from panda3d.core import Vec3

    apply_metadrive_patches()

    C3_POSITION = Vec3(0, 0, 1)

    class RGBCameraWide(RGBCamera):
      def __init__(self, *args, **kwargs):
        super(RGBCameraWide, self).__init__(*args, **kwargs)
        cam = self.get_cam()
        cam.setPos(C3_POSITION)
        lens = self.get_lens()
        lens.setFov(160)

    class RGBCameraRoad(RGBCamera):
      def __init__(self, *args, **kwargs):
        super(RGBCameraRoad, self).__init__(*args, **kwargs)
        cam = self.get_cam()
        cam.setPos(C3_POSITION)
        lens = self.get_lens()
        lens.setFov(40)

    sensors = {
      "rgb_road": (RGBCameraRoad, W, H, )
    }

    if self.dual_camera:
      sensors["rgb_wide"] = (RGBCameraWide, W, H)

    env = MetaDriveEnv(
        dict(
          use_render=self.should_render,
          vehicle_config=dict(
            enable_reverse=False,
            image_source="rgb_road",
            spawn_longitude=15
          ),
          sensors=sensors,
          image_on_cuda=_cuda_enable,
          image_observation=True,
          interface_panel=[],
          out_of_route_done=False,
          on_continuous_line_done=False,
          crash_vehicle_done=False,
          crash_object_done=False,
        )
      )

    env.reset()

    return MetaDriveWorld(env, self.TICKS_PER_FRAME)