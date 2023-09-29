from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.base_camera import _cuda_enable
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.engine.core.engine_core import EngineCore
from metadrive.engine.core.image_buffer import ImageBuffer
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.obs.image_obs import ImageObservation
from panda3d.core import Vec3

from openpilot.tools.sim.bridge.common import SimulatorBridge
from openpilot.tools.sim.bridge.metadrive.metadrive_world import MetaDriveWorld
from openpilot.tools.sim.lib.camerad import W, H


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


def straight_block(length):
  return {
    "id": "S",
    "pre_block_socket_index": 0,
    "length": length
  }

def curve_block(length, angle=45, direction=0):
  return {
    "id": "C",
    "pre_block_socket_index": 0,
    "length": length,
    "radius": length,
    "angle": angle,
    "dir": direction
  }


class MetaDriveBridge(SimulatorBridge):
  TICKS_PER_FRAME = 2

  def __init__(self, args):
    self.should_render = False

    super(MetaDriveBridge, self).__init__(args)

  def spawn_world(self):
    print("----------------------------------------------------------")
    print("---- Spawning Metadrive world, this might take awhile ----")
    print("----------------------------------------------------------")

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
          map_config=dict(
            type=MapGenerateMethod.PG_MAP_FILE,
            config=[
              None,
              straight_block(120),
              curve_block(120, 90),
              straight_block(120),
              curve_block(120, 90),
              straight_block(120),
              curve_block(120, 90),
              straight_block(120),
              curve_block(120, 90),
            ]
          )
        )
      )

    env.reset()

    return MetaDriveWorld(env)