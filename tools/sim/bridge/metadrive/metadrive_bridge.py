import numpy as np

from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.base_camera import _cuda_enable
from metadrive.component.map.pg_map import MapGenerateMethod
from panda3d.core import Vec3, Texture, GraphicsOutput

from openpilot.tools.sim.bridge.common import SimulatorBridge
from openpilot.tools.sim.bridge.metadrive.metadrive_world import MetaDriveWorld
from openpilot.tools.sim.lib.camerad import W, H


C3_POSITION = Vec3(0, 0, 1)


class CopyRamRGBCamera(RGBCamera):
  """Camera which copies its content into RAM during the render process, for faster image grabbing."""
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.cpu_texture = Texture()
    self.buffer.addRenderTexture(self.cpu_texture, GraphicsOutput.RTMCopyRam)

  def get_rgb_array_cpu(self):
    origin_img = self.cpu_texture
    img = np.frombuffer(origin_img.getRamImage().getData(), dtype=np.uint8)
    img = img.reshape((origin_img.getYSize(), origin_img.getXSize(), -1))
    img = img[:,:,:3] # RGBA to RGB
    # img = np.swapaxes(img, 1, 0)
    img = img[::-1] # Flip on vertical axis
    return img


class RGBCameraWide(CopyRamRGBCamera):
  def __init__(self, *args, **kwargs):
    super(RGBCameraWide, self).__init__(*args, **kwargs)
    cam = self.get_cam()
    cam.setPos(C3_POSITION)
    lens = self.get_lens()
    lens.setFov(120)
    lens.setNear(0.1)

class RGBCameraRoad(CopyRamRGBCamera):
  def __init__(self, *args, **kwargs):
    super(RGBCameraRoad, self).__init__(*args, **kwargs)
    cam = self.get_cam()
    cam.setPos(C3_POSITION)
    lens = self.get_lens()
    lens.setFov(40)
    lens.setNear(0.1)


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
  TICKS_PER_FRAME = 5

  def __init__(self, args):
    self.should_render = False

    super(MetaDriveBridge, self).__init__(args)

  def spawn_world(self):
    sensors = {
      "rgb_road": (RGBCameraRoad, W, H, )
    }

    if self.dual_camera:
      sensors["rgb_wide"] = (RGBCameraWide, W, H)

    config = dict(
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
      traffic_density=0.0, # traffic is incredibly expensive
      map_config=dict(
        type=MapGenerateMethod.PG_MAP_FILE,
        config=[
          None,
          straight_block(120),
          curve_block(240, 90),
          straight_block(120),
          curve_block(240, 90),
          straight_block(120),
          curve_block(240, 90),
          straight_block(120),
          curve_block(240, 90),
        ]
      ),
      decision_repeat=1,
      physics_world_step_size=self.TICKS_PER_FRAME/100,
      preload_models=False
    )

    return MetaDriveWorld(config, self.dual_camera)
