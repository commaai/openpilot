from collections import namedtuple
from multiprocessing import Queue

from metadrive.component.sensors.base_camera import _cuda_enable
from metadrive.component.map.pg_map import MapGenerateMethod

from openpilot.tools.sim.bridge.common import SimulatorBridge
from openpilot.tools.sim.bridge.metadrive.metadrive_common import RGBCameraRoad, RGBCameraWide
from openpilot.tools.sim.bridge.metadrive.metadrive_world import MetaDriveWorld
from openpilot.tools.sim.lib.camerad import W, H


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

def create_map(track_size=60, ci=False):
  mtd_map = dict(
    type=MapGenerateMethod.PG_MAP_FILE,
    lane_num=2,
    lane_width=4,
    config=[
      None,
      straight_block(track_size),
      curve_block(track_size*2, 90),
      straight_block(track_size),
      curve_block(track_size*2, 90),
      straight_block(track_size),
      curve_block(track_size*2, 90),
      straight_block(track_size),
      curve_block(track_size*2, 90),
    ]
  )

  # None block is to make sure we have a complete loop, but having this would make metadrive detect wrong destination for run in CI to complete
  if ci:
    mtd_map["config"] = mtd_map["config"][1:]

  return mtd_map

done_config = namedtuple("done_config",
                              ["out_of_route_done", "on_continuous_line_done", "arrive_dest"],
                              defaults=[False, False, False])

class MetaDriveBridge(SimulatorBridge):
  TICKS_PER_FRAME = 5

  def __init__(self, dual_camera, high_quality, track_size=60, ci=False):
    self.should_render = False
    self.done_config = done_config(True, True, True) if ci else done_config()
    self.track_size = track_size
    self.ci = ci

    super().__init__(dual_camera, high_quality)

  def spawn_world(self, queue: Queue):
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
      ),
      sensors=sensors,
      image_on_cuda=_cuda_enable,
      image_observation=True,
      interface_panel=[],
      out_of_route_done=self.done_config.out_of_route_done,
      on_continuous_line_done=self.done_config.on_continuous_line_done,
      crash_vehicle_done=False,
      crash_object_done=False,
      arrive_dest_done=self.done_config.arrive_dest,
      traffic_density=0.0, # traffic is incredibly expensive
      map_config=create_map(self.track_size, self.ci),
      decision_repeat=1,
      physics_world_step_size=self.TICKS_PER_FRAME/100,
      preload_models=False
    )

    return MetaDriveWorld(queue, config, self.dual_camera, self.ci)
