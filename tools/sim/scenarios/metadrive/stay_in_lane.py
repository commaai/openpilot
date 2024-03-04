#!/usr/bin/env python

from typing import Any
from multiprocessing import Queue

from metadrive.component.sensors.base_camera import _cuda_enable
from metadrive.component.map.pg_map import MapGenerateMethod

from openpilot.tools.sim.bridge.common import SimulatorBridge
from openpilot.tools.sim.bridge.metadrive.metadrive_common import RGBCameraRoad, RGBCameraWide
from openpilot.tools.sim.bridge.metadrive.metadrive_world import MetaDriveWorld
from openpilot.tools.sim.lib.camerad import W, H


def create_map():
  return dict(
    type=MapGenerateMethod.PG_MAP_FILE,
    lane_num=2,
    lane_width=3.5,
    config=[
      {
        "id": "S",
        "pre_block_socket_index": 0,
        "length": 60,
      },
      {
        "id": "C",
        "pre_block_socket_index": 0,
        "length": 60,
        "radius": 600,
        "angle": 45,
        "dir": 0,
      },
    ]
  )


class MetaDriveBridge(SimulatorBridge):
  TICKS_PER_FRAME = 5

  def __init__(self, dual_camera, high_quality):
    self.should_render = False

    super().__init__(dual_camera, high_quality)

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
      ),
      sensors=sensors,
      image_on_cuda=_cuda_enable,
      image_observation=True,
      interface_panel=[],
      out_of_route_done=True,
      on_continuous_line_done=True,
      crash_vehicle_done=True,
      crash_object_done=True,
      traffic_density=0.0,
      map_config=create_map(),
      decision_repeat=1,
      physics_world_step_size=self.TICKS_PER_FRAME/100,
      preload_models=False
    )

    return MetaDriveWorld(config, self.dual_camera)


if __name__ == "__main__":
  queue: Any = Queue()
  simulator_bridge = MetaDriveBridge(True, False)
  simulator_process = simulator_bridge.run(queue)
  simulator_process.join()
