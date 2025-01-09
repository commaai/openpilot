from multiprocessing import Queue
from openpilot.tools.sim.bridge.common import SimulatorBridge
from openpilot.tools.sim.bridge.gz.gz_world import GZWorld

class GZDriveBridge(SimulatorBridge):
  def __init__(self, dual_camera, high_quality):
    super().__init__(dual_camera, high_quality)

  def spawn_world(self, q: Queue):
    world =  GZWorld(q)
    world.start_vehicle()
    return world
