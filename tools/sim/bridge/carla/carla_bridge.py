import carla

from openpilot.tools.sim.bridge.common import SimulatorBridge
from openpilot.tools.sim.bridge.carla.carla_world import CarlaWorld


class CarlaBridge(SimulatorBridge):
  TICKS_PER_FRAME = 5

  def __init__(self, arguments):
    super().__init__(arguments)
    self.host = arguments.host
    self.port = arguments.port
    self.town = arguments.town
    self.num_selected_spawn_point = arguments.num_selected_spawn_point

  def spawn_world(self):
    client = carla.Client(self.host, self.port)
    client.set_timeout(5)

    return CarlaWorld(client, high_quality=self.high_quality, dual_camera=self.dual_camera,
                      num_selected_spawn_point=self.num_selected_spawn_point, town=self.town)