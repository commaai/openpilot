import warnings

# Since metadrive depends on pkg_resources, and pkg_resources is deprecated as an API
warnings.filterwarnings("ignore", category=DeprecationWarning)

from openpilot.selfdrive.test.helpers import slow
from openpilot.tools.sim.bridge.metadrive.metadrive_bridge import MetaDriveBridge
from openpilot.tools.sim.tests.test_sim_bridge import TestSimBridgeBase

@slow
class TestMetaDriveBridge(TestSimBridgeBase):
  def setUp(self):
    super().setUp()
    self.test_duration = 30

  def create_bridge(self):
    return MetaDriveBridge(False, False, self.test_duration, True)
