import pytest
import warnings

# Since metadrive depends on pkg_resources, and pkg_resources is deprecated as an API
warnings.filterwarnings("ignore", category=DeprecationWarning)

from openpilot.tools.sim.bridge.metadrive.metadrive_bridge import MetaDriveBridge
from openpilot.tools.sim.tests.test_sim_bridge import TestSimBridgeBase

@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::pyopencl.CompilerWarning") # Unimportant warning of non-empty compile log
class TestMetaDriveBridge(TestSimBridgeBase):
  @pytest.fixture(autouse=True)
  def setup_create_bridge(self):
    self.test_duration = 15
    self.minimal_distance = 10

  def create_bridge(self):
    return MetaDriveBridge(False, False, self.test_duration, self.minimal_distance, True)
