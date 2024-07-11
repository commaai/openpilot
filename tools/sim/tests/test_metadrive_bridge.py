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
  def setup_create_bridge(self, test_duration):
    # run bridge test for at least 60s, since not-moving check runs every 30s
    if test_duration < 60:
      test_duration = 60
    self.test_duration = test_duration

  def create_bridge(self):
    return MetaDriveBridge(False, False, self.test_duration, True)
