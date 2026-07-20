import warnings
import unittest
import importlib

# Since metadrive depends on pkg_resources, and pkg_resources is deprecated as an API
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
  MetaDriveBridge = importlib.import_module("openpilot.tools.sim.bridge.metadrive.metadrive_bridge").MetaDriveBridge
except ModuleNotFoundError:
  MetaDriveBridge = None
from openpilot.tools.sim.tests.test_sim_bridge import TestSimBridgeBase

@unittest.skipIf(MetaDriveBridge is None, "metadrive is not installed")
class TestMetaDriveBridge(TestSimBridgeBase):
  def setup_method(self):
    super().setup_method()
    self.test_duration = 30

  def create_bridge(self):
    assert MetaDriveBridge is not None
    return MetaDriveBridge(False, False, self.test_duration, True)
