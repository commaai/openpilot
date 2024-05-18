#!/usr/bin/env python3
import pytest
import warnings

# Since metadrive depends on pkg_resources, and pkg_resources is deprecated as an API
warnings.filterwarnings("ignore", category=DeprecationWarning)

from openpilot.tools.sim.bridge.metadrive.metadrive_bridge import MetaDriveBridge
from openpilot.tools.sim.tests.test_sim_bridge import TestSimBridgeBase

@pytest.mark.slow
class TestMetaDriveBridge(TestSimBridgeBase):
  TRACK_SIZE = 60

  @pytest.fixture(autouse=True)
  def setup_class(self, track_size):
    self.track_size = track_size

  def create_bridge(self):
    return MetaDriveBridge(False, False, self.track_size, True)