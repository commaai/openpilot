#!/usr/bin/env python3
import pytest
import warnings

# Since metadrive depends on pkg_resources, and pkg_resources is deprecated as an API
warnings.filterwarnings("ignore", category=DeprecationWarning)

from openpilot.tools.sim.bridge.metadrive.metadrive_bridge import MetaDriveBridge
from openpilot.tools.sim.tests.test_sim_bridge import TestSimBridgeBase

@pytest.mark.slow
class TestMetaDriveBridge(TestSimBridgeBase):
  @pytest.fixture(autouse=True)
  def setup_class(self, time_done):
    self.time_done = time_done

  def create_bridge(self):
    return MetaDriveBridge(False, False, self.time_done, True)
