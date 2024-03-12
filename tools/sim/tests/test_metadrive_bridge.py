#!/usr/bin/env python3
import pytest
import unittest

from openpilot.tools.sim.bridge.metadrive.metadrive_bridge import MetaDriveBridge
from openpilot.tools.sim.tests.test_sim_bridge import TestSimBridgeBase

@pytest.mark.slow
class TestMetaDriveBridge(TestSimBridgeBase):
  def create_bridge(self):
    return MetaDriveBridge(False, False)


if __name__ == "__main__":
  unittest.main()
