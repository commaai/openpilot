#!/usr/bin/env python3
import pytest
import unittest
import sys

from openpilot.tools.sim.bridge.metadrive.metadrive_bridge import MetaDriveBridge
from openpilot.tools.sim.tests.test_sim_bridge import TestSimBridgeBase

@pytest.mark.slow
class TestMetaDriveBridge(TestSimBridgeBase):
  TRACK_SIZE = 60

  def create_bridge(self):
    return MetaDriveBridge(False, False, self.TRACK_SIZE, True)


if __name__ == "__main__":
  if len(sys.argv) > 1:
    TestMetaDriveBridge.TRACK_SIZE = int(sys.argv.pop())
  unittest.main()
