#!/usr/bin/env python3
import unittest
import os

from openpilot.tools.sim.run_bridge import parse_args
from openpilot.tools.sim.bridge.metadrive.metadrive_bridge import MetaDriveBridge
from openpilot.tools.sim.tests.test_sim_bridge import TestSimBridgeBase

class TestMetaDriveBridge(TestSimBridgeBase):
  drive_time = 60
  def create_bridge(self):
    return MetaDriveBridge(parse_args([]))

if __name__ == "__main__":
  TestMetaDriveBridge.drive_time = int(os.getenv("DRIVE_TIME", TestMetaDriveBridge.drive_time))
  unittest.main()
