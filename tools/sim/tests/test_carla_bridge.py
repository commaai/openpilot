#!/usr/bin/env python3
import subprocess
import time
import unittest

from openpilot.selfdrive.manager.helpers import unblock_stdout
from openpilot.tools.sim.run_bridge import parse_args
from openpilot.tools.sim.bridge.carla.carla_bridge import CarlaBridge
from openpilot.tools.sim.tests.test_sim_bridge import SIM_DIR, TestSimBridgeBase


class TestCarlaBridge(TestSimBridgeBase):
  """
  Tests need Carla simulator to run
  """
  carla_process = None

  def setUp(self):
    super().setUp()

    # We want to make sure that carla_sim docker isn't still running.
    subprocess.run("docker rm -f carla_sim", shell=True, stderr=subprocess.PIPE, check=False)
    self.carla_process = subprocess.Popen("./start_carla.sh", cwd=SIM_DIR)

    # Too many lagging messages in bridge.py can cause a crash. This prevents it.
    unblock_stdout()
    # Wait 10 seconds to startup carla
    time.sleep(10)

  def create_bridge(self):
    return CarlaBridge(parse_args([]))

  def tearDown(self):
    super().tearDown()

    # Stop carla simulator by removing docker container
    subprocess.run("docker rm -f carla_sim", shell=True, stderr=subprocess.PIPE, check=False)
    if self.carla_process is not None:
      self.carla_process.wait()


if __name__ == "__main__":
  unittest.main()
