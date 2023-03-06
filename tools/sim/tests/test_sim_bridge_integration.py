#!/usr/bin/env python3
from contextlib import AbstractContextManager
import subprocess
import time
from typing import List
import unittest
import os
from multiprocessing import Queue
from tools.sim.bridge.carla import CarlaBridge

from cereal import messaging
from common.basedir import BASEDIR
from selfdrive.manager.helpers import unblock_stdout
from tools.sim.bridge.common import parse_args
from tools.sim.bridge.metadrive import MetaDriveBridge

CI = "CI" in os.environ

SIM_DIR = os.path.join(BASEDIR, "tools/sim")

class TestSimBridgeIntegrationSmoke(unittest.TestCase):
  processes: List[subprocess.Popen] = []

  def test_engage(self):
    # Startup manager and bridge.py. Check processes are running, then engage and verify.
    for simulator in ["carla", "metadrive"]:
      with self.subTest(simulator=simulator), ProcessHolder() as procs:
        q = Queue()
        if simulator == "carla":
          try:
            subprocess.check_output('nvidia-smi')
          except FileNotFoundError as e:
            print("Skipping: CARLA requires CUDA to run", e)
            continue
          procs.setUpCarla()
          sim_bridge = CarlaBridge(parse_args([]))
        elif simulator == "metadrive":
          sim_bridge = MetaDriveBridge(parse_args(["--ticks_per_frame", "10"]))
        else:
          raise AssertionError("Unsupported simulator")
        p_bridge = sim_bridge.run(q, retries=10)
        procs.processes.append(p_bridge)
        time.sleep(10.0)

        p_manager = subprocess.Popen("./launch_openpilot.sh", cwd=SIM_DIR)
        procs.processes.append(p_manager)

        sm = messaging.SubMaster(['controlsState', 'carEvents', 'managerState'])

        max_time_per_step = 60

        # Wait for bridge to startup
        start_waiting = time.monotonic()
        while not sim_bridge.started and time.monotonic() < start_waiting + max_time_per_step:
          time.sleep(0.1)
        self.assertEqual(p_bridge.exitcode, None, f"Bridge process should be running, but exited with code {p_bridge.exitcode}")

        start_time = time.monotonic()
        no_car_events_issues_once = False
        car_event_issues = []
        not_running = []
        while time.monotonic() < start_time + max_time_per_step:
          sm.update()

          not_running = [p.name for p in sm['managerState'].processes if not p.running and p.shouldBeRunning]
          car_event_issues = [event.name for event in sm['carEvents'] if any([event.noEntry, event.softDisable, event.immediateDisable])]

          if sm.all_alive() and len(car_event_issues) == 0 and len(not_running) == 0:
            no_car_events_issues_once = True
            break

        self.assertTrue(no_car_events_issues_once, f"Failed because no messages received, or CarEvents '{car_event_issues}' or processes not running '{not_running}'")

        start_time = time.monotonic()
        min_counts_control_active = 100
        control_active = 0

        while time.monotonic() < start_time + max_time_per_step:
          sm.update()

          q.put("cruise_down")  # Try engaging

          if sm.all_alive() and sm['controlsState'].active:
            control_active += 1

            if control_active == min_counts_control_active:
              break

        self.assertEqual(min_counts_control_active, control_active, f"Simulator did not engage a minimal of {min_counts_control_active} steps was {control_active}")

class ProcessHolder(AbstractContextManager):
  def __init__(self):
    self.processes = []
    self.carla_process = None

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    for p in reversed(self.processes):
      p.terminate()

    for p in reversed(self.processes):
      if isinstance(p, subprocess.Popen):
        p.wait(15)
      else:
        p.join(15)

    if self.carla_process is not None:
      # Stop carla simulator by removing docker container
      subprocess.run("docker rm -f carla_sim", shell=True, stderr=subprocess.PIPE, check=False)
      self.carla_process.wait()

  def setUpCarla(self):
    if not CI:
      # We want to make sure that carla_sim docker isn't still running.
      subprocess.run("docker rm -f carla_sim", shell=True, stderr=subprocess.PIPE, check=False)
      self.carla_process = subprocess.Popen("./start_carla.sh", cwd=SIM_DIR)

    # Too many lagging messages in bridge.py can cause a crash. This prevents it.
    unblock_stdout()
    # Wait 10 seconds to startup carla
    time.sleep(10)


if __name__ == "__main__":
  unittest.main()
