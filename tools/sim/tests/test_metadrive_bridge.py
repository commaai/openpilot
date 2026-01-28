import pytest
import warnings
import time
import subprocess
from multiprocessing import Queue
import numpy as np

# Since metadrive depends on pkg_resources, and pkg_resources is deprecated as an API
warnings.filterwarnings("ignore", category=DeprecationWarning)

from cereal import messaging
from openpilot.tools.sim.bridge.metadrive.metadrive_bridge import MetaDriveBridge
from openpilot.tools.sim.tests.test_sim_bridge import TestSimBridgeBase, SIM_DIR

@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::pyopencl.CompilerWarning") # Unimportant warning of non-empty compile log
class TestMetaDriveBridge(TestSimBridgeBase):
  @pytest.fixture(autouse=True)
  def setup_create_bridge(self, test_duration):
    self.test_duration = 30

  def create_bridge(self):
    return MetaDriveBridge(False, False, self.test_duration, True)

  def test_imu_data(self):
    p_manager = subprocess.Popen("./launch_openpilot.sh", cwd=SIM_DIR)
    self.processes.append(p_manager)

    sm = messaging.SubMaster(['accelerometer', 'gyroscope', 'selfdriveState'])
    q = Queue()
    bridge = self.create_bridge()
    p_bridge = bridge.run(q, retries=10)
    self.processes.append(p_bridge)

    max_time_per_step = 60

    # Wait for bridge to startup
    start_waiting = time.monotonic()
    while not bridge.started.value and time.monotonic() < start_waiting + max_time_per_step:
      time.sleep(0.1)
    assert p_bridge.exitcode is None, f"Bridge process should be running, but exited with code {p_bridge.exitcode}"

    # Wait for engagement
    start_time = time.monotonic()
    engaged = False
    while time.monotonic() < start_time + max_time_per_step:
      sm.update()
      if sm.updated['selfdriveState'] and sm['selfdriveState'].active:
        engaged = True
        break
      time.sleep(0.1)
    assert engaged, "openpilot did not engage"

    start_time = time.monotonic()
    accel_values = []
    gyro_values = []
    # run for 10 seconds and collect some imu data
    while time.monotonic() < start_time + 10:
      sm.update()
      if sm.updated['accelerometer']:
        accel_values.append(list(sm['accelerometer'].acceleration.v))
      if sm.updated['gyroscope']:
        gyro_values.append(list(sm['gyroscope'].gyroUncalibrated.v))
      time.sleep(0.1)

    assert len(accel_values) > 10
    assert len(gyro_values) > 10

    # Check that the values are not all the same
    accel_std = np.std(np.array(accel_values), axis=0)
    gyro_std = np.std(np.array(gyro_values), axis=0)

    assert any(x > 1e-3 for x in accel_std), f"accel_std: {accel_std}"
    assert any(x > 1e-5 for x in gyro_std), f"gyro_std: {gyro_std}"  # Lower threshold for gyroscope
