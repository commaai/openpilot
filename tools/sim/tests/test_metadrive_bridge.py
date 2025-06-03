import pytest
import warnings
import math
# Since metadrive depends on pkg_resources, and pkg_resources is deprecated as an API
warnings.filterwarnings("ignore", category=DeprecationWarning)

from openpilot.tools.sim.bridge.metadrive.metadrive_bridge import MetaDriveBridge
from openpilot.tools.sim.tests.test_sim_bridge import TestSimBridgeBase

@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::pyopencl.CompilerWarning") # Unimportant warning of non-empty compile log
class TestMetaDriveBridge(TestSimBridgeBase):
  @pytest.fixture(autouse=True)
  def setup_create_bridge(self, test_duration):
    self.test_duration = 30

  def create_bridge(self):
    return MetaDriveBridge(False, False, self.test_duration, True)
def test_imu_update():
  from openpilot.tools.sim.lib.common import IMUState, vec3

  imu = IMUState()
  last_vel = vec3(0, 0, 0)
  current_vel = vec3(10, 0, 0)
  last_bearing = 0
  current_bearing = 90
  dt = 1.0

  IMUState.update_imu_state(imu, current_vel, last_vel, current_bearing, last_bearing, dt)

  assert math.isclose(imu.accelerometer.x, 10)
  assert math.isclose(imu.accelerometer.y, 0)
  assert math.isclose(imu.gyroscope.z, 90)
  assert math.isclose(imu.bearing, 90)
