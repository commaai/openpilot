
from openpilot.common.parameterized import parameterized
from openpilot.common.test import OpenpilotTestCase
from openpilot.system.hardware.fan_controller import FanController, voice_fan_limit, VOICE_IDLE_FAN_CAP

ALL_CONTROLLERS = [FanController]

class TestFanController(OpenpilotTestCase):
  def wind_up(self, controller, ignition=True):
    for _ in range(1000):
      controller.update(100, ignition)

  def wind_down(self, controller, ignition=False):
    for _ in range(1000):
      controller.update(10, ignition)

  @parameterized.expand(ALL_CONTROLLERS)
  def test_hot_onroad(self, controller_class):
    controller = controller_class(2)
    self.wind_up(controller)
    assert controller.update(100, True) >= 70

  @parameterized.expand(ALL_CONTROLLERS)
  def test_offroad_limits(self, controller_class):
    controller = controller_class(2)
    self.wind_up(controller)
    assert controller.update(100, False) <= 30

  @parameterized.expand(ALL_CONTROLLERS)
  def test_no_fan_wear(self, controller_class):
    controller = controller_class(2)
    self.wind_down(controller)
    assert controller.update(10, False) == 0

  @parameterized.expand(ALL_CONTROLLERS)
  def test_limited(self, controller_class):
    controller = controller_class(2)
    self.wind_up(controller, True)
    assert controller.update(100, True) == 100

  @parameterized.expand(ALL_CONTROLLERS)
  def test_windup_speed(self, controller_class):
    controller = controller_class(2)
    self.wind_down(controller, True)
    for _ in range(10):
      controller.update(90, True)
    assert controller.update(90, True) >= 60


def test_voice_fan_limit():
  # thermals win above the limit, otherwise cap while listening and duck during a query
  assert voice_fan_limit(90, 80., True, True) == 90
  assert voice_fan_limit(90, 60., True, False) == VOICE_IDLE_FAN_CAP
  assert voice_fan_limit(90, 60., True, True) == 0
  assert voice_fan_limit(40, 60., True, False) == 40
  assert voice_fan_limit(90, 60., False, True) == 90
