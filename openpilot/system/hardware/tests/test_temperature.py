from openpilot.common.test import OpenpilotTestCase
from openpilot.system.hardware.temperature import format_temperature


class TestTemperature(OpenpilotTestCase):
  def test_format_metric_temperature(self):
    assert format_temperature(25.0, True) == "25.0°C"

  def test_format_imperial_temperature(self):
    assert format_temperature(25.0, False) == "77.0°F"
