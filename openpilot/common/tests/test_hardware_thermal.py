import math
import unittest
from unittest.mock import mock_open, patch

from openpilot.common.hardware.base import ThermalZone, max_valid_temperature


READ_ERROR = OSError("thermal read failed")
THERMAL_ZONE_NAME = "pm8005_tz"
HOT_TEMPERATURE = 91.0
WARM_TEMPERATURE = 72.0
THERMAL_FALLBACK_TEMPERATURE = 0.0


class TestHardwareThermal(unittest.TestCase):
  def test_thermal_read_error_logs_nan(self):
    zone = ThermalZone(THERMAL_ZONE_NAME)
    zone.zone_number = 4

    with patch("builtins.open", mock_open()) as open_mock:
      open_mock.side_effect = READ_ERROR

      assert math.isnan(zone.read())

  def test_max_valid_temperature_ignores_nan_readings(self):
    assert max_valid_temperature([math.nan, HOT_TEMPERATURE, WARM_TEMPERATURE]) == HOT_TEMPERATURE

  def test_max_valid_temperature_falls_back_when_all_readings_invalid(self):
    assert max_valid_temperature([math.nan, math.nan]) == THERMAL_FALLBACK_TEMPERATURE
