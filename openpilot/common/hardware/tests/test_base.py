import errno
import math
import unittest
from unittest import mock

from openpilot.common.hardware.base import ThermalZone, max_valid_temperature


class TestMaxValidTemperature(unittest.TestCase):
  def test_ignores_non_finite_readings(self):
    self.assertEqual(max_valid_temperature([math.nan, 42.5, math.inf, 38.0]), 42.5)

  def test_returns_none_without_valid_readings(self):
    self.assertIsNone(max_valid_temperature([math.nan, math.inf, -math.inf]))


class TestThermalZone(unittest.TestCase):
  def test_read_returns_nan_on_io_error(self):
    zone = ThermalZone("pm8005_tz")
    zone.zone_number = 4

    with mock.patch("builtins.open", side_effect=OSError(errno.EIO, "Input/output error")):
      self.assertTrue(math.isnan(zone.read()))

  def test_read_recovers_after_transient_io_error(self):
    zone = ThermalZone("pm8005_tz")
    zone.zone_number = 4
    temperature_file = mock.mock_open(read_data="42000").return_value

    with mock.patch("builtins.open", side_effect=[OSError(errno.EIO, "Input/output error"), temperature_file]):
      self.assertTrue(math.isnan(zone.read()))
      self.assertEqual(zone.read(), 42.0)

  def test_missing_zone_is_rediscovered(self):
    zone = ThermalZone("pm8005_tz")
    zone.zone_number = 4

    with mock.patch("builtins.open", side_effect=FileNotFoundError):
      self.assertTrue(math.isnan(zone.read()))
    self.assertEqual(zone.zone_number, -1)
