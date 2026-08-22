import errno
import math
import unittest
from unittest.mock import mock_open, patch

from openpilot.common.hardware.base import ThermalZone
from openpilot.system.hardware.hardwared import max_valid_temperature


class TestThermalZone(unittest.TestCase):
  def test_read(self):
    zone = ThermalZone("cpu", scale=1000.)
    zone.zone_number = 2
    with patch("builtins.open", mock_open(read_data="42500")):
      assert zone.read() == 42.5

  def test_read_failure(self):
    for error in (OSError(errno.EIO, "I/O error"), FileNotFoundError(), ValueError()):
      with self.subTest(error=error):
        zone = ThermalZone("cpu")
        zone.zone_number = 2
        with patch("builtins.open", side_effect=error):
          assert math.isnan(zone.read())

  def test_discovery_failure(self):
    with patch("os.listdir", side_effect=OSError(errno.EIO, "I/O error")):
      assert math.isnan(ThermalZone("cpu").read())

  def test_max_valid_temperature(self):
    cases = (
      ([math.nan, 50., 42.], 50.),
      ([math.nan, math.nan], 0.),
      ([], 0.),
    )
    for temperatures, expected in cases:
      with self.subTest(temperatures=temperatures):
        assert max_valid_temperature(temperatures) == expected
