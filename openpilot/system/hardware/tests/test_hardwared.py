import errno
import math
from unittest.mock import mock_open, patch

import pytest

from openpilot.common.hardware.base import ThermalZone
from openpilot.system.hardware.hardwared import max_valid_temperature


class TestThermalZone:
  def test_read(self):
    zone = ThermalZone("cpu", scale=1000.)
    zone.zone_number = 2
    with patch("builtins.open", mock_open(read_data="42500")):
      assert zone.read() == 42.5

  @pytest.mark.parametrize("error", [
    OSError(errno.EIO, "I/O error"),
    FileNotFoundError(),
    ValueError(),
  ])
  def test_read_failure(self, error):
    zone = ThermalZone("cpu")
    zone.zone_number = 2
    with patch("builtins.open", side_effect=error):
      assert math.isnan(zone.read())

  def test_discovery_failure(self):
    with patch("os.listdir", side_effect=OSError(errno.EIO, "I/O error")):
      assert math.isnan(ThermalZone("cpu").read())


@pytest.mark.parametrize(("temperatures", "expected"), [
  ([math.nan, 50., 42.], 50.),
  ([math.nan, math.nan], 0.),
  ([], 0.),
])
def test_max_valid_temperature(temperatures, expected):
  assert max_valid_temperature(temperatures) == expected
