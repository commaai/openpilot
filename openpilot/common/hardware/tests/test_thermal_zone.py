import errno
import math
from unittest import mock

from openpilot.common.test import OpenpilotTestCase
from openpilot.common.hardware.base import ThermalZone


class TestThermalZone(OpenpilotTestCase):
  def setUp(self):
    self.zone = ThermalZone(name="cpu0-silver-usr")
    self.zone.zone_number = 1  # skip discovery

  def test_read_ok(self):
    with mock.patch("builtins.open", mock.mock_open(read_data="45000")):
      assert self.zone.read() == 45.0

  def test_read_io_error_returns_nan(self):
    # sensor transaction failures surface as EIO from sysfs (#36690)
    with mock.patch("builtins.open", side_effect=OSError(errno.EIO, "i/o error")):
      assert math.isnan(self.zone.read())

  def test_missing_zone_returns_nan(self):
    with mock.patch("builtins.open", side_effect=FileNotFoundError):
      assert math.isnan(self.zone.read())

  def test_garbage_read_returns_nan(self):
    with mock.patch("builtins.open", mock.mock_open(read_data="")):
      assert math.isnan(self.zone.read())


class TestMaxTemp(OpenpilotTestCase):
  def test_filters_nan(self):
    from openpilot.system.hardware.hardwared import max_temp
    nan = float("nan")
    assert max_temp([40.0, nan, 55.0]) == 55.0
    assert max_temp([nan, 41.0]) == 41.0  # max() alone would return nan here
    assert max_temp([nan, nan]) == 0.0
    assert max_temp([]) == 0.0
