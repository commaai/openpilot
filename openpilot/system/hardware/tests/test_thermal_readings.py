import errno
import math
from unittest import mock

from openpilot.common.test import OpenpilotTestCase
from openpilot.common.hardware.base import ThermalZone
from openpilot.system.hardware.hardwared import max_valid_temp


class TestThermalZone(OpenpilotTestCase):
  def setUp(self):
    self.zone = ThermalZone(name="cpu0-silver-usr")
    self.zone.zone_number = 1  # skip discovery

  def test_read_ok(self):
    with mock.patch("builtins.open", mock.mock_open(read_data="45000")):
      assert self.zone.read() == 45.0

  def test_read_io_error(self):
    # the failure mode from the issue: spmi transaction fails, read gets EIO
    with mock.patch("builtins.open", side_effect=OSError(errno.EIO, "read failed")):
      assert math.isnan(self.zone.read())

  def test_read_garbage(self):
    with mock.patch("builtins.open", mock.mock_open(read_data="not a number")):
      assert math.isnan(self.zone.read())

  def test_read_missing_zone(self):
    with mock.patch("builtins.open", side_effect=FileNotFoundError):
      assert math.isnan(self.zone.read())


class TestMaxValidTemp(OpenpilotTestCase):
  def test_no_sensors_configured(self):
    # platforms without these sensors keep today's 0.0 behavior
    assert max_valid_temp([]) == 0.

  def test_all_valid(self):
    assert max_valid_temp([40., 55., 45.]) == 55.

  def test_ignores_failed_reads(self):
    assert max_valid_temp([40., float("nan"), 55.]) == 55.

  def test_all_failed_propagates(self):
    # configured-but-failing must not silently read as 0
    assert math.isnan(max_valid_temp([float("nan"), float("nan")]))

  def test_order_independent(self):
    # python's builtin max() with NaN is order-dependent, which is the bug
    # this helper exists to avoid: max(nan, 55) is nan but max(55, nan) is 55
    a = [float("nan"), 55.]
    b = [55., float("nan")]
    assert max_valid_temp(a) == max_valid_temp(b) == 55.
