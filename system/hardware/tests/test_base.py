import pytest

from openpilot.system.hardware.base import ThermalZone


class TestThermalZone:
  def test_read(self, mocker):
    mocker.patch("builtins.open", mocker.mock_open(read_data="48000"))
    zone = ThermalZone("cpu")
    zone.zone_number = 0
    assert zone.read() == 48.0

  @pytest.mark.parametrize("exc", [FileNotFoundError, PermissionError, OSError, ValueError])
  def test_read_failure_does_not_raise(self, mocker, exc):
    # a failing sensor read (e.g. a PMIC/regmap transaction error) must not crash hardwared
    mocker.patch("builtins.open", side_effect=exc)
    zone = ThermalZone("cpu")
    zone.zone_number = 0
    assert zone.read() == 0
