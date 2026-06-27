import math
from io import StringIO

from openpilot.common.hardware.base import ThermalZone


def test_thermal_zone_read_returns_nan_on_temp_read_error(monkeypatch):
  def fake_open(path):
    if path.endswith("/type"):
      return StringIO("pmic\n")
    if path.endswith("/temp"):
      raise OSError("thermal read failed")
    raise FileNotFoundError

  monkeypatch.setattr("os.listdir", lambda _: ["thermal_zone0"])
  monkeypatch.setattr("builtins.open", fake_open)

  assert math.isnan(ThermalZone("pmic").read())


def test_thermal_zone_read_returns_nan_when_zone_is_missing(monkeypatch):
  monkeypatch.setattr("os.listdir", lambda _: [])

  assert math.isnan(ThermalZone("missing").read())
