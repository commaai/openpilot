import os
from typing import cast

from openpilot.common.params import Params
from openpilot.system.hardware.base import HardwareBase
from openpilot.system.hardware.tici.hardware import Tici
from openpilot.system.hardware.pc.hardware import Pc

def ublox_available() -> bool:
  return os.path.exists('/dev/ttyHS0') and not os.path.exists('/persist/comma/use-quectel-gps')


def ublox(params) -> bool:
  use_ublox = ublox_available()
  if use_ublox != params.get_bool("UbloxAvailable"):
    params.put_bool("UbloxAvailable", use_ublox)
  return use_ublox


TICI = os.path.isfile('/TICI')
AGNOS = os.path.isfile('/AGNOS')
TIZI = TICI and not ublox(Params())
PC = not TICI
SIM = os.environ.get("SIM", None) is not None


if TICI:
  HARDWARE = cast(HardwareBase, Tici())
else:
  HARDWARE = cast(HardwareBase, Pc())
