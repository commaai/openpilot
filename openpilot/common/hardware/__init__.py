import os
from typing import cast

from openpilot.common.hardware.base import HardwareBase
from openpilot.common.hardware.tici.hardware import Tici
from openpilot.common.hardware.pc.hardware import Pc

TICI = os.path.isfile('/TICI')
AGNOS = os.path.isfile('/AGNOS')
PC = not TICI


if TICI:
  HARDWARE = cast(HardwareBase, Tici())
else:
  HARDWARE = cast(HardwareBase, Pc())
