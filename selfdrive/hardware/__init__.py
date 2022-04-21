import os
from typing import cast

from selfdrive.hardware.base import HardwareBase
from selfdrive.hardware.tici.hardware import Tici
from selfdrive.hardware.pc.hardware import Pc

TICI = os.path.isfile('/TICI')
PC = not TICI


if TICI:
  HARDWARE = cast(HardwareBase, Tici())
else:
  HARDWARE = cast(HardwareBase, Pc())
