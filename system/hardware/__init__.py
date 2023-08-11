import os
from typing import cast

from system.hardware.base import HardwareBase
from system.hardware.pc.hardware import Pc
from system.hardware.tici.hardware import Tici

TICI = os.path.isfile('/TICI')
AGNOS = os.path.isfile('/AGNOS')
PC = not TICI


if TICI:
  HARDWARE = cast(HardwareBase, Tici())
else:
  HARDWARE = cast(HardwareBase, Pc())
