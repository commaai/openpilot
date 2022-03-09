import os
from typing import cast

from system.hardware.base import HardwareBase
from system.hardware.eon.hardware import Android
from system.hardware.tici.hardware import Tici
from system.hardware.pc.hardware import Pc

EON = os.path.isfile('/EON')
TICI = os.path.isfile('/TICI')
PC = not (EON or TICI)


if EON:
  HARDWARE = cast(HardwareBase, Android())
elif TICI:
  HARDWARE = cast(HardwareBase, Tici())
else:
  HARDWARE = cast(HardwareBase, Pc())
