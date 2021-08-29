import os
from typing import cast

from selfdrive.hardware.base import HardwareBase
from selfdrive.hardware.eon.hardware import Android
from selfdrive.hardware.tici.hardware import Tici
from selfdrive.hardware.pc.hardware import Pc

EON = os.path.isfile('/EON')
TICI = os.path.isfile('/TICI')
PC = not (EON or TICI)


if EON:
  HARDWARE = cast(HardwareBase, Android())
elif TICI:
  HARDWARE = cast(HardwareBase, Tici())
else:
  HARDWARE = cast(HardwareBase, Pc())
