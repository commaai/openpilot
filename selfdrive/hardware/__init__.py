import os
from typing import cast

from selfdrive.hardware.base import HardwareBase
from selfdrive.hardware.eon.hardware import Android
from selfdrive.hardware.tici.hardware import Tici
from selfdrive.hardware.pc.hardware import Pc
from selfdrive.hardware.jetson.hardware import Jetson

EON = os.path.isfile('/EON')
TICI = os.path.isfile('/TICI')
JETSON = os.path.isfile('/JETSON')
PC = not (EON or TICI or JETSON)


if EON:
  HARDWARE = cast(HardwareBase, Android())
elif TICI:
  HARDWARE = cast(HardwareBase, Tici())
elif JETSON:
  HARDWARE = cast(HardwareBase, Jetson())
else:
  HARDWARE = cast(HardwareBase, Pc())
