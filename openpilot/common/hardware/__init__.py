import os
from typing import cast

from openpilot.common.hardware.base import HardwareBase
from openpilot.common.hardware.comma.hardware import HardwareComma
from openpilot.common.hardware.pc.hardware import HardwarePc

AGNOS = os.path.isfile('/AGNOS')
COMMA_HARDWARE = AGNOS
PC = not COMMA_HARDWARE


if COMMA_HARDWARE:
  HARDWARE = cast(HardwareBase, HardwareComma())
else:
  HARDWARE = cast(HardwareBase, HardwarePc())
