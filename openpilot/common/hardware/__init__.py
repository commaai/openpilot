import os
from typing import cast

from openpilot.common.hardware.base import HardwareBase
from openpilot.common.hardware.comma.hardware import Comma
from openpilot.common.hardware.pc.hardware import Pc

AGNOS = os.path.isfile('/AGNOS')
COMMA = AGNOS
PC = not COMMA


if COMMA:
  HARDWARE = cast(HardwareBase, Comma())
else:
  HARDWARE = cast(HardwareBase, Pc())
