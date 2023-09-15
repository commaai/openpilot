import os
import enum
from typing import List, NamedTuple

BASEDIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")
FW_PATH = os.path.join(BASEDIR, "board/obj/")

USBPACKET_MAX_SIZE = 0x40

class McuConfig(NamedTuple):
  mcu: str
  mcu_idcode: int
  sector_sizes: List[int]
  uid_address: int
  block_size: int
  serial_number_address: int
  app_address: int
  app_fn: str
  bootstub_address: int
  bootstub_fn: str

  def sector_address(self, i):
    # assume bootstub is in sector 0
    return self.bootstub_address + sum(self.sector_sizes[:i])

Fx = (
  0x1FFF7A10,
  0x800,
  0x1FFF79C0,
  0x8004000,
  "panda.bin.signed",
  0x8000000,
  "bootstub.panda.bin",
)
F2Config = McuConfig("STM32F2", 0x411, [0x4000 for _ in range(4)] + [0x10000] + [0x20000 for _ in range(7)], *Fx)
F4Config = McuConfig("STM32F4", 0x463, [0x4000 for _ in range(4)] + [0x10000] + [0x20000 for _ in range(11)], *Fx)

H7Config = McuConfig(
  "STM32H7",
  0x483,
  [0x20000 for _ in range(7)],
  0x1FF1E800,
  0x400,
  # there is an 8th sector, but we use that for the provisioning chunk, so don't program over that!
  0x080FFFC0,
  0x8020000,
  "panda_h7.bin.signed",
  0x8000000,
  "bootstub.panda_h7.bin",
)

@enum.unique
class McuType(enum.Enum):
  F2 = F2Config
  F4 = F4Config
  H7 = H7Config

  @property
  def config(self):
    return self.value

MCU_TYPE_BY_IDCODE = {m.config.mcu_idcode: m for m in McuType}
