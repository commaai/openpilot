#!/usr/bin/env python3
from panda import Panda, PandaDFU

if __name__ == "__main__":
  try:
    from openpilot.system.hardware import HARDWARE
    HARDWARE.recover_internal_panda()
    Panda.wait_for_dfu(None, 5)
  except Exception:
    pass

  p = PandaDFU(None)
  cfg = p.get_mcu_type().config

  def readmem(addr, length, fn):
    print(f"reading {hex(addr)} {hex(length)} bytes to {fn}")
    max_size = 255
    with open(fn, "wb") as f:
      to_read = length
      while to_read > 0:
        l = min(to_read, max_size)
        dat = p._handle.read(addr, l)
        assert len(dat) == l
        f.write(dat)

        to_read -= len(dat)
        addr += len(dat)

  addr = cfg.bootstub_address
  for i, sector_size in enumerate(cfg.sector_sizes):
    readmem(addr, sector_size, f"sector_{i}.bin")
    addr += sector_size
