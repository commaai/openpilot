#!/usr/bin/env python3
from panda import Panda, PandaDFU, STBootloaderSPIHandle

if __name__ == "__main__":
  try:
    from openpilot.system.hardware import HARDWARE
    HARDWARE.recover_internal_panda()
    Panda.wait_for_dfu(None, 5)
  except Exception:
    pass

  p = PandaDFU(None)
  assert isinstance(p._handle, STBootloaderSPIHandle)
  cfg = p.get_mcu_type().config

  print("restoring from backup...")
  addr = cfg.bootstub_address
  for i, sector_size in enumerate(cfg.sector_sizes):
    print(f"- sector #{i}")
    p._handle.erase_sector(i)
    with open(f"sector_{i}.bin", "rb") as f:
      dat = f.read()
      assert len(dat) == sector_size
      p._handle.program(addr, dat)
    addr += len(dat)

  p.reset()
