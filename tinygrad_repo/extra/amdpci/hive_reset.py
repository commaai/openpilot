#!/usr/bin/env python3

from tinygrad.helpers import Context
from tinygrad.runtime.support.system import System, PCIDevice, PCIDevImplBase
from tinygrad.runtime.support.am.amdev import AMDev

if __name__ == "__main__":
  gpus = System.pci_scan_bus(0x1002, [(0xffff, [0x74a1, 0x75a0])])
  pcidevs = [PCIDevice(f"reset:{gpu}", gpu, bars=[0, 2, 5]) for gpu in gpus]
  amdevs = []
  with Context(DEBUG=2):
    for pcidev in pcidevs:
      amdevs.append(AMDev(pcidev, reset_mode=True))
    for amdev in amdevs: amdev.smu.mode1_reset()
