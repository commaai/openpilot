import array, time
from hexdump import hexdump
from tinygrad.runtime.support.usb import ASM24Controller
from tinygrad.runtime.autogen import pci

usb = ASM24Controller()

def print_cfg(bus, dev):
  cfg = []
  for i in range(0, 256, 4):
    cfg.append(usb.pcie_cfg_req(i, bus=bus, dev=dev, fn=0, value=None, size=4))

  print("bus={}, dev={}".format(bus, dev))
  dmp = bytearray(array.array('I', cfg))
  hexdump(dmp)
  return dmp

def rescan_bus(bus, gpu_bus):
  print("set PCI_SUBORDINATE_BUS bus={} to {}".format(bus, gpu_bus))
  usb.pcie_cfg_req(pci.PCI_SUBORDINATE_BUS, bus=bus, dev=0, fn=0, value=gpu_bus, size=1)
  usb.pcie_cfg_req(pci.PCI_SECONDARY_BUS, bus=bus, dev=0, fn=0, value=bus+1, size=1)
  usb.pcie_cfg_req(pci.PCI_PRIMARY_BUS, bus=bus, dev=0, fn=0, value=max(0, bus-1), size=1)

  print("rescan bus={}".format(bus))
  usb.pcie_cfg_req(pci.PCI_BRIDGE_CONTROL, bus=bus, dev=0, fn=0, value=pci.PCI_BRIDGE_CTL_BUS_RESET, size=1)
  time.sleep(0.1)
  usb.pcie_cfg_req(pci.PCI_BRIDGE_CONTROL, bus=bus, dev=0, fn=0, value=pci.PCI_BRIDGE_CTL_PARITY|pci.PCI_BRIDGE_CTL_SERR, size=1)

  usb.pcie_cfg_req(pci.PCI_MEMORY_BASE, bus=bus, dev=0, fn=0, value=0x1000, size=2)
  usb.pcie_cfg_req(pci.PCI_MEMORY_LIMIT, bus=bus, dev=0, fn=0, value=0x2000, size=2)
  usb.pcie_cfg_req(pci.PCI_PREF_MEMORY_BASE, bus=bus, dev=0, fn=0, value=0x2000, size=2)
  usb.pcie_cfg_req(pci.PCI_PREF_MEMORY_LIMIT, bus=bus, dev=0, fn=0, value=0xffff, size=2)

print_cfg(0, 0)
rescan_bus(0, gpu_bus=4)

print_cfg(1, 0)
rescan_bus(1, gpu_bus=4)

time.sleep(0.1)
print_cfg(2, 0)

def setup_bus(bus, gpu_bus):
  print("setup bus={}".format(bus))
  usb.pcie_cfg_req(pci.PCI_SUBORDINATE_BUS, bus=bus, dev=0, fn=0, value=gpu_bus, size=1)
  usb.pcie_cfg_req(pci.PCI_SECONDARY_BUS, bus=bus, dev=0, fn=0, value=bus+1, size=1)
  usb.pcie_cfg_req(pci.PCI_PRIMARY_BUS, bus=bus, dev=0, fn=0, value=max(0, bus-1), size=1)

  usb.pcie_cfg_req(pci.PCI_BRIDGE_CONTROL, bus=bus, dev=0, fn=0, value=pci.PCI_BRIDGE_CTL_BUS_RESET, size=1)
  usb.pcie_cfg_req(pci.PCI_BRIDGE_CONTROL, bus=bus, dev=0, fn=0, value=pci.PCI_BRIDGE_CTL_PARITY|pci.PCI_BRIDGE_CTL_SERR, size=1)
  usb.pcie_cfg_req(pci.PCI_COMMAND, bus=bus, dev=0, fn=0, value=pci.PCI_COMMAND_IO | pci.PCI_COMMAND_MEMORY | pci.PCI_COMMAND_MASTER, size=1)

  usb.pcie_cfg_req(pci.PCI_MEMORY_BASE, bus=bus, dev=0, fn=0, value=0x1000, size=2)
  usb.pcie_cfg_req(pci.PCI_MEMORY_LIMIT, bus=bus, dev=0, fn=0, value=0x2000, size=2)

  usb.pcie_cfg_req(pci.PCI_PREF_MEMORY_BASE, bus=bus, dev=0, fn=0, value=0x2000, size=2)
  usb.pcie_cfg_req(pci.PCI_PREF_MEMORY_LIMIT, bus=bus, dev=0, fn=0, value=0xffff, size=2)

setup_bus(2, gpu_bus=4)
print_cfg(3, 0)

setup_bus(3, gpu_bus=4)
dmp = print_cfg(4, 0)
print(dmp[0:4])
assert dmp[0:4] in (b"\x02\x10\x80\x74", b"\x02\x10\x4c\x74", b"\x02\x10\x50\x75"), "GPU NOT FOUND!"

print("GPU FOUND!")
