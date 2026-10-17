from __future__ import annotations
import struct, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tinygrad.runtime.support.usb import USB3
from tinygrad.runtime.support.system import PCIDevice, System, USBPCIDevice

USB_IDS = ((0x3801, 0x0001), (0xADD1, 0x0001))
NAVI31_DEVICES = ((0xffff, (0x744c,)),)


def open_gpu(index: int = 0, transport: str = 'auto') -> PCIDevice:
  """Open an AMD GPU through tinygrad's transport-independent PCI interface."""
  if transport not in ('auto', 'usb', 'pci'): raise ValueError(f"unsupported transport {transport!r}")
  candidates = []
  if transport in ('auto', 'usb'):
    for vendor, product in USB_IDS:
      candidates += [(USBPCIDevice, dev) for dev in USB3.list_devices(vendor, product)]
  if transport in ('auto', 'pci'):
    candidates += System.list_devices(0x1002, NAVI31_DEVICES)
  if not candidates: raise RuntimeError(f"no supported {transport} AMD GPU found")
  if not 0 <= index < len(candidates): raise RuntimeError(f"device index {index} out of range (found {len(candidates)})")
  cls, descriptor = candidates[index]
  return cls("AM", *descriptor) if cls is USBPCIDevice else cls("AM", descriptor)


class MMIO:
  """Transport-independent byte view of BAR5."""
  def __init__(self, pci_dev: PCIDevice): self.bar = pci_dev.map_bar(5, fmt='B')

  def read32(self, offset: int) -> int:
    return struct.unpack('<I', bytes(self.bar[offset:offset+4]))[0]

  def write32(self, offset: int, value: int):
    self.write(offset, struct.pack('<I', value & 0xffffffff))

  def read(self, offset: int, size: int) -> bytes:
    return bytes(self.bar[offset:offset+size])

  def write(self, offset: int, data: bytes):
    self.bar[offset:offset+len(data)] = data


def wait_until(fn, timeout: float, message: str, interval: float = 0.001):
  if timeout <= 0 or timeout > 60: raise ValueError("timeout must be in (0, 60] seconds")
  end = time.monotonic() + timeout
  while True:
    value = fn()
    if value: return value
    if time.monotonic() >= end: raise TimeoutError(message)
    time.sleep(interval)
