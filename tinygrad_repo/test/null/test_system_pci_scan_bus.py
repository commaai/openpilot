import sys
import pytest

@pytest.mark.skipif(sys.platform != "linux", reason="uses linux sysfs layout")
def test_pci_scan_bus_filters_vendor(monkeypatch):
  import tinygrad.runtime.support.system as system

  fake = {
    "/sys/bus/pci/devices/0000:00:01.0/vendor": "0x1234",
    "/sys/bus/pci/devices/0000:00:01.0/device": "0x1111",
    "/sys/bus/pci/devices/0000:00:02.0/vendor": "0xabcd",
    "/sys/bus/pci/devices/0000:00:02.0/device": "0x1111",
  }

  class FakeFileIOInterface:
    def __init__(self, path, *args, **kwargs):
      self.path = path

    def listdir(self):
      assert self.path == "/sys/bus/pci/devices"
      return ["0000:00:01.0", "0000:00:02.0"]

    def read(self, *args, **kwargs):
      return fake[self.path]

  monkeypatch.setattr(system, "FileIOInterface", FakeFileIOInterface)

  assert system.System.pci_scan_bus(0x1234, devices=[(0xffff, [0x1111])]) == ["0000:00:01.0"]
