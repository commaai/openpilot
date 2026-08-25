import time

from openpilot.common.hardware.usb import is_chestnut_usb_device, is_current_chestnut_firmware
from openpilot.common.version import CHESTNUT_RELEASE_BRANCHES
from openpilot.selfdrive.modeld.helpers import usbgpu_compiled
from openpilot.system.hardware.chestnut.flash import get_pcie_state


class ChestnutStatus:
  def __init__(self):
    self._started = time.monotonic()
    self._last_link_check = 0.
    self.pcie_state: int | None = None

  def update(self, offroad: bool, branch: str, usb_state: list[dict], firmware_failed: bool, set_alert):
    devices = [d for d in usb_state if is_chestnut_usb_device(d["vendorId"], d["productId"])]
    all_devices = [d for d in usb_state if is_chestnut_usb_device(d["vendorId"], d["productId"], include_bootloader=True)]
    firmware_ok = any(is_current_chestnut_firmware(d["product"]) for d in devices)

    if not offroad or not firmware_ok:
      self.pcie_state = None
    elif time.monotonic() - self._last_link_check >= 5.:
      self._last_link_check = time.monotonic()
      self.pcie_state = get_pcie_state()

    release = branch in CHESTNUT_RELEASE_BRANCHES
    set_alert("Offroad_ChestnutFirmwareUpdateFailed", firmware_failed)
    set_alert("Offroad_ChestnutNotDetected", offroad and release and time.monotonic() - self._started > 10. and
              len(all_devices) != 1)
    slow_usb = offroad and release and len(devices) == 1 and devices[0]["speedMbps"] < 5000
    set_alert("Offroad_ChestnutUsbLinkSlow", slow_usb, f"{devices[0]['speedMbps']} Mbps" if slow_usb else None)
    set_alert("Offroad_ChestnutPcieLinkUnavailable", offroad and firmware_ok and self.pcie_state != 0x78)
    set_alert("Offroad_ChestnutModelNotCompiled", offroad and release and firmware_ok and self.pcie_state == 0x78 and
              not usbgpu_compiled())
