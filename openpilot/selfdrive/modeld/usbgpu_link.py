import time
from pathlib import Path

from openpilot.common.swaglog import cloudlog
from openpilot.common.hardware.usb import CHESTNUT_VENDOR_ID, CHESTNUT_PRODUCT_ID, usb_devices, controller, read_int

STABLE_SECONDS = 2.0
STABLE_THRESHOLD = 5.0  # link errors per second


def _chestnut_portli() -> Path | None:
  for device in usb_devices():
    if read_int(device / "idVendor", 16) == CHESTNUT_VENDOR_ID and \
       read_int(device / "idProduct", 16) == CHESTNUT_PRODUCT_ID:
      ctrl = controller(device)
      if ctrl is not None and (ctrl / "portli").exists():
        return ctrl / "portli"
  return None


def wait_usbgpu_link(timeout: float = 30.0) -> None:
  portli = _chestnut_portli()
  if portli is None:
    return

  t0 = time.monotonic()
  while time.monotonic() - t0 < timeout:
    start = read_int(portli, 0)
    time.sleep(STABLE_SECONDS)
    rate = (read_int(portli, 0) - start) / STABLE_SECONDS
    if rate <= STABLE_THRESHOLD:
      return
    cloudlog.warning(f"usbgpu link not stable: {rate:.0f} errors/s")
  cloudlog.error("usbgpu link never stabilized")
