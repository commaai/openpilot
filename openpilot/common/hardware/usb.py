import os
from pathlib import Path

from openpilot.common.kernel import upstream_kernel

CHESTNUT_FW_VERSION = "8516b768"
CHESTNUT_USB_IDS = ((0xADD1, 0x0001), (0x3801, 0x0001))
CHESTNUT_ROM_USB_IDS = ((0x174C, 0x2464), (0x174C, 0x2463))
CHESTNUT_USB_PRODUCT = f"custom {CHESTNUT_FW_VERSION}-CLEAN"
USB_DEVICES_PATH = Path("/sys/bus/usb/devices")
TYPEC_CC_ORIENTATION_PATH = Path("/sys/class/power_supply/usb/typec_cc_orientation")
TYPEC_PORT_ORIENTATION_PATH = Path("/sys/class/typec/port0/orientation")  # upstream kernel
UDC_PATH = Path("/sys/class/udc")
# the DWC3 cores, named *.ssusb by the downstream kernel
USB_CONTROLLERS = ("a600000.ssusb", "a800000.ssusb", "a600000.usb", "a800000.usb")
PRIMARY_USB_CONTROLLERS = ("a600000.ssusb", "a600000.usb")


def is_chestnut_usb_id(vendor_id: int, product_id: int, include_bootloader: bool = False) -> bool:
  ids = CHESTNUT_USB_IDS + CHESTNUT_ROM_USB_IDS if include_bootloader else CHESTNUT_USB_IDS
  return (vendor_id, product_id) in ids


def get_usb_topology() -> set[str]:
  try:
    return set(os.listdir(USB_DEVICES_PATH))
  except OSError:
    return set()


def read(path: Path) -> str | None:
  try:
    return path.read_text().strip()
  except OSError:
    return None


def read_int(path: Path, base: int = 10) -> int:
  try:
    return int(path.read_text(), base)
  except (OSError, ValueError, TypeError):
    return 0


def typec_orientation() -> int:
  """0 when nothing is connected, 1 or 2 for the CC line the cable is on"""
  if upstream_kernel():
    return {"normal": 1, "reverse": 2}.get(read(TYPEC_PORT_ORIENTATION_PATH) or "", 0)
  return read_int(TYPEC_CC_ORIENTATION_PATH)


def cable_connected() -> bool:
  return typec_orientation() != 0


def usb_host_connected() -> bool:
  """Whether a USB host, e.g. a PC with adb or the UP in CI, is on the USB-C port, also while it has suspended this device.
  A chestnut is a USB device, so then it isn't on that port."""
  try:
    return any(read(p) not in (None, "not attached") for p in UDC_PATH.glob("*/state"))
  except OSError:
    return False


def usb_devices() -> list[Path]:
  try:
    devices = (d for d in USB_DEVICES_PATH.glob("*") if (d / "idVendor").exists())
    return sorted(devices, key=lambda p: p.name)
  except OSError:
    return []


def controller(device: Path) -> Path | None:
  try:
    return next((parent for parent in device.resolve().parents if parent.name in USB_CONTROLLERS), None)
  except OSError:
    return None


def get_usb_state() -> list[dict]:
  devices = []
  orientation = typec_orientation()
  for device in usb_devices():
    vendor_id = read_int(device / "idVendor", 16)
    product_id = read_int(device / "idProduct", 16)
    ctrl = controller(device)
    devices.append({
      "busnum": read_int(device / "busnum"),
      "devnum": read_int(device / "devnum"),
      "vendorId": vendor_id,
      "productId": product_id,
      "speedMbps": read_int(device / "speed"),
      "manufacturer": read(device / "manufacturer") or "",
      "product": read(device / "product") or "",
      "linkErrorCount": read_int(ctrl / "portli", 0) & 0xFFFF if ctrl is not None else 0,
      "usb3Lane": {1: "a", 2: "b"}.get(orientation, "unknown") if ctrl is not None and ctrl.name in PRIMARY_USB_CONTROLLERS else "unknown",
    })
  return devices


def set_usb_state(device_state, devices: list[dict]) -> None:
  entries = device_state.usbState.init('devices', len(devices))

  chestnut_present = False
  for entry, device in zip(entries, devices, strict=True):
    entry.busnum = device["busnum"]
    entry.devnum = device["devnum"]
    entry.vendorId = device["vendorId"]
    entry.productId = device["productId"]
    entry.speedMbps = device["speedMbps"]
    entry.manufacturer = device["manufacturer"]
    entry.product = device["product"]
    entry.linkErrorCount = device["linkErrorCount"]
    entry.usb3Lane = device.get("usb3Lane", "unknown")

    if is_chestnut_usb_id(entry.vendorId, entry.productId):
      chestnut_present = True

  device_state.chestnutPresent = chestnut_present
  device_state.usbState.connected = cable_connected()
