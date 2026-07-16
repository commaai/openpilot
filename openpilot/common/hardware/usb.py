from pathlib import Path

CHESTNUT_VENDOR_ID = 0xADD1
CHESTNUT_PRODUCT_ID = 0x0001
USB_DEVICES_PATH = Path("/sys/bus/usb/devices")


def read(path: Path) -> str | None:
  try:
    return path.read_text().strip()
  except OSError:
    return None


def read_int(path: Path, base: int = 10) -> int:
  try:
    return int(path.read_text(), base)
  except (OSError, ValueError):
    return 0


def usb_devices() -> list[Path]:
  try:
    devices = (d for d in USB_DEVICES_PATH.glob("*") if (d / "idVendor").exists())
    return sorted(devices, key=lambda p: p.name)
  except OSError:
    return []


def controller(device: Path) -> Path | None:
  try:
    return next((parent for parent in device.resolve().parents if parent.name.endswith(".ssusb")), None)
  except OSError:
    return None


def get_usb_state() -> list[dict]:
  devices = []
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

    if (entry.vendorId, entry.productId) == (CHESTNUT_VENDOR_ID, CHESTNUT_PRODUCT_ID):
      chestnut_present = True

  device_state.chestnutPresent = chestnut_present
