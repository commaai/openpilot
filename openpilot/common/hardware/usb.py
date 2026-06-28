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
  s = read(path)
  try:
    if s is None:
      return 0
    return int(s, base)
  except ValueError:
    try:
      return int(float(s)) if base == 10 else 0
    except ValueError:
      return 0


def usb_devices() -> list[Path]:
  devices = (d for d in USB_DEVICES_PATH.glob("*") if (d / "idVendor").exists())
  return sorted(devices, key=lambda p: p.name)


def controller(device: Path) -> Path | None:
  try:
    for parent in device.resolve().parents:
      if parent.name.endswith(".ssusb"):
        return parent
  except OSError:
    pass
  return None


def update_usb_state(device_state) -> None:
  devices = usb_devices()
  entries = device_state.usbState.init('devices', len(devices))

  chestnut_present = False
  for entry, device in zip(entries, devices, strict=True):
    vendor_id = read_int(device / "idVendor", 16)
    product_id = read_int(device / "idProduct", 16)

    entry.busnum = read_int(device / "busnum")
    entry.devnum = read_int(device / "devnum")
    entry.vendorId = vendor_id
    entry.productId = product_id
    entry.speedMbps = read_int(device / "speed")
    entry.product = read(device / "product") or ""

    ctrl = controller(device)
    if ctrl is not None:
      entry.linkErrorCount = read_int(ctrl / "portli", 0) & 0xFFFF

    if (vendor_id, product_id) == (CHESTNUT_VENDOR_ID, CHESTNUT_PRODUCT_ID):
      chestnut_present = True

  device_state.chestnutPresent = chestnut_present
