from pathlib import Path

from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.modeld.helpers import USBGPU_VID, USBGPU_PID


def read(path: Path) -> str | None:
  try:
    return path.read_text().strip()
  except OSError:
    return None


def read_int(path: Path, base: int = 10) -> int:
  s = read(path)
  try:
    return int(s, base) if s is not None else 0
  except ValueError:
    return 0


def usb_devices() -> list[Path]:
  # enumerated USB devices
  devices = (d for d in Path("/sys/bus/usb/devices").glob("*") if (d / "idVendor").exists())
  return sorted(devices, key=lambda p: p.name)


def root_hub_port(device: Path) -> Path:
  bus, _, port = device.name.partition("-")
  return Path(f"/sys/bus/usb/devices/usb{bus}/{bus}-0:1.0/usb{bus}-port{port}")


def controller(device: Path) -> Path | None:
  # get SS port registers
  for parent in device.resolve().parents:
    if parent.name.endswith(".ssusb"):
      return parent
  return None


class UsbLogger:
  def __init__(self):
    self.prev: set[tuple[int, int]] = set()

  def update(self, device_state) -> None:
    devices = usb_devices()

    # low level state
    state = device_state.usbState
    state.vbusMv = read_int(Path("/sys/class/power_supply/usb/voltage_now")) // 1000
    entries = state.init('devices', len(devices))

    present: dict[tuple[int, int], Path] = {}
    chestnut_present = False
    for entry, device in zip(entries, devices, strict=True):
      vendor_id = read_int(device / "idVendor", 16)
      product_id = read_int(device / "idProduct", 16)
      busnum = read_int(device / "busnum")
      devnum = read_int(device / "devnum")
      present[(busnum, devnum)] = device

      entry.busnum = busnum
      entry.devnum = devnum
      entry.vendorId = vendor_id
      entry.productId = product_id
      entry.speedMbps = read_int(device / "speed")
      entry.product = read(device / "product") or ""
      entry.pmActive = read(device / "power/runtime_status") == "active"
      entry.runtimeSuspendedMs = read_int(device / "power/runtime_suspended_time")
      entry.overCurrentCount = read_int(root_hub_port(device) / "over_current_count")

      ctrl = controller(device)
      if ctrl is not None:
        entry.linkErrorCount = read_int(ctrl / "portli", 0) & 0xFFFF  # decode PORTLI[15:0]

      if (vendor_id, product_id) == (USBGPU_VID, USBGPU_PID):
        chestnut_present = True

    # parse peripherals
    device_state.chestnutPresent = chestnut_present

    # connect/disconnect events
    for key in present.keys() - self.prev:
      device = present[key]
      cloudlog.event("usb_connected", busnum=key[0], devnum=key[1],
                     vid=read(device / "idVendor"), pid=read(device / "idProduct"),
                     speed=read(device / "speed"), product=read(device / "product"))
    for key in self.prev - present.keys():
      cloudlog.event("usb_disconnected", busnum=key[0], devnum=key[1])
    self.prev = set(present)
