import ctypes
import fcntl
import os
import time
from pathlib import Path

from openpilot.common.swaglog import cloudlog
from openpilot.common.hardware.usb import CHESTNUT_VENDOR_ID, CHESTNUT_PRODUCT_ID, usb_devices, controller, read_int

STABLE_SECONDS = 2.0
STABLE_THRESHOLD = 5.0  # link errors per second

USBGPU_RECOVER = 0xF4          # bridge vendor cmd: recover the wedged pcie tunnel
USBGPU_LTSSM = 0xB450          # pcie link state register, 0x78 = link up
LTSSM_UP = 0x78
USBGPU_XWRITE = 0xE5           # xdata write vendor cmd
CPU_RESET_REG = 0xCC31        # write 0x01 -> reboot bridge cpu + re-enumerate usb
_USBDEVFS_CONTROL = 0xC0185500  # _IOWR('U', 0, struct usbdevfs_ctrltransfer), 64-bit


def _clear_tg_usb_cache() -> None:
  # a cpu reset re-enumerates usb, so tinygrad's cached device pointer goes stale and the
  # retry fails with "No interface for AMD". drop the cache so it re-scans. lazy import so
  # non-modeld importers (ui) do not pull in tinygrad.
  try:
    from tinygrad.runtime.support.usb import USB3
    USB3.list_devices.cache_clear()
  except Exception:
    pass


class _CtrlTransfer(ctypes.Structure):
  _fields_ = [("bRequestType", ctypes.c_uint8), ("bRequest", ctypes.c_uint8),
              ("wValue", ctypes.c_uint16), ("wIndex", ctypes.c_uint16),
              ("wLength", ctypes.c_uint16), ("timeout", ctypes.c_uint32),
              ("data", ctypes.c_void_p)]


def _ltssm(fd: int) -> int:
  buf = (ctypes.c_ubyte * 1)()
  req = _CtrlTransfer(0xC0, 0xE4, USBGPU_LTSSM, 0, 1, 2000, ctypes.cast(buf, ctypes.c_void_p))
  try:
    fcntl.ioctl(fd, _USBDEVFS_CONTROL, req)
    return buf[0]
  except OSError:
    return -1


def _chestnut_device() -> Path | None:
  for device in usb_devices():
    if read_int(device / "idVendor", 16) == CHESTNUT_VENDOR_ID and \
       read_int(device / "idProduct", 16) == CHESTNUT_PRODUCT_ID:
      return device
  return None


def _chestnut_portli() -> Path | None:
  device = _chestnut_device()
  if device is not None:
    ctrl = controller(device)
    if ctrl is not None and (ctrl / "portli").exists():
      return ctrl / "portli"
  return None


def release_leaked_locks() -> None:
  # a failed gpu init leaks its device flock fd, so the next in-process retry blocks
  # with EAGAIN ("Failed to acquire lock file"). close any leaked am_usb lock fds.
  for fd in os.listdir('/proc/self/fd'):
    try:
      if 'am_usb' in os.readlink(f'/proc/self/fd/{fd}'):
        cloudlog.warning(f"releasing leaked usbgpu lock fd {fd}")
        os.close(int(fd))
    except OSError:
      pass


def recover_usbgpu(timeout: float = 15.0) -> None:
  # release the leaked device lock, then reset the bridge so the retry gets a clean device.
  # 0xF4 retrains a down pcie link; if the usb link itself is erroring (bulk/control i/o
  # errors that 0xF4 cannot clear or even be delivered through), the cpu reset reboots the
  # bridge and re-enumerates usb. the gpu psp is reset separately via AM_RESET on the retry.
  release_leaked_locks()
  device = _chestnut_device()
  if device is not None:
    node = f"/dev/bus/usb/{read_int(device / 'busnum'):03d}/{read_int(device / 'devnum'):03d}"
    try:
      fd = os.open(node, os.O_RDWR)
      try:
        lt = _ltssm(fd)
        cloudlog.warning(f"usbgpu recover: ltssm=0x{lt:02X} ({'link down' if lt != LTSSM_UP else 'psp hang'}), 0xF4 + cpu reset")
        try:
          fcntl.ioctl(fd, _USBDEVFS_CONTROL, _CtrlTransfer(0x40, USBGPU_RECOVER, 0, 0, 0, 10000, None))
        except OSError:
          pass
        try:
          fcntl.ioctl(fd, _USBDEVFS_CONTROL, _CtrlTransfer(0x40, USBGPU_XWRITE, CPU_RESET_REG, 0x01, 0, 3000, None))
        except OSError:
          pass  # the transfer drops when the reset fires, that's expected
      finally:
        os.close(fd)
    except OSError:
      pass
  _clear_tg_usb_cache()  # the cpu reset re-enumerates usb, so the cached device pointer is stale
  time.sleep(6)  # let the bridge reboot and re-enumerate before the retry
  t0 = time.monotonic()
  while time.monotonic() - t0 < timeout:
    if _chestnut_device() is not None:
      return
    time.sleep(0.5)


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
