#!/usr/bin/env python3
"""chestnut (ASM2464) SPI flasher using data-USB EP0 control transfers."""
import argparse
import ctypes
import errno
import fcntl
import glob
import hashlib
import os
import re
import signal
import struct
import sys
import time
import zlib
from pathlib import Path

VID_PIDS = (("add1", "0001"), ("3801", "0001"))
ROM_VID_PIDS = (("174c", "2464"), ("174c", "2463"))
ROM_PRODUCT = "USB 3.2 PCIe TinyEnclosure"
FIRMWARE_PATH = Path(__file__).with_name("firmware_wrapped.bin")
CONFIG_DIR = "/data/chestnut_config"
PM_PATHS = ("/sys/bus/platform/devices/a600000.ssusb", "/sys/bus/usb/devices/usb4")
VBUS_PATH = "/sys/kernel/debug/regulator/smb2-vbus/enable"
IMAGE_OFFSET = 0x100
SECTOR, PAGE = 4096, 128
MAX_CODE_SIZE = 0x10000
FLASH_BUDGET = 600.0
USBDEVFS_CONTROL = 0xC0185500
USBDEVFS_BULK = 0xC0185502
USBDEVFS_SETINTERFACE = 0x80085504
USBDEVFS_SETCONFIGURATION = 0x80045505
USBDEVFS_CLAIMINTERFACE = 0x8004550F
USBDEVFS_RESET = 0x5514
USBDEVFS_CLEAR_HALT = 0x80045515

_deadline = float("inf")


def check_budget():
  if time.monotonic() > _deadline:
    raise TimeoutError(f"flash did not converge within {FLASH_BUDGET:g}s")


class Ctrl(ctypes.Structure):
  _fields_ = [("request_type", ctypes.c_uint8), ("request", ctypes.c_uint8),
              ("value", ctypes.c_uint16), ("index", ctypes.c_uint16),
              ("length", ctypes.c_uint16), ("timeout", ctypes.c_uint32),
              ("data", ctypes.c_void_p)]


class Bulk(ctypes.Structure):
  _fields_ = [("ep", ctypes.c_uint), ("len", ctypes.c_uint),
              ("timeout", ctypes.c_uint), ("data", ctypes.c_void_p)]


class RomFallback(Exception):
  pass


def find_chestnut():
  found = []
  for d in glob.glob("/sys/bus/usb/devices/*"):
    try:
      vid_pid = (open(d + "/idVendor").read().strip(), open(d + "/idProduct").read().strip())
      if vid_pid in VID_PIDS + ROM_VID_PIDS:
        found.append((d, vid_pid, open(d + "/product").read().strip()))
    except OSError:
      pass
  if len(found) > 1:
    raise RuntimeError(f"expected one chestnut, found {len(found)}")
  return found[0] if found else (None, None, None)


def in_rom_bootloader(vid_pid, product):
  # the ROM bootloader reports the config page strings, or its own when the config page is lost
  return vid_pid in ROM_VID_PIDS or product == ROM_PRODUCT or (product or "").startswith("AS2462")


def disable_runtime_pm(path):
  control = os.path.join(path, "power/control")
  if not os.path.exists(control):
    return
  with open(control, "w") as f:
    f.write("on\n")
  if open(control).read().strip() != "on":
    raise RuntimeError(f"could not disable USB runtime PM: {control}")
  delay = os.path.join(path, "power/autosuspend_delay_ms")
  if os.path.exists(delay):
    with open(delay, "w") as f:
      f.write("-1\n")


def unbind_drivers(path):
  for interface in glob.glob(path + ":*"):
    driver = interface + "/driver"
    if os.path.islink(driver):
      with open(os.path.realpath(driver) + "/unbind", "w") as f:
        f.write(os.path.basename(interface))


def open_device(path):
  bus, dev = int(open(path + "/busnum").read()), int(open(path + "/devnum").read())
  return os.open(f"/dev/bus/usb/{bus:03d}/{dev:03d}", os.O_RDWR)


def link_up() -> bool:
  # asm enumerates on USB-C alone, gpu is only usable once pcie link is up
  try:
    path, _, _ = find_chestnut()
    if path is None:
      return False
    fd = open_device(path)
  except (OSError, RuntimeError):
    return False
  try:
    fcntl.ioctl(fd, USBDEVFS_CONTROL, Ctrl(0x40, 0xF3, 1, 0, 0, 2000, None))
    buf = (ctypes.c_ubyte * 1)()
    fcntl.ioctl(fd, USBDEVFS_CONTROL, Ctrl(0xC0, 0xE4, 0xB450, 0, 1, 1000, ctypes.cast(buf, ctypes.c_void_p)))
    return buf[0] == 0x78  # LTSSM L0
  except OSError:
    return False
  finally:
    os.close(fd)


def claim_interface(path, setup=False):
  # unbind usb-storage, which binds to the ROM bootloader
  disable_runtime_pm(path)
  unbind_drivers(path)
  fd = open_device(path)
  try:
    if setup:
      fcntl.ioctl(fd, USBDEVFS_SETCONFIGURATION, struct.pack("I", 1))
    fcntl.ioctl(fd, USBDEVFS_CLAIMINTERFACE, struct.pack("I", 0))
    if setup:
      fcntl.ioctl(fd, USBDEVFS_SETINTERFACE, struct.pack("II", 0, 0))
  except OSError as e:
    os.close(fd)
    if e.errno == errno.EBUSY:
      raise RuntimeError("chestnut is in use, stop modeld/GPU processes before flashing") from e
    raise
  return fd


class Flash:
  def __init__(self):
    self.fd = -1

  def close(self):
    if self.fd >= 0:
      os.close(self.fd)
      self.fd = -1

  def connect(self, timeout=5.0):
    self.close()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
      path, vid_pid, product = find_chestnut()
      if in_rom_bootloader(vid_pid, product):
        raise RomFallback("chestnut fell back to the ROM bootloader")
      if path is not None:
        self.fd = claim_interface(path)
        return
      time.sleep(0.1)
    raise RuntimeError(f"chestnut did not enumerate within {timeout:g}s")

  def reg_write(self, addr, value):
    fcntl.ioctl(self.fd, USBDEVFS_CONTROL,
                Ctrl(0x40, 0xE5, addr & 0xFFFF, value & 0xFFFF, 0, 2000, None))

  def reg_read(self, addr, length=1):
    buf = (ctypes.c_ubyte * length)()
    fcntl.ioctl(self.fd, USBDEVFS_CONTROL,
                Ctrl(0xC0, 0xE4, addr & 0xFFFF, 0, length, 2000, ctypes.cast(buf, ctypes.c_void_p)))
    return bytes(buf)

  def write_buffer(self, data):
    for i, value in enumerate(data):
      self.reg_write(0x7000 + i, value)

  def wait_controller(self, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
      if not self.reg_read(0xC8A9)[0] & 1:
        return
    raise TimeoutError("flash controller timeout")

  def transaction(self, command, addr=0, length=0, addr_len=0x07, mode=0):
    for reg, value in ((0xC8AD, mode), (0xC8AE, 0), (0xC8AF, 0), (0xC8AA, command), (0xC8AC, addr_len),
                       (0xC8A1, addr), (0xC8A2, addr >> 8), (0xC8AB, addr >> 16), (0xC8A3, length >> 8), (0xC8A4, length)):
      self.reg_write(reg, value & 0xFF)
    self.reg_write(0xC8A9, 1)
    self.wait_controller()
    for _ in range(4):
      self.reg_write(0xC8AD, 0)

  def write_enable(self):
    for reg, value in ((0xC8AD, 0), (0xC8AA, 0x06), (0xC8AC, 0x04), (0xC8A3, 0), (0xC8A4, 0), (0xC8A9, 1)):
      self.reg_write(reg, value)
    self.wait_controller()

  def status(self):
    self.transaction(0x05, length=1, addr_len=0x04)
    return self.reg_read(0x7000)[0]

  def wait_write_done(self, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
      if not self.status() & 1:
        return
      time.sleep(0.005)
    raise TimeoutError("SPI flash WIP timeout")

  def init(self):
    self.reg_write(0xCC33, 0x04)
    self.reg_write(0xCA81, self.reg_read(0xCA81)[0] | 1)
    self.reg_write(0xC805, 0x02)
    self.reg_write(0xC8A6, 0x04)
    for _ in range(5):
      self.write_enable()
      self.write_buffer(bytes(4))
      self.transaction(0x01, length=1, addr_len=0x04, mode=1)
      time.sleep(0.01)
      if not self.status() & 0x1C:
        return
    raise RuntimeError("could not clear SPI block protection")

  def read(self, addr, length):
    out = bytearray()
    while len(out) < length:
      n = min(4096, length - len(out))
      self.transaction(0x03, addr + len(out), max(4096, n))
      for off in range(0, n, 255):
        out += self.reg_read(0x7000 + off, min(255, n - off))
    return bytes(out)

  def erase_sector(self, addr):
    self.write_enable()
    self.transaction(0x20, addr)
    self.wait_write_done()

  def program(self, addr, data):
    self.write_buffer(data + bytes((-len(data)) % 4))
    self.write_enable()
    self.transaction(0x02, addr, len(data), mode=1)
    self.wait_write_done()


def validate_image(data):
  if len(data) < 10:
    raise ValueError("wrapped firmware is too short")
  body_len = int.from_bytes(data[:4], "little")
  if body_len > MAX_CODE_SIZE:
    raise ValueError(f"wrapped firmware body exceeds {MAX_CODE_SIZE} bytes")
  if len(data) != body_len + 10 or data[4 + body_len] != 0xA5:
    raise ValueError("invalid wrapped firmware length or magic")
  body = data[4:4 + body_len]
  if data[5 + body_len] != sum(body) & 0xFF:
    raise ValueError("invalid wrapped firmware checksum")
  if data[6 + body_len:] != zlib.crc32(body).to_bytes(4, "little"):
    raise ValueError("invalid wrapped firmware CRC")


def image_product(image):
  match = re.search(rb"custom [0-9a-f]{8}-CLEAN", image)
  if match is None:
    raise ValueError("no product string in wrapped firmware")
  return match.group().decode()


def reconnect(flash):
  attempt = 0
  while True:
    attempt += 1
    check_budget()
    try:
      flash.connect()
      flash.init()
      return
    except (OSError, TimeoutError, RuntimeError) as e:
      print(f"waiting for chestnut (attempt {attempt}): {e}", flush=True)
      time.sleep(1)


def with_retries(flash, label, operation):
  # on any transfer error, reconnect and restart the operation
  attempt = 0
  while True:
    attempt += 1
    try:
      return operation()
    except (OSError, TimeoutError, RuntimeError) as e:
      check_budget()
      print(f"{label} attempt {attempt}: {e}", flush=True)
      reconnect(flash)


def stable_read(flash, addr, length, count=2):
  def read():
    reads = [flash.read(addr, length) for _ in range(count)]
    if any(x != reads[0] for x in reads[1:]):
      raise RuntimeError(f"unstable flash read at 0x{addr:05x}")
    return reads[0]
  return with_retries(flash, f"read 0x{addr:05x}", read)


def program_sector(flash, addr, target):
  def program():
    flash.erase_sector(addr)
    if flash.read(addr, SECTOR) != bytes([0xFF]) * SECTOR:
      raise RuntimeError("sector erase verification failed")
    for off in range(0, SECTOR, PAGE):
      chunk = target[off:off + PAGE]
      if chunk != bytes([0xFF]) * len(chunk):
        flash.program(addr + off, chunk)
        if flash.read(addr + off, len(chunk)) != chunk:
          raise RuntimeError(f"page verify failed at 0x{addr + off:05x}")
    if flash.read(addr, SECTOR) != target:
      raise RuntimeError("sector verification failed")
  with_retries(flash, f"sector 0x{addr:05x}", program)


def config_path():
  return os.path.join(CONFIG_DIR, f"{os.uname().nodename}.bin")


def saved_config(path, data):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  try:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
  except FileExistsError as e:
    backup = open(path, "rb").read()
    if len(backup) != 0x100:
      raise RuntimeError(f"invalid config backup: {path}") from e
    if backup != data:
      print(f"restoring config from {path}", flush=True)
    return backup
  with os.fdopen(fd, "wb") as f:
    f.write(data)
    f.flush()
    os.fsync(f.fileno())
  return data


def rom_write(image, config):
  # the ROM bootloader implements only the BOT protocol, and requires a port reset before bulk transfers
  path, _, _ = find_chestnut()
  if path is None:
    raise RuntimeError("chestnut disappeared before recovery")
  unbind_drivers(path)
  fd = open_device(path)
  try:
    fcntl.ioctl(fd, USBDEVFS_RESET)
  finally:
    os.close(fd)
  time.sleep(3)
  path, _, _ = find_chestnut()
  if path is None:
    raise RuntimeError("chestnut did not re-enumerate after reset")
  fd = claim_interface(path, setup=True)
  for ep in (0x02, 0x81):
    fcntl.ioctl(fd, USBDEVFS_CLEAR_HALT, struct.pack("I", ep))
  tag = 0

  def bulk(ep, payload, timeout):
    buf = ctypes.create_string_buffer(bytes(payload), len(payload))
    fcntl.ioctl(fd, USBDEVFS_BULK, Bulk(ep, len(payload), timeout, ctypes.cast(buf, ctypes.c_void_p)))
    return buf.raw

  def cmd(cdb, data=b"", timeout=30000):
    nonlocal tag
    tag += 1
    bulk(0x02, struct.pack("<IIIBBB16s", 0x43425355, tag, len(data), 0, 0, len(cdb), cdb), timeout)
    if data:
      bulk(0x02, data, timeout)
    try:
      csw = bulk(0x81, bytes(13), timeout)
    except OSError as e:
      if e.errno != errno.EPIPE:
        raise
      fcntl.ioctl(fd, USBDEVFS_CLEAR_HALT, struct.pack("I", 0x81))
      csw = bulk(0x81, bytes(13), timeout)
    if csw[:4] != b"USBS" or csw[12] != 0:
      raise RuntimeError(f"ROM flash command {cdb[0]:02x} {cdb[1]:02x} failed")

  print("recovering from the ROM bootloader", flush=True)
  try:
    cmd(struct.pack(">BBB12x", 0xE1, 0x50, 0), config[:0x80])
    cmd(struct.pack(">BBB12x", 0xE1, 0x50, 1), config[0x80:])
    cmd(struct.pack(">BBI", 0xE3, 0x50, min(len(image), 0xFF00)), image[:0xFF00])
    if len(image) > 0xFF00:
      cmd(struct.pack(">BBI", 0xE3, 0xD0, len(image) - 0xFF00), image[0xFF00:])
    cmd(struct.pack(">BB13x", 0xE8, 0x51))
  finally:
    os.close(fd)
  print("recovery flash done", flush=True)


def vbus_write(value):
  try:
    with open(VBUS_PATH, "w") as f:
      f.write(value + "\n")
  except OSError:
    pass


def vbus_cycle():
  if os.path.exists(VBUS_PATH):
    vbus_write("0")
    time.sleep(2)
    vbus_write("1")
    time.sleep(5)


def activate(expected_product):
  if not os.path.exists(VBUS_PATH):
    print("no VBUS control, firmware activates on the next chestnut power cycle", flush=True)
    return
  print("power-cycling chestnut VBUS", flush=True)
  vbus_write("0")
  disconnected = False
  deadline = time.monotonic() + 5.0
  while time.monotonic() < deadline:
    path, _, _ = find_chestnut()
    if path is None:
      disconnected = True
      break
    time.sleep(0.2)
  time.sleep(1)
  vbus_write("1")
  if not disconnected:
    print("chestnut stayed powered, firmware activates on its next power cycle", flush=True)
    return
  deadline = time.monotonic() + 15.0
  while time.monotonic() < deadline:
    _, _, product = find_chestnut()
    if product is not None:
      if product == expected_product:
        print(f"activated {expected_product}", flush=True)
      else:
        print(f"chestnut re-enumerated with {product!r}, firmware activates on its next power cycle", flush=True)
      return
    time.sleep(0.2)
  print("chestnut did not re-enumerate, firmware activates on its next power cycle", flush=True)


def defer_signal(signum, _frame):
  # writing from a handler must not reenter a print already in progress
  os.write(1, f"signal {signum} deferred until the chestnut is powered back up\n".encode())


def flash_chestnut(expected_version=None, force=False):
  global _deadline

  image = FIRMWARE_PATH.read_bytes()
  validate_image(image)
  expected_product = image_product(image)
  if expected_version is not None and expected_product != f"custom {expected_version}-CLEAN":
    raise RuntimeError(f"bundled firmware is {expected_product!r}, expected version {expected_version}")

  path, vid_pid, product = find_chestnut()
  if path is None:
    print("no chestnut connected", flush=True)
    return
  if product == expected_product and not force:
    print(f"chestnut firmware is up to date ({expected_product})", flush=True)
    return

  _deadline = time.monotonic() + FLASH_BUDGET
  for pm_path in PM_PATHS:
    disable_runtime_pm(pm_path)

  previous = {sig: signal.signal(sig, defer_signal) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
  try:
    if in_rom_bootloader(vid_pid, product):
      if not recover_from_rom(image, expected_product):
        return
      # firmware is back, verify it against the bundled image
      force, product = True, None
    write_image(image, expected_product, product, force)
  finally:
    for sig, handler in previous.items():
      signal.signal(sig, handler)


def recover_from_rom(image, expected_product):
  # returns whether the chestnut came back on custom firmware
  backup = config_path()
  if not os.path.isfile(backup):
    raise RuntimeError(f"cannot recover from the ROM bootloader without a config backup at {backup}")
  config = open(backup, "rb").read()
  if len(config) != 0x100:
    raise RuntimeError(f"invalid config backup: {backup}")

  committed = False
  while True:
    check_budget()
    path, vid_pid, product = find_chestnut()
    if path is None:
      if committed:
        print("chestnut is offline, recovered firmware boots on its next power cycle", flush=True)
        return False
      vbus_cycle()
      continue
    if not in_rom_bootloader(vid_pid, product):
      return True
    if committed:
      print("chestnut stayed powered, recovered firmware boots on its next power cycle", flush=True)
      return False
    try:
      rom_write(image, config)
      committed = True
    except (OSError, TimeoutError, RuntimeError) as e:
      print(f"ROM recovery failed, retrying: {e}", flush=True)
      vbus_cycle()
      continue
    activate(expected_product)


def write_image(image, expected_product, product, force):
  if force:
    print(f"forced reflash of {expected_product}", flush=True)
  else:
    print(f"chestnut firmware mismatch: {product!r}; expected {expected_product!r}", flush=True)

  flash = Flash()
  try:
    reconnect(flash)
    config = stable_read(flash, 0, 0x100, 3)
    config = saved_config(config_path(), config)
    image_end = IMAGE_OFFSET + len(image)
    first_sector = IMAGE_OFFSET & ~(SECTOR - 1)
    span = (image_end + SECTOR - 1) & ~(SECTOR - 1)
    current = stable_read(flash, first_sector, span - first_sector)
    target = bytearray(current)
    target[:len(config)] = config
    target[IMAGE_OFFSET - first_sector:image_end - first_sector] = image
    target = bytes(target)
    print(f"target {len(image)} bytes at 0x{IMAGE_OFFSET:05x}, sha256={hashlib.sha256(image).hexdigest()}", flush=True)

    for addr in range(first_sector, span, SECTOR):
      off = addr - first_sector
      wanted = target[off:off + SECTOR]
      if current[off:off + SECTOR] == wanted:
        print(f"sector 0x{addr:05x}: unchanged", flush=True)
      else:
        print(f"sector 0x{addr:05x}: programming", flush=True)
        program_sector(flash, addr, wanted)

    verified = stable_read(flash, first_sector, span - first_sector, 3)
    if verified != target:
      raise RuntimeError("final full-image verification failed")
    print(f"verified sha256={hashlib.sha256(verified).hexdigest()}", flush=True)
  finally:
    flash.close()

  activate(expected_product)


def main():
  parser = argparse.ArgumentParser(description="check and flash the bundled chestnut firmware")
  parser.add_argument("version", nargs="?", help="expected firmware version hash")
  parser.add_argument("--force", action="store_true", help="reflash even when the version matches")
  args = parser.parse_args()
  if os.geteuid() != 0:
    raise RuntimeError("flash.py must run as root")
  flash_chestnut(expected_version=args.version, force=args.force)


if __name__ == "__main__":
  try:
    main()
  except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)
