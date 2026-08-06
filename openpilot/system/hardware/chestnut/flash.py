#!/usr/bin/env python3
"""chestnut (ASM2464) SPI flasher using only data-USB EP0 control transfers."""
import argparse
import ctypes
import errno
import fcntl
import glob
import hashlib
import os
from pathlib import Path
import signal
import struct
import sys
import time
import zlib

VID_PIDS = (("add1", "0001"), ("3801", "0001"))
STOCK_VID_PIDS = (("174c", "2464"), ("174c", "2463"))
STOCK_PRODUCT = "USB 3.2 PCIe TinyEnclosure"
EXPECTED_VERSION = "bef953a4"
EXPECTED_PRODUCT = f"custom {EXPECTED_VERSION}-CLEAN"
FIRMWARE_PATH = Path(__file__).with_name("firmware_wrapped.bin")
FIRMWARE_SHA256 = "88a4c169234cd858ca70d268a2bb7bab68cba87c07f88685d11c3bfcb49c43d0"
BACKUP_DIR = "/data/asm_flash_backups"
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


def unbind_drivers(path):
  for interface in glob.glob(path + ":*"):
    driver = interface + "/driver"
    if os.path.islink(driver):
      with open(os.path.realpath(driver) + "/unbind", "w") as f:
        f.write(os.path.basename(interface))


def claim_device(path, setup=False):
  # unbind kernel drivers (usb-storage in stock mode) and claim interface 0
  pin_runtime_power(path)
  unbind_drivers(path)
  bus, dev = int(open(path + "/busnum").read()), int(open(path + "/devnum").read())
  fd = os.open(f"/dev/bus/usb/{bus:03d}/{dev:03d}", os.O_RDWR)
  try:
    if setup:
      fcntl.ioctl(fd, USBDEVFS_SETCONFIGURATION, struct.pack("I", 1))
    fcntl.ioctl(fd, USBDEVFS_CLAIMINTERFACE, struct.pack("I", 0))
    if setup:
      fcntl.ioctl(fd, USBDEVFS_SETINTERFACE, struct.pack("II", 0, 0))
  except OSError as e:
    os.close(fd)
    if e.errno == errno.EBUSY:
      raise RuntimeError("Chestnut is in use, stop modeld/GPU processes before flashing") from e
    raise
  return fd


def stock_recover(serial, image, config):
  # interrupted flashing falls back to the ROM bootloader, which only speaks the BOT vendor protocol
  # and needs a fresh link train (port reset) before it accepts bulk transfers on this xHCI
  path, _, _ = installed_chestnut()
  unbind_drivers(path)
  bus, dev = int(open(path + "/busnum").read()), int(open(path + "/devnum").read())
  fd = os.open(f"/dev/bus/usb/{bus:03d}/{dev:03d}", os.O_RDWR)
  try:
    fcntl.ioctl(fd, USBDEVFS_RESET)
  finally:
    os.close(fd)
  time.sleep(3)
  path, _, _ = installed_chestnut()
  if path is None:
    raise RuntimeError("Chestnut did not re-enumerate after reset")
  fd = claim_device(path, setup=True)
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
      raise RuntimeError(f"stock flash command {cdb[0]:02x} {cdb[1]:02x} failed")

  print(f"[{serial}] recovering from stock bootloader mode", flush=True)
  try:
    cmd(struct.pack(">BBB12x", 0xE1, 0x50, 0), config[:0x80])
    cmd(struct.pack(">BBB12x", 0xE1, 0x50, 1), config[0x80:])
    cmd(struct.pack(">BBI", 0xE3, 0x50, min(len(image), 0xFF00)), image[:0xFF00])
    if len(image) > 0xFF00:
      cmd(struct.pack(">BBI", 0xE3, 0xD0, len(image) - 0xFF00), image[0xFF00:])
    cmd(struct.pack(">BB13x", 0xE8, 0x51))
  finally:
    os.close(fd)
  print(f"[{serial}] stock recovery flash complete", flush=True)


class StockFallback(Exception):
  pass


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
      path, vid_pid, product = installed_chestnut()
      if is_stock(vid_pid, product):
        raise StockFallback("Chestnut fell back to the stock bootloader")
      if path is not None:
        self.fd = claim_device(path)
        return
      time.sleep(0.1)
    raise RuntimeError(f"Chestnut did not enumerate within {timeout:g}s")

  def wr(self, addr, value):
    fcntl.ioctl(self.fd, USBDEVFS_CONTROL,
                Ctrl(0x40, 0xE5, addr & 0xFFFF, value & 0xFFFF, 0, 2000, None))

  def rd(self, addr, length=1):
    buf = (ctypes.c_ubyte * length)()
    fcntl.ioctl(self.fd, USBDEVFS_CONTROL,
                Ctrl(0xC0, 0xE4, addr & 0xFFFF, 0, length, 2000, ctypes.cast(buf, ctypes.c_void_p)))
    return bytes(buf)

  def wrbuf(self, data):
    for i, value in enumerate(data):
      self.wr(0x7000 + i, value)

  def wait_csr(self, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
      if not self.rd(0xC8A9)[0] & 1:
        return
    raise TimeoutError("flash controller timeout")

  def transaction(self, command, addr=0, length=0, addr_len=0x07, mode=0):
    self.wr(0xC8AD, mode)
    self.wr(0xC8AE, 0)
    self.wr(0xC8AF, 0)
    self.wr(0xC8AA, command)
    self.wr(0xC8AC, addr_len)
    self.wr(0xC8A1, addr & 0xFF)
    self.wr(0xC8A2, (addr >> 8) & 0xFF)
    self.wr(0xC8AB, (addr >> 16) & 0xFF)
    self.wr(0xC8A3, (length >> 8) & 0xFF)
    self.wr(0xC8A4, length & 0xFF)
    self.wr(0xC8A9, 1)
    self.wait_csr()
    for _ in range(4):
      self.wr(0xC8AD, 0)

  def wren(self):
    self.wr(0xC8AD, 0)
    self.wr(0xC8AA, 0x06)
    self.wr(0xC8AC, 0x04)
    self.wr(0xC8A3, 0)
    self.wr(0xC8A4, 0)
    self.wr(0xC8A9, 1)
    self.wait_csr()

  def status(self):
    self.transaction(0x05, length=1, addr_len=0x04)
    return self.rd(0x7000)[0]

  def wait_wip(self, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
      if not self.status() & 1:
        return
      time.sleep(0.005)
    raise TimeoutError("SPI flash WIP timeout")

  def init(self):
    self.wr(0xCC33, 0x04)
    self.wr(0xCA81, self.rd(0xCA81)[0] | 1)
    self.wr(0xC805, 0x02)
    self.wr(0xC8A6, 0x04)
    for _ in range(5):
      self.wren()
      self.wrbuf(bytes(4))
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
        out += self.rd(0x7000 + off, min(255, n - off))
    return bytes(out)

  def erase_sector(self, addr):
    self.wren()
    self.transaction(0x20, addr)
    self.wait_wip()

  def program(self, addr, data):
    self.wrbuf(data + bytes((-len(data)) % 4))
    self.wren()
    self.transaction(0x02, addr, len(data), mode=1)
    self.wait_wip()


def validate_wrapped(data):
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
      print(f"  waiting for Chestnut (attempt {attempt}): {e}", flush=True)
      time.sleep(1)


def stable_read(flash, addr, length, count=2):
  attempt = 0
  while True:
    attempt += 1
    try:
      reads = [flash.read(addr, length) for _ in range(count)]
      if any(x != reads[0] for x in reads[1:]):
        raise RuntimeError(f"unstable flash read at 0x{addr:05x}")
      return reads[0]
    except (OSError, TimeoutError, RuntimeError) as e:
      check_budget()
      print(f"  read 0x{addr:05x} attempt {attempt}: {e}", flush=True)
      reconnect(flash)


def program_sector(flash, addr, target):
  attempt = 0
  while True:
    attempt += 1
    try:
      flash.erase_sector(addr)
      if flash.read(addr, SECTOR) != bytes([0xFF]) * SECTOR:
        raise RuntimeError("sector erase verification failed")
      for off in range(0, SECTOR, PAGE):
        chunk = target[off:off + PAGE]
        if chunk != bytes([0xFF]) * len(chunk):
          flash.program(addr + off, chunk)
          if flash.read(addr + off, len(chunk)) != chunk:
            raise RuntimeError(f"page verify failed at 0x{addr + off:05x}")
      if flash.read(addr, SECTOR) == target:
        return
      raise RuntimeError("sector verification failed")
    except (OSError, TimeoutError, RuntimeError) as e:
      check_budget()
      print(f"  sector 0x{addr:05x} attempt {attempt}: {e}", flush=True)
      reconnect(flash)


def backed_up_config(path, data):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  try:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
  except FileExistsError as e:
    backup = open(path, "rb").read()
    if len(backup) != 0x100:
      raise RuntimeError(f"invalid config backup: {path}") from e
    if backup != data:
      print(f"  restoring config from {path}", flush=True)
    return backup
  with os.fdopen(fd, "wb") as f:
    f.write(data)
    f.flush()
    os.fsync(f.fileno())
  return data


def pin_runtime_power(path):
  control = os.path.join(path, "power/control")
  if not os.path.exists(control):
    raise RuntimeError(f"USB power policy missing: {control}")
  with open(control, "w") as f:
    f.write("on\n")
  if open(control).read().strip() != "on":
    raise RuntimeError(f"could not disable USB runtime PM: {control}")
  delay = os.path.join(path, "power/autosuspend_delay_ms")
  if os.path.exists(delay):
    with open(delay, "w") as f:
      f.write("-1\n")


def disable_usb_runtime_pm():
  for path in [
    "/sys/bus/platform/devices/a600000.ssusb/power/control",
    "/sys/bus/usb/devices/usb4/power/control",
  ]:
    if not os.path.exists(path):
      raise RuntimeError(f"USB power policy missing: {path}")
    with open(path, "w") as f:
      f.write("on\n")
    if open(path).read().strip() != "on":
      raise RuntimeError(f"could not disable USB runtime PM: {path}")


def installed_chestnut():
  found = []
  for d in glob.glob("/sys/bus/usb/devices/*"):
    try:
      vid_pid = (open(d + "/idVendor").read().strip(), open(d + "/idProduct").read().strip())
      if vid_pid in VID_PIDS + STOCK_VID_PIDS:
        found.append((d, vid_pid, open(d + "/product").read().strip()))
    except OSError:
      pass
  if len(found) > 1:
    raise RuntimeError(f"expected one Chestnut, found {len(found)}")
  return found[0] if found else (None, None, None)


def installed_product():
  return installed_chestnut()[2]


def is_stock(vid_pid, product):
  # the ROM bootloader shows config-page strings (TinyEnclosure) when a config exists, AS2462/174c without one
  return vid_pid in STOCK_VID_PIDS or product == STOCK_PRODUCT or (product or "").startswith("AS2462")


def vbus_cycle():
  if not os.path.exists(VBUS_PATH):
    return
  for value, delay in (("0", 2.0), ("1", 5.0)):
    try:
      with open(VBUS_PATH, "w") as f:
        f.write(value + "\n")
    except OSError:
      pass
    time.sleep(delay)


def activate(serial):
  # vbus cycle only resets the ASIC when chestnut is bus-powered, report honestly otherwise
  if not os.path.exists(VBUS_PATH):
    print(f"[{serial}] no VBUS control, firmware activates on the next Chestnut power cycle", flush=True)
    return
  print(f"[{serial}] power-cycling Chestnut VBUS", flush=True)
  try:
    with open(VBUS_PATH, "w") as f:
      f.write("0\n")
  except OSError:
    pass
  disconnected = False
  deadline = time.monotonic() + 5.0
  while time.monotonic() < deadline:
    if installed_product() is None:
      disconnected = True
      break
    time.sleep(0.2)
  time.sleep(1)
  try:
    with open(VBUS_PATH, "w") as f:
      f.write("1\n")
  except OSError:
    pass
  if not disconnected:
    print(f"[{serial}] Chestnut is externally powered, firmware activates on its next power cycle", flush=True)
    return
  deadline = time.monotonic() + 15.0
  while time.monotonic() < deadline:
    product = installed_product()
    if product is not None:
      if product == EXPECTED_PRODUCT:
        print(f"[{serial}] ACTIVATED {EXPECTED_PRODUCT}", flush=True)
      else:
        print(f"[{serial}] Chestnut re-enumerated with {product!r}, firmware activates on its next power cycle", flush=True)
      return
    time.sleep(0.2)
  print(f"[{serial}] Chestnut did not re-enumerate, firmware activates on its next power cycle", flush=True)


def flash_chestnut(dry_run=False, force=False):
  global _deadline
  serial = os.uname().nodename

  image = FIRMWARE_PATH.read_bytes()
  if hashlib.sha256(image).hexdigest() != FIRMWARE_SHA256:
    raise RuntimeError("bundled Chestnut firmware checksum mismatch")
  validate_wrapped(image)

  path, vid_pid, product = installed_chestnut()
  if path is None:
    print(f"[{serial}] no Chestnut connected", flush=True)
    return
  if product == EXPECTED_PRODUCT and not force:
    print(f"[{serial}] Chestnut firmware is up to date ({EXPECTED_PRODUCT})", flush=True)
    return

  _deadline = time.monotonic() + FLASH_BUDGET

  if is_stock(vid_pid, product):
    if dry_run:
      print(f"[{serial}] DRY RUN: would recover from stock bootloader mode", flush=True)
      return
    backup = os.path.join(BACKUP_DIR, f"{serial}.config.bin")
    if not os.path.isfile(backup):
      raise RuntimeError(f"cannot recover from stock mode without a config backup at {backup}")
    config = open(backup, "rb").read()
    if len(config) != 0x100:
      raise RuntimeError(f"invalid config backup: {backup}")
    disable_usb_runtime_pm()
    committed = False
    while True:
      check_budget()
      path, vid_pid, product = installed_chestnut()
      if path is None:
        if committed:
          print(f"[{serial}] Chestnut is offline, recovered firmware boots on its next power cycle", flush=True)
          return
        vbus_cycle()
        continue
      if not is_stock(vid_pid, product):
        break
      if committed:
        print(f"[{serial}] Chestnut is externally powered, recovered firmware boots on its next power cycle", flush=True)
        return
      try:
        stock_recover(serial, image, config)
        committed = True
      except (OSError, TimeoutError, RuntimeError) as e:
        print(f"  stock recovery failed, retrying: {e}", flush=True)
        vbus_cycle()
        continue
      activate(serial)
    force = True  # firmware is running again, fall through for a full readback verification

  if force:
    print(f"[{serial}] forced reflash of {EXPECTED_PRODUCT}", flush=True)
  else:
    print(f"[{serial}] Chestnut firmware mismatch: {product!r}; expected {EXPECTED_PRODUCT!r}", flush=True)

  disable_usb_runtime_pm()
  flash = Flash()
  prev_handlers = {}

  def finish_safely(signum, _frame):
    print(f"signal {signum} deferred until flashing and verification complete", flush=True)

  try:
    reconnect(flash)
    config = stable_read(flash, 0, 0x100, 3)
    config = backed_up_config(os.path.join(BACKUP_DIR, f"{serial}.config.bin"), config)
    image_end = IMAGE_OFFSET + len(image)
    first_sector = IMAGE_OFFSET & ~(SECTOR - 1)
    span = (image_end + SECTOR - 1) & ~(SECTOR - 1)
    current = stable_read(flash, first_sector, span - first_sector)
    target = bytearray(current)
    target[:len(config)] = config
    target[IMAGE_OFFSET - first_sector:image_end - first_sector] = image
    target = bytes(target)
    print(f"[{serial}] target {len(image)} bytes at 0x{IMAGE_OFFSET:05x}, sha256={FIRMWARE_SHA256}", flush=True)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
      prev_handlers[sig] = signal.signal(sig, finish_safely)

    for addr in range(first_sector, span, SECTOR):
      off = addr - first_sector
      wanted = target[off:off + SECTOR]
      if current[off:off + SECTOR] == wanted:
        print(f"  sector 0x{addr:05x}: unchanged", flush=True)
        continue
      if dry_run:
        changed = sum(a != b for a, b in zip(current[off:off + SECTOR], wanted, strict=True))
        print(f"  sector 0x{addr:05x}: would program ({changed} bytes differ)", flush=True)
        continue
      print(f"  sector 0x{addr:05x}: programming", flush=True)
      program_sector(flash, addr, wanted)

    if dry_run:
      print(f"[{serial}] DRY RUN OK", flush=True)
      return
    verified = stable_read(flash, first_sector, span - first_sector, 3)
    if verified != target:
      raise RuntimeError("final full-image verification failed")
    print(f"[{serial}] VERIFY OK sha256={hashlib.sha256(verified).hexdigest()}", flush=True)
  finally:
    flash.close()
    for sig, handler in prev_handlers.items():
      signal.signal(sig, handler)

  activate(serial)


def main():
  parser = argparse.ArgumentParser(description="Check and flash the bundled Chestnut firmware")
  parser.add_argument("--dry-run", action="store_true", help="validate and compare without erasing or programming")
  parser.add_argument("--force", action="store_true", help="verify/reflash even when the version matches")
  args = parser.parse_args()
  if os.geteuid() != 0:
    os.execvp("sudo", ["sudo", sys.executable, os.path.abspath(__file__)] + sys.argv[1:])
  flash_chestnut(dry_run=args.dry_run, force=args.force)


if __name__ == "__main__":
  try:
    main()
  except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)
