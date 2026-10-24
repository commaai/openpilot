#!/usr/bin/env python3
"""Direct Navi31 ROM_SW access for GD25LQ16E-class 2 MiB SPI flash."""
from __future__ import annotations
import argparse, hashlib, sys, time
from pathlib import Path
from common import MMIO, open_gpu, wait_until

FLASH_SIZE, SECTOR_SIZE, PAGE_SIZE, MAX_DATA = 0x200000, 0x1000, 0x100, 0x100
ROM_CNTL, PAGE_MIRROR_CNTL = 0x5A380, 0x5A384
ROM_SW_CNTL, ROM_SW_STATUS, ROM_SW_COMMAND, ROM_SW_DATA = 0x5A3A0, 0x5A3A4, 0x5A3A8, 0x5A3B0
GPIO_PAD_MASK, GPIO_PAD_A, GPIO_PAD_EN = 0x5A504, 0x5A508, 0x5A510
SPI_GPIO_BITS, RETURN_DATA_EN = 0x780, 0x80000
EXPECTED_JEDEC = b'\xc8\x60\x15'


class Navi31SPI:
  def __init__(self, pci_dev, prescale: int = 8):
    if not 0 <= prescale <= 15: raise ValueError("prescale must be 0..15")
    self.mmio = MMIO(pci_dev)
    rc = self.mmio.read32(ROM_CNTL)
    # Select the prescaler instead of inheriting a potentially unusable BL value.
    self.mmio.write32(ROM_CNTL, (rc & 0xE0FFFFFF) | (1 << 28) | (prescale << 24) | 1)

  def transfer(self, opcode: int, *, address: int = 0, address_len: int = 0,
               data_out: bytes = b'', data_in: int = 0, timeout: float = 2.0) -> bytes:
    if data_out and data_in: raise ValueError("simultaneous TX and RX is unsupported")
    if not 0 <= address_len <= 3: raise ValueError("address_len must be 0..3")
    count = len(data_out) if data_out else data_in
    if not 0 <= count <= MAX_DATA: raise ValueError(f"transfer data must be <= {MAX_DATA} bytes")
    ncmd = 1 + address_len
    m = self.mmio
    gpio_mask, gpio_a, gpio_en = m.read32(GPIO_PAD_MASK), m.read32(GPIO_PAD_A), m.read32(GPIO_PAD_EN)
    page_mirror, rom_cntl = m.read32(PAGE_MIRROR_CNTL), m.read32(ROM_CNTL)
    try:
      m.write32(GPIO_PAD_MASK, gpio_mask & ~SPI_GPIO_BITS)
      m.write32(GPIO_PAD_A, gpio_a & ~SPI_GPIO_BITS)
      m.write32(GPIO_PAD_EN, gpio_en & ~SPI_GPIO_BITS)
      m.write32(PAGE_MIRROR_CNTL, (page_mirror & 0xF1FFFFFF) | 0x06000000)
      m.write32(ROM_CNTL, (rom_cntl & ~0xF) | 8)
      m.write32(ROM_SW_CNTL, 0)
      m.write32(ROM_SW_STATUS, 0)
      if m.read32(ROM_SW_STATUS) != 0: raise RuntimeError("ROM_SW_STATUS did not clear")

      # Navi31 serializes the low instruction byte first, followed by ADDRESS[23:0].
      m.write32(ROM_SW_COMMAND, ((address & 0xFFFFFF) << 8) | (opcode & 0xFF))
      for offset in range(0, len(data_out), 4):
        word = data_out[offset:offset+4].ljust(4, b'\0')
        m.write32(ROM_SW_DATA + offset, int.from_bytes(word, 'big'))

      control = ((ncmd - 1) << 16) | (RETURN_DATA_EN if data_in else 0) | count
      m.write32(ROM_SW_CNTL, control)
      m.read32(ROM_SW_CNTL)  # posted-write flush
      wait_until(lambda: m.read32(ROM_SW_STATUS) & 1, timeout,
                 f"ROM_SW transaction timeout (status={m.read32(ROM_SW_STATUS):#x}); engine may be gated after SOS boot")
      return m.read(ROM_SW_DATA, (data_in + 3) & ~3)[:data_in] if data_in else b''
    finally:
      m.write32(ROM_SW_CNTL, 0)
      m.write32(ROM_SW_STATUS, 0)
      m.write32(ROM_CNTL, rom_cntl)
      m.write32(PAGE_MIRROR_CNTL, page_mirror)
      m.write32(GPIO_PAD_A, gpio_a)
      m.write32(GPIO_PAD_EN, gpio_en)
      m.write32(GPIO_PAD_MASK, gpio_mask)


class GD25LQ16E:
  def __init__(self, spi: Navi31SPI): self.spi = spi

  def read_register(self, opcode: int, count: int = 1) -> bytes:
    # Navi31 exposes the preceding transaction's RX capture. Prime identically.
    self.spi.transfer(opcode, data_in=max(2, count))
    return self.spi.transfer(opcode, data_in=count)

  def status(self, opcode: int = 0x05) -> int: return self.read_register(opcode)[0]
  def rdid(self) -> bytes: return self.read_register(0x9F, 4)

  def sfdp(self, count: int = 20) -> bytes:
    # 5Ah has one dummy byte after its 24-bit address; retain it for diagnostics.
    self.spi.transfer(0x5A, address_len=3, data_in=count)
    return self.spi.transfer(0x5A, address_len=3, data_in=count)

  def wait_idle(self, timeout: float = 2.0) -> int:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
      sr1 = self.status()
      if not sr1 & 1: return sr1
      time.sleep(0.002)
    raise TimeoutError(f"flash remained busy for {timeout}s")

  def write_enable(self):
    self.spi.transfer(0x06)
    sr1 = self.status()
    if not sr1 & 2: raise RuntimeError(f"WREN failed (SR1={sr1:#04x})")

  def clear_cmp(self):
    sr1, sr2 = self.status(), self.status(0x35)
    if not sr2 & 0x40: return False
    self.write_enable()
    # BUSY/WEL are not writable; preserve all protection/QE fields except CMP.
    self.spi.transfer(0x01, data_out=bytes((sr1 & 0xFC, sr2 & ~0x40)))
    self.wait_idle(1.0)
    new_sr2 = self.status(0x35)
    if new_sr2 & 0x40: raise RuntimeError(f"failed to clear CMP (SR2={new_sr2:#04x})")
    return True

  def erase_sector(self, address: int):
    if address & (SECTOR_SIZE - 1): raise ValueError("sector address is not 4 KiB aligned")
    self.write_enable()
    self.spi.transfer(0x20, address=address, address_len=3)
    self.wait_idle(2.0)

  def program_page(self, address: int, data: bytes):
    if not data or len(data) > PAGE_SIZE or (address & 0xFF) + len(data) > PAGE_SIZE:
      raise ValueError("page program crosses a 256-byte boundary")
    self.write_enable()
    self.spi.transfer(0x02, address=address, address_len=3, data_out=data)
    self.wait_idle(1.0)

  def read(self, address: int, count: int) -> bytes:
    if address < 0 or count < 0 or address + count > FLASH_SIZE: raise ValueError("read outside 2 MiB flash")
    output = bytearray()
    while count:
      size = min(count, MAX_DATA)
      self.spi.transfer(0x03, address=address, address_len=3, data_in=size)
      output += self.spi.transfer(0x03, address=address, address_len=3, data_in=size)
      address, count = address + size, count - size
    return bytes(output)


def has_jedec(raw: bytes) -> bool:
  return EXPECTED_JEDEC in raw + raw[:2]


def open_flash(args) -> GD25LQ16E:
  flash = GD25LQ16E(Navi31SPI(open_gpu(args.device, 'usb'), args.prescale))
  raw = flash.rdid()
  if not has_jedec(raw): raise RuntimeError(f"unexpected GD25LQ16E JEDEC capture: {raw.hex()}")
  return flash


def cmd_info(args):
  f = open_flash(args)
  sr1, sr2, sr3 = f.status(), f.status(0x35), f.status(0x15)
  sfdp = f.sfdp(24)
  pos = sfdp.find(b'SFDP')
  print(f"JEDEC capture: {f.rdid().hex()} (C8 60 15 detected)")
  print(f"SR1/SR2/SR3: {sr1:02x}/{sr2:02x}/{sr3:02x}  CMP={'set' if sr2 & 0x40 else 'clear'}")
  print(f"SFDP capture: {sfdp.hex()}  signature_offset={pos}")


def cmd_read(args):
  data = open_flash(args).read(args.address, args.size)
  if args.output: Path(args.output).write_bytes(data)
  else: print(data.hex())


def cmd_dump(args):
  f = open_flash(args)
  out = Path(args.output)
  digest = hashlib.sha256()
  with out.open('wb') as file:
    for address in range(0, FLASH_SIZE, SECTOR_SIZE):
      data = f.read(address, SECTOR_SIZE)
      file.write(data)
      digest.update(data)
      if not (address & 0xFFFF): print(f"{address + SECTOR_SIZE:#08x}/{FLASH_SIZE:#08x}", flush=True)
  print(f"wrote {out} sha256={digest.hexdigest()}")


def cmd_verify(args):
  expected = Path(args.image).read_bytes()
  if len(expected) != FLASH_SIZE: raise ValueError(f"image must be exactly {FLASH_SIZE:#x} bytes")
  f = open_flash(args)
  digest = hashlib.sha256()
  for address in range(0, FLASH_SIZE, SECTOR_SIZE):
    got, wanted = f.read(address, SECTOR_SIZE), expected[address:address+SECTOR_SIZE]
    digest.update(got)
    if got != wanted:
      index = next(i for i, (a, b) in enumerate(zip(got, wanted)) if a != b)
      raise RuntimeError(f"verify mismatch at {address+index:#x}: flash={got[index]:02x} image={wanted[index]:02x}")
  print(f"verified {FLASH_SIZE:#x} bytes sha256={digest.hexdigest()}")


def cmd_flash(args):
  if not args.yes: raise RuntimeError("refusing to write without --yes")
  image = Path(args.image).read_bytes()
  if len(image) != FLASH_SIZE: raise ValueError(f"image must be exactly {FLASH_SIZE:#x} bytes")
  total_sectors = FLASH_SIZE // SECTOR_SIZE
  start, count = args.start_sector, args.sector_count if args.sector_count is not None else total_sectors - args.start_sector
  if not 0 <= start < total_sectors or not 1 <= count <= total_sectors - start: raise ValueError("invalid sector range")
  f = open_flash(args)
  if f.status(0x35) & 0x40:
    if not args.clear_cmp: raise RuntimeError("CMP protects the full array; rerun with --clear-cmp")
    f.clear_cmp()
    print("cleared SR2.CMP", flush=True)
  begin = time.monotonic()
  for sector in range(start, start + count):
    address = sector * SECTOR_SIZE
    wanted = image[address:address+SECTOR_SIZE]
    f.erase_sector(address)
    for offset in range(0, SECTOR_SIZE, PAGE_SIZE):
      page = wanted[offset:offset+PAGE_SIZE]
      if page != b'\xff' * PAGE_SIZE: f.program_page(address + offset, page)
    got = f.read(address, SECTOR_SIZE)
    if got != wanted:
      index = next(i for i, (a, b) in enumerate(zip(got, wanted)) if a != b)
      raise RuntimeError(f"verify mismatch at {address+index:#x}: flash={got[index]:02x} image={wanted[index]:02x}")
    print(f"OK sector {sector:03d}/{total_sectors-1} @{address:#07x} elapsed={time.monotonic()-begin:.1f}s", flush=True)


def parser():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--device', type=int, default=0, help='USB bridge device index')
  p.add_argument('--prescale', type=int, default=8, help='SCK prescaler 0..15 (default: 8)')
  sub = p.add_subparsers(dest='command', required=True)
  sub.add_parser('info', help='read JEDEC, status and SFDP').set_defaults(func=cmd_info)

  r = sub.add_parser('read', help='read a flash range')
  r.add_argument('address', type=lambda x:int(x, 0))
  r.add_argument('size', type=lambda x:int(x, 0))
  r.add_argument('-o', '--output')
  r.set_defaults(func=cmd_read)

  d = sub.add_parser('dump', help='dump the complete 2 MiB flash')
  d.add_argument('output')
  d.set_defaults(func=cmd_dump)

  v = sub.add_parser('verify', help='compare the complete flash with an image')
  v.add_argument('image')
  v.set_defaults(func=cmd_verify)

  w = sub.add_parser('flash', help='erase, program, and verify one or more sectors')
  w.add_argument('image')
  w.add_argument('--start-sector', type=lambda x:int(x, 0), default=0)
  w.add_argument('--sector-count', type=lambda x:int(x, 0))
  w.add_argument('--clear-cmp', action='store_true')
  w.add_argument('--yes', action='store_true')
  w.set_defaults(func=cmd_flash)
  return p


def main():
  args = parser().parse_args()
  try: args.func(args)
  except (RuntimeError, TimeoutError, ValueError, OSError) as error:
    print(f"error: {error}", file=sys.stderr)
    raise SystemExit(1)

if __name__ == '__main__': main()
