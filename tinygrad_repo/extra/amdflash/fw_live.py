#!/usr/bin/env python3
"""Navi31 firmware-mediated flash access and ROM aperture dumping.

Early item streaming must run after autonomous PSP boot but before a host
driver or AMDev loads SOS. A fully initialized SOS rejects those commands.
"""
from __future__ import annotations
import argparse, struct, sys, time
from pathlib import Path
from common import MMIO, open_gpu, wait_until

ROM_CNTL, ROM_INDEX, ROM_DATA = 0x5A380, 0x5A390, 0x5A394
FLASH_SIZE, INDEX_PAGE = 0x200000, 0x10000

def bswap32(value: int) -> int: return int.from_bytes(value.to_bytes(4, 'little'), 'big')

COMMAND_DATA, COMMAND, DOORBELL = 0x582D0, 0x582CC, 0x58224
GET_BOOT_PARTITION, GET_FB_STATE, GET_TRANSFER_TYPE = 0x01, 0x06, 0x07
START_TRANSFER, DATA_TRANSFER, END_TRANSFER = 0x08, 0x09, 0x0A
SPI_GET_MODEL_ID = 0x0B
LIVE_ADDR_LO, LIVE_ADDR_HI, LIVE_UPDATE = 0x02, 0x03, 0x04
PSP_ERRORS = {
  0x01: "generic error", 0x02: "out of bounds", 0x03: "invalid parameter",
  0x04: "off-chip boot error", 0x05: "address not set", 0x06: "parse off-chip error",
  0x07: "address map error", 0x08: "parse on-chip error", 0x09: "full update error",
  0x0A: "partition update error", 0x0B: "map on-chip error", 0x0C: "write to SPI error",
  0x0D: "signature validation error", 0x0E: "invalid command", 0x0F: "signature not found",
  0x10: "state machine not initialized", 0x11: "state machine transfer error",
  0x12: "initialization error",
}


class PSPFlashMailbox:
  def __init__(self, pci_dev): self.mmio = MMIO(pci_dev)

  def command(self, command: int, data: int | None = None, *, timeout: float = 10.0) -> tuple[int, int]:
    status = self.mmio.read32(COMMAND)
    if not status & 0x80000000:
      raise RuntimeError(f"PSP mailbox is not ready before command {command:#x}: status={status:#010x}")
    if data is not None: self.mmio.write32(COMMAND_DATA, data)
    self.mmio.write32(COMMAND, command << 16)
    self.mmio.write32(DOORBELL, 1)
    wait_until(lambda: self.mmio.read32(COMMAND) & 0x80000000, timeout,
               f"PSP mailbox command {command:#x} timed out")
    value = self.mmio.read32(COMMAND)
    return value & 0xffff, self.mmio.read32(COMMAND_DATA)

  def require(self, command: int, data: int | None = None, *, timeout: float = 10.0, name: str = '') -> int:
    error, response = self.command(command, data, timeout=timeout)
    if error:
      detail = PSP_ERRORS.get(error, "unknown error")
      raise RuntimeError(f"PSP {name or hex(command)} failed: error={error:#x} ({detail})")
    return response

  def probe(self) -> dict[str, tuple[int, int]]:
    result = {}
    for name, command in (("boot_partition", GET_BOOT_PARTITION), ("fb_state", GET_FB_STATE),
                          ("model_id", SPI_GET_MODEL_ID), ("transfer_type", GET_TRANSFER_TYPE)):
      result[name] = self.command(command)
    return result

  def stream(self, payload: bytes, item_type: int, transfer_type: int | None = None):
    if not payload: raise ValueError("payload is empty")
    if len(payload) > 0xFFFFFF: raise ValueError("payload exceeds the mailbox's 24-bit size field")
    if len(payload) & 3: raise ValueError("payload size must be divisible by four")
    if not 0 <= item_type <= 0xff: raise ValueError("item type must fit in eight bits")
    if transfer_type is None: transfer_type = self.require(GET_TRANSFER_TYPE, name="GET_TRANSFER_TYPE")
    requested = transfer_type & 0xff
    print(f"firmware transfer_type={transfer_type:#x}", flush=True)
    if requested != item_type:
      raise RuntimeError(f"firmware requests item {requested:#x}, not {item_type:#x}")
    self.require(START_TRANSFER, (len(payload) << 8) | item_type, name="START_TRANSFER")
    sent, started = 0, time.monotonic()
    try:
      for offset in range(0, len(payload), 4):
        word = struct.unpack_from('<I', payload, offset)[0]
        self.require(DATA_TRANSFER, word, name=f"DATA_TRANSFER@{offset:#x}")
        sent = offset + 4
        if sent % 0x1000 == 0:
          print(f"{sent:#x}/{len(payload):#x} ({sent/(time.monotonic()-started)/1024:.1f} KiB/s)", flush=True)
      self.require(END_TRANSFER, (sent << 8) | item_type, timeout=60.0, name="END_TRANSFER")
    except BaseException:
      # Give firmware a chance to terminate an interrupted partial session. Do
      # not submit END_TRANSFER twice if firmware rejected the original END.
      if sent != len(payload):
        try: self.command(END_TRANSFER, (sent << 8) | item_type, timeout=10.0)
        except Exception: pass
      raise
    print(f"stream complete: type={item_type:#x} size={sent:#x} elapsed={time.monotonic()-started:.1f}s")


def resolve_ifwi_item(image: bytes, item_type: int) -> tuple[int, bytes]:
  """Resolve AMDVBFlash recovery-layout item types to exact IFWI bytes."""
  if item_type == 0x01: offset, size = 0, 0x54
  elif item_type in (0x02, 0x03):
    offset = 0x2000 if item_type == 0x02 else 0x3000
    if image[offset:offset+4] != b'$PSP': raise ValueError(f"invalid PSP directory at {offset:#x}")
    size = (struct.unpack_from('<I', image, offset + 8)[0] + 1) * 0x10
  elif item_type == 0x04: offset, size = 0x10000, 0x1000
  elif item_type == 0x05: offset, size = 0x11000, 0x1000
  elif item_type == 0x06: offset, size = 0x12000, 0x20
  elif item_type == 0x07: offset, size = 0x13000, 0x20
  elif item_type == 0x80: offset, size = 0x1000, 4
  elif item_type == 0x81:
    offset = struct.unpack_from('<I', image, 0x1000)[0]
    if image[offset:offset+4] != b'$SGN': raise ValueError("invalid $SGN table pointer")
    size = (struct.unpack_from('<I', image, offset + 8)[0] + 1) * 0x10
  elif 0x82 <= item_type <= 0x88:
    table = struct.unpack_from('<I', image, 0x1000)[0]
    if image[table:table+4] != b'$SGN': raise ValueError("invalid $SGN table pointer")
    wanted = item_type - 0x81  # 82h..88h map to SIGN_TYPE 1..7
    count = struct.unpack_from('<I', image, table + 8)[0]
    entries = [struct.unpack_from('<IIII', image, table + 0x10 + i*0x10) for i in range(count)]
    match = [entry for entry in entries if entry[0] == wanted]
    if len(match) != 1: raise ValueError(f"missing $SGN type {wanted}")
    _, _, size, offset = match[0]
  elif item_type == 0x89: offset, size = 0x1f0000, 0x100
  elif item_type == 0x08:
    # AMDVBFlash's GetPartitionDetails follows the first ISH entry (firmware ID
    # 0x13c) and streams its payload. PSP, not the host resolver, selects the
    # destination partition.
    offset = struct.unpack_from('<I', image, 0x12000 + 0x10)[0]
    size = struct.unpack_from('<I', image, 0x12000 + 0x18)[0]
  else:
    raise ValueError(f"IFWI resolver does not yet support requested item {item_type:#x}")
  payload = image[offset:offset+size]
  if len(payload) != size: raise ValueError(f"item {item_type:#x} extends beyond IFWI")
  print(f"resolved requested item {item_type:#x}: offset={offset:#x} size={size:#x}")
  return offset, payload


class LivePSPFlash:
  """Linux psp_v13_0_update_spirom protocol, used with SOS and trained VRAM."""
  def __init__(self, pci_dev): self.mailbox = PSPFlashMailbox(pci_dev)

  def command(self, command: int, data: int | None = None, timeout: float = 10.0):
    # Same C2PMSG registers, but the live PSP command set uses IDs 2/3/4.
    return self.mailbox.require(command, data, timeout=timeout, name=f"LIVE_SPI_{command:#x}")

  def update(self, mc_address: int):
    status = self.mailbox.mmio.read32(COMMAND)
    if not status & 0x80000000: raise RuntimeError(f"live PSP mailbox is not ready: {status:#x}")
    self.command(LIVE_ADDR_LO, mc_address & 0xffffffff)
    self.command(LIVE_ADDR_HI, mc_address >> 32)
    self.command(LIVE_UPDATE, timeout=60.0)


def open_mailbox(args): return PSPFlashMailbox(open_gpu(args.device, args.transport))


def reject_unvalidated_firmware_write():
  raise RuntimeError("firmware writes are disabled: stock reflash validation failed; use romless.py for recovery")


def cmd_probe(args):
  result = open_mailbox(args).probe()
  for name, (error, response) in result.items(): print(f"{name}: error={error:#x} response={response:#x}")
  if result['transfer_type'][0] == 0xA: print("update commands gated: reset card and do not initialize AMDev/SOS", file=sys.stderr)


def cmd_stream(args):
  if not args.yes: raise RuntimeError("refusing to stream without --yes")
  reject_unvalidated_firmware_write()
  payload = Path(args.image).read_bytes()
  open_mailbox(args).stream(payload, args.item_type)


def cmd_ifwi_step(args):
  if not args.yes: raise RuntimeError("refusing to stream without --yes")
  reject_unvalidated_firmware_write()
  image = Path(args.ifwi).read_bytes()
  if len(image) != 0x200000: raise ValueError("Navi31 IFWI image must be exactly 2 MiB")
  mailbox = open_mailbox(args)
  state = mailbox.require(GET_TRANSFER_TYPE, name="GET_TRANSFER_TYPE")
  request = state & 0xff
  _, payload = resolve_ifwi_item(image, request)
  mailbox.stream(payload, request, transfer_type=state)
  next_request = mailbox.require(GET_TRANSFER_TYPE, name="GET_TRANSFER_TYPE")
  print(f"next firmware transfer_type={next_request:#x}")


def cmd_ifwi_all(args):
  if not args.yes: raise RuntimeError("refusing to stream without --yes")
  image = Path(args.ifwi).read_bytes()
  if len(image) != 0x200000: raise ValueError("Navi31 IFWI image must be exactly 2 MiB")
  mailbox = open_mailbox(args)
  current = mailbox.require(GET_TRANSFER_TYPE, name="GET_TRANSFER_TYPE")
  for step in range(19):  # Navi31 ROMItemCount from AMDVBFlash ASICDetails.xml
    request, phase = current & 0xff, current >> 8
    print(f"IFWI step {step}: state={current:#x} item={request:#x} phase={phase}", flush=True)
    _, payload = resolve_ifwi_item(image, request)
    mailbox.stream(payload, request, transfer_type=current)
    # AMDVBFlash tests the high byte belonging to the item just streamed. Phase
    # 2 terminates the loop only after that item has completed successfully.
    if phase == 2:
      print(f"IFWI stream complete after terminal state {current:#x}; hard power cycle required")
      return
    current = mailbox.require(GET_TRANSFER_TYPE, name="GET_TRANSFER_TYPE")
  raise RuntimeError(f"IFWI stream did not reach terminal phase after 19 items (state={current:#x})")


def cmd_live_flash(args):
  if not args.yes: raise RuntimeError("refusing to flash without --yes")
  reject_unvalidated_firmware_write()
  image = Path(args.ifwi).read_bytes()
  if not image or len(image) > 16 * 1024 * 1024 or len(image) & 3:
    raise ValueError("live PSP image must be non-empty, 4-byte aligned, and at most 16 MiB")
  pci_dev = open_gpu(args.device, args.transport)
  from tinygrad.runtime.support.am.amdev import AMDev
  started = time.monotonic()
  adev = AMDev(pci_dev)
  print(f"AMDev booted, SOS alive={adev.psp.is_sos_alive()}", flush=True)
  paddr = adev.mm.palloc(len(image), align=0x1000, zero=False)
  try:
    adev.vram.view(paddr, len(image), 'B')[:] = image
    adev.gmc.flush_hdp()
    mc_address = adev.paddr2mc(paddr)
    print(f"staged IFWI at VRAM paddr={paddr:#x} mc={mc_address:#x}", flush=True)
    LivePSPFlash(pci_dev).update(mc_address)
    print(f"live PSP flash update complete in {time.monotonic()-started:.1f}s")
  finally:
    adev.mm.pfree(paddr)


def cmd_dump(args):
  import hashlib
  pci_dev = open_gpu(args.device, args.transport)
  mmio, output, started = MMIO(pci_dev), bytearray(), time.monotonic()
  original_cntl, original_index = mmio.read32(ROM_CNTL), mmio.read32(ROM_INDEX)
  if original_cntl == 0xFFFFFFFF:
    raise RuntimeError("raw SMUIO ROM controller is unavailable; the SOC15 function-ROM aperture is not a raw SPI dump")
  try:
    # ROM_DATA must be read one dword at a time; a block read increments MMIO
    # addresses rather than repeatedly reading the flash aperture register.
    mmio.write32(ROM_CNTL, bswap32(original_cntl | (1 << 29)))
    for page in range(0, FLASH_SIZE, INDEX_PAGE):
      mmio.write32(ROM_INDEX, bswap32(page >> 8))
      for _ in range(INDEX_PAGE // 4): output += struct.pack('<I', mmio.read32(ROM_DATA))
      print(f"{page+INDEX_PAGE:#08x}/{FLASH_SIZE:#08x}", flush=True)
  finally:
    mmio.write32(ROM_INDEX, bswap32(original_index))
    mmio.write32(ROM_CNTL, bswap32(original_cntl))
  if len(output) != FLASH_SIZE or output[:4] != b'\xaa\x55\xaa\x55':
    raise RuntimeError(f"invalid raw flash dump: size={len(output):#x} magic={output[:4].hex()}")
  if output[:FLASH_SIZE//2] == output[FLASH_SIZE//2:]:
    raise RuntimeError("ROM aperture contains mirrored 1 MiB halves; refusing to write a non-raw 2 MiB dump")
  Path(args.output).write_bytes(output)
  print(f"dumped {len(output):#x} bytes in {time.monotonic()-started:.1f}s sha256={hashlib.sha256(output).hexdigest()}")


def parser():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--device', type=int, default=0, help='device index for the selected transport')
  p.add_argument('--transport', choices=('auto', 'usb', 'pci'), default='auto', help='PCIe transport (default: USB first, then native PCI)')
  sub = p.add_subparsers(dest='command', required=True)
  sub.add_parser('probe', help='query firmware mailbox state without writing').set_defaults(func=cmd_probe)

  s = sub.add_parser('stream', help='stream one exact PSP ROM-item payload')
  s.add_argument('item_type', type=lambda x:int(x, 0))
  s.add_argument('image')
  s.add_argument('--yes', action='store_true')
  s.set_defaults(func=cmd_stream)

  v = sub.add_parser('ifwi-step', help='resolve and stream the next early-firmware-requested item from a 2 MiB IFWI')
  v.add_argument('ifwi')
  v.add_argument('--yes', action='store_true')
  v.set_defaults(func=cmd_ifwi_step)

  a = sub.add_parser('ifwi-all', help='stream requested IFWI items until firmware reports completion')
  a.add_argument('ifwi')
  a.add_argument('--yes', action='store_true')
  a.set_defaults(func=cmd_ifwi_all)

  l = sub.add_parser('live-flash', help='stage an image in VRAM and invoke the PSP v13 live-update command')
  l.add_argument('ifwi')
  l.add_argument('--yes', action='store_true')
  l.set_defaults(func=cmd_live_flash)

  d = sub.add_parser('dump', help='dump the exact 2 MiB flash through ROM_INDEX/ROM_DATA')
  d.add_argument('output')
  d.set_defaults(func=cmd_dump)
  return p


def main():
  args = parser().parse_args()
  try: args.func(args)
  except (RuntimeError, TimeoutError, ValueError, OSError) as error:
    print(f"error: {error}", file=sys.stderr)
    raise SystemExit(1)

if __name__ == '__main__': main()
