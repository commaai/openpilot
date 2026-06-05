import ctypes, struct, dataclasses, array, itertools, time, functools
from typing import Sequence
from tinygrad.runtime.autogen import libusb
from tinygrad.helpers import DEBUG, DEV, to_mv, round_up, OSX, getenv, ceildiv
from tinygrad.runtime.support.hcq import MMIOInterface
from tinygrad.runtime.support import c

def alloc_cbuffer(sz:int) -> tuple[ctypes.Array, memoryview]: return (buf:=(ctypes.c_ubyte * sz)()), to_mv(ctypes.addressof(buf), sz)
def checked(fn, msg=None):
  @functools.wraps(fn)
  def wrapper(*args):
    if (rc:=fn(*args)) < 0: raise RuntimeError(f"{msg or fn.__name__}: {ctypes.string_at(libusb.libusb_strerror(rc)).decode()}")
    return rc
  return wrapper

class USB3:
  @staticmethod
  @functools.cache
  def ctx():
    ctx = c.init_c_var(ctypes.POINTER(libusb.struct_libusb_context), checked(libusb.libusb_init))
    if DEBUG >= 6: checked(libusb.libusb_set_option)(ctx, libusb.LIBUSB_OPTION_LOG_LEVEL, 4)
    return ctx

  @classmethod
  @functools.cache
  def list_devices(cls, vendor:int, dev:int) -> list[tuple[c.POINTER[libusb.struct_libusb_device], str]]:
    ret = []
    for i in range(checked(libusb.libusb_get_device_list)(cls.ctx(), devs:=ctypes.POINTER(ctypes.POINTER(libusb.struct_libusb_device))())):
      desc = c.init_c_var(libusb.struct_libusb_device_descriptor, lambda x: checked(libusb.libusb_get_device_descriptor)(devs[i], x))
      if (desc.idVendor, desc.idProduct) == (vendor, dev):
        ret.append((libusb.libusb_ref_device(devs[i]), f"usb:{libusb.libusb_get_bus_number(devs[i])}-{libusb.libusb_get_device_address(devs[i])}"))
    libusb.libusb_free_device_list(devs, 1)
    return ret

  def __init__(self, dev:c.POINTER[libusb.struct_libusb_device], ep_data_in:int, ep_stat_in:int, ep_data_out:int, ep_cmd_out:int,
               max_streams:int=31, use_bot=False):
    self.ep_data_in, self.ep_stat_in, self.ep_data_out, self.ep_cmd_out = ep_data_in, ep_stat_in, ep_data_out, ep_cmd_out
    self.max_streams, self.use_bot = max_streams, use_bot
    self._transferred = ctypes.c_int(0)
    self._bulk_in_buf, self._bulk_in_mv = alloc_cbuffer(4 << 20)
    self._bulk_out_buf, self._bulk_out_mv = alloc_cbuffer(4 << 20)

    self.handle = c.init_c_var(c.POINTER[libusb.struct_libusb_device_handle], lambda x: checked(libusb.libusb_open)(dev, x))

    # Read product string descriptor
    _buf = (ctypes.c_ubyte * 256)()
    _desc = libusb.struct_libusb_device_descriptor()
    checked(libusb.libusb_get_device_descriptor)(libusb.libusb_get_device(self.handle), ctypes.byref(_desc))
    _ret = checked(libusb.libusb_get_string_descriptor_ascii)(self.handle, _desc.iProduct, _buf, 256)
    self.product = bytes(_buf[:_ret]).decode("ascii", errors="replace")
    self.is_custom = self.product.startswith("custom")
    if self.is_custom: self.use_bot = use_bot = True

    # Detach kernel driver if needed
    if checked(libusb.libusb_kernel_driver_active)(self.handle, 0):
      checked(libusb.libusb_detach_kernel_driver)(self.handle, 0)
      checked(libusb.libusb_reset_device)(self.handle)

    # Set configuration and claim interface
    checked(libusb.libusb_set_configuration)(self.handle, 1)
    checked(libusb.libusb_claim_interface)(self.handle, 0)

    if use_bot:
      checked(libusb.libusb_set_interface_alt_setting)(self.handle, 0, 0)
      self._tag = 0
    else:
      checked(libusb.libusb_set_interface_alt_setting)(self.handle, 0, 1)

      # Clear any stalled endpoints
      all_eps = (self.ep_data_out, self.ep_data_in, self.ep_stat_in, self.ep_cmd_out)
      for ep in all_eps: checked(libusb.libusb_clear_halt)(self.handle, ep)

      # Allocate streams
      stream_eps = (ctypes.c_uint8 * 3)(self.ep_data_out, self.ep_data_in, self.ep_stat_in)
      checked(libusb.libusb_alloc_streams)(self.handle, self.max_streams * len(stream_eps), stream_eps, len(stream_eps))

      # Base cmd
      cmd_template = bytes([0x01, 0x00, 0x00, 0x01, *([0] * 12), 0xE4, 0x24, 0x00, 0xB2, 0x1A, 0x00, 0x00, 0x00, *([0] * 8)])

      # Init pools
      self.tr = {ep: [libusb.libusb_alloc_transfer(0) for _ in range(self.max_streams)] for ep in all_eps}

      self.buf_cmd = [(ctypes.c_uint8 * len(cmd_template))(*cmd_template) for _ in range(self.max_streams)]
      self.buf_stat = [(ctypes.c_uint8 * 64)() for _ in range(self.max_streams)]
      self.buf_data_in = [(ctypes.c_uint8 * 0x1000)() for _ in range(self.max_streams)]
      self.buf_data_out = [(ctypes.c_uint8 * 0x80000)() for _ in range(self.max_streams)]
      self.buf_data_out_mvs = [to_mv(ctypes.addressof(self.buf_data_out[i]), 0x80000) for i in range(self.max_streams)]

      for slot in range(self.max_streams): struct.pack_into(">B", self.buf_cmd[slot], 3, slot + 1)

  def _prep_transfer(self, tr, ep, stream_id, buf, length):
    tr.contents.dev_handle, tr.contents.endpoint, tr.contents.length, tr.contents.buffer = self.handle, ep, length, buf
    tr.contents.status, tr.contents.flags, tr.contents.timeout, tr.contents.num_iso_packets = 0xff, 0, 1000, 0
    tr.contents.type = (libusb.LIBUSB_TRANSFER_TYPE_BULK_STREAM if stream_id is not None else libusb.LIBUSB_TRANSFER_TYPE_BULK)
    if stream_id is not None: libusb.libusb_transfer_set_stream_id(tr, stream_id)
    return tr

  def _submit_and_wait(self, cmds):
    for tr in cmds: checked(libusb.libusb_submit_transfer)(tr)

    running = len(cmds)
    while running:
      checked(libusb.libusb_handle_events)(USB3.ctx())
      running = len(cmds)
      for tr in cmds:
        if tr.contents.status == libusb.LIBUSB_TRANSFER_COMPLETED: running -= 1
        elif tr.contents.status != 0xFF: raise RuntimeError(f"EP 0x{tr.contents.endpoint:02X} error: {tr.contents.status}")

  def _bulk_out(self, ep: int, payload: bytes, timeout: int = 1000):
    if len(payload) > len(self._bulk_out_mv): self._bulk_out_buf, self._bulk_out_mv = alloc_cbuffer(len(payload))
    self._bulk_out_mv[:len(payload)] = payload
    checked(libusb.libusb_bulk_transfer, f"bulk OUT 0x{ep:02X} failed")(self.handle, ep, self._bulk_out_buf, len(payload), self._transferred, timeout)
    assert self._transferred.value == len(payload), f"bulk OUT short write on 0x{ep:02X}: {self._transferred.value}/{len(payload)} bytes"

  def _bulk_in(self, ep: int, length: int, timeout: int = 1000) -> memoryview:
    if length > len(self._bulk_in_mv): self._bulk_in_buf, self._bulk_in_mv = alloc_cbuffer(length)
    checked(libusb.libusb_bulk_transfer, f"bulk IN 0x{ep:02X} failed")(self.handle, ep, self._bulk_in_buf, length, self._transferred, timeout)
    return self._bulk_in_mv[:self._transferred.value]

  def send_batch(self, cdbs:list[bytes], idata:list[int]|None=None, odata:list[bytes|None]|None=None) -> list[bytes|None]:
    idata, odata = idata or [0] * len(cdbs), odata or [None] * len(cdbs)
    results:list[bytes|None] = []
    tr_window, op_window = [], []

    for idx, (cdb, rlen, send_data) in enumerate(zip(cdbs, idata, odata)):
      if self.use_bot:
        dir_in = rlen > 0
        data_len = rlen if dir_in else (len(send_data) if send_data is not None else 0)
        assert not (rlen > 0 and send_data is not None), "BOT mode only supports either read or write per command"

        # CBW
        self._tag += 1
        flags = 0x80 if dir_in else 0x00
        cbw = struct.pack("<IIIBBB", 0x43425355, self._tag, data_len, flags, 0, len(cdb)) + cdb + b"\x00" * (16 - len(cdb))
        self._bulk_out(self.ep_data_out, cbw)

        # DAT
        if dir_in:
          results.append(bytes(self._bulk_in(self.ep_data_in, rlen)))
        else:
          if send_data is not None:
            self._bulk_out(self.ep_data_out, send_data)
          results.append(None)

        # CSW
        sig, rtag, residue, status = struct.unpack("<IIIB", self._bulk_in(self.ep_data_in, 13, timeout=2000))
        assert sig == 0x53425355, f"Bad CSW signature 0x{sig:08X}, expected 0x53425355"
        assert rtag == self._tag, f"CSW tag mismatch: got {rtag}, expected {self._tag}"
        assert status == 0, f"SCSI command failed, CSW status=0x{status:02X}, residue={residue}"
      else:
        # allocate slot and stream. stream is 1-based
        slot, stream = idx % self.max_streams, (idx % self.max_streams) + 1

        # build cmd packet
        self.buf_cmd[slot][16:16+len(cdb)] = list(cdb)

        # cmd + stat transfers
        tr_window.append(self._prep_transfer(self.tr[self.ep_cmd_out][slot], self.ep_cmd_out, None, self.buf_cmd[slot], len(self.buf_cmd[slot])))
        tr_window.append(self._prep_transfer(self.tr[self.ep_stat_in][slot], self.ep_stat_in, stream, self.buf_stat[slot], 64))

        if rlen:
          if rlen > len(self.buf_data_in[slot]): self.buf_data_in[slot] = (ctypes.c_uint8 * round_up(rlen, 0x1000))()
          tr_window.append(self._prep_transfer(self.tr[self.ep_data_in][slot], self.ep_data_in, stream, self.buf_data_in[slot], rlen))

        if send_data is not None:
          if len(send_data) > len(self.buf_data_out[slot]):
            self.buf_data_out[slot] = (ctypes.c_uint8 * len(send_data))()
            self.buf_data_out_mvs[slot] = to_mv(ctypes.addressof(self.buf_data_out[slot]), len(send_data))

          self.buf_data_out_mvs[slot][:len(send_data)] = bytes(send_data)
          tr_window.append(self._prep_transfer(self.tr[self.ep_data_out][slot], self.ep_data_out, stream, self.buf_data_out[slot], len(send_data)))

        op_window.append((idx, slot, rlen))
        if (idx + 1 == len(cdbs)) or len(op_window) >= self.max_streams:
          self._submit_and_wait(tr_window)
          for idx, slot, rlen in op_window: results.append(bytes(self.buf_data_in[slot][:rlen]) if rlen else None)
          tr_window = []

    return results

@dataclasses.dataclass(frozen=True)
class WriteOp: addr:int; data:bytes; ignore_cache:bool=True # noqa: E702

@dataclasses.dataclass(frozen=True)
class ReadOp: addr:int; size:int # noqa: E702

@dataclasses.dataclass(frozen=True)
class ScsiWriteOp: data:bytes; lba:int=0 # noqa: E702

class CustomASM24Controller:
  def __init__(self, usb:USB3|None=None):
    if not usb:
      devs = USB3.list_devices(0xADD1, 0x0001)
      assert len(devs), "no ASM24 controller found"
      self.usb = USB3(devs[0][0], 0x81, 0x83, 0x02, 0x04, use_bot=True)
    else: self.usb = usb
    self._pci_cacheable: list[tuple[int, int]] = []
    self._pci_cache: dict[int, int|None] = {}

    self._f0_out_buf, self._f0_out_mv = alloc_cbuffer(0x1000) # for f0 and e4, allocate big enough for e4
    self._f0_in_buf, _ = alloc_cbuffer(8)

    # Custom firmware now boots with PCIe off. Power it on before probing the link.
    ltssm = self.read(0xB450, 1)[0]
    if ltssm != 0x78: self.set_pcie_power(True)
    ltssm = self.read(0xB450, 1)[0]
    if ltssm != 0x78: raise RuntimeError(f"PCIe link not up (LTSSM=0x{ltssm:02X}), custom firmware not ready")

  def set_pcie_power(self, enabled:bool, timeout:int=10000):
    checked(libusb.libusb_control_transfer,
            f"F3 PCIe power {'on' if enabled else 'off'} failed")(self.usb.handle, 0x40, 0xF3, int(enabled), 0, None, 0, timeout)

  # === PCIe TLP via 0xF0 vendor command ===

  def _f0_out(self, fmt_type:int, byte_en:int, address:int, value:int, mode:int=0):
    struct.pack_into('<III', self._f0_out_mv, 0, address & 0xFFFFFFFF, address >> 32, value)
    ret = libusb.libusb_control_transfer(self.usb.handle, 0x40, 0xF0, fmt_type | (byte_en << 8), mode & 0x03, self._f0_out_buf, 12, 5000)
    assert ret == 12, f"F0 OUT failed: {ret}"

  def _f0_in(self) -> tuple[int, int, int]:
    ret = libusb.libusb_control_transfer(self.usb.handle, 0xC0, 0xF0, 0, 0, self._f0_in_buf, 8, 5000)
    assert ret == 8, f"F0 IN failed: {ret}"
    return struct.unpack_from('<I', self._f0_in_buf, 0)[0], (self._f0_in_buf[4] >> 5) & 0x7, self._f0_in_buf[7]

  def _is_pci_cacheable(self, addr:int) -> bool: return any(x <= addr <= x + sz for x, sz in self._pci_cacheable)

  def pcie_request(self, fmt_type:int, address:int, value:int|None=None, size:int=4, cnt:int=10):
    if fmt_type == 0x60 and size == 4 and self._is_pci_cacheable(address) and self._pci_cache.get(address) == value: return
    assert size > 0 and size <= 4, f"Invalid size {size}"
    if DEBUG >= 5: print("pcie_request", hex(fmt_type), hex(address), value, size)

    offset = address & 0x3
    byte_en = ((1 << size) - 1) << offset
    self._pci_cache[address] = value if size == 4 and fmt_type == 0x60 else None

    self._f0_out(fmt_type, byte_en, address & ~0x3, (value << (8 * offset)) if value is not None else 0)

    # Fast path: memory writes and messages don't return completions (same logic as ASM24Controller).
    if ((fmt_type & 0b11011111) == 0b01000000) or ((fmt_type & 0b10111000) == 0b00110000): return

    # Read TLPs and config writes: read completion via 0xF0 IN. Retry on error/timeout.
    data, cpl_status, ret_status = self._f0_in()
    if ret_status != 0:
      time.sleep(0.001)  # TODO: this sleep is very picky
      if cnt > 0:
        return self.pcie_request(fmt_type, address, value, size, cnt=cnt-1)
      raise RuntimeError(f"TLP error after retries: ret_status={ret_status}, address={address:#x}")

    if cpl_status:
      status_map = {0b001: f"Unsupported Request: {address:#x}", 0b100: "Completer Abort", 0b010: "Config Retry"}
      raise RuntimeError(f"TLP completion status: {status_map.get(cpl_status, f'Reserved (0b{cpl_status:03b})')}")

    if value is None: return (data >> (8 * offset)) & ((1 << (8 * size)) - 1)

  def pcie_cfg_req(self, byte_addr:int, bus:int=1, dev:int=0, fn:int=0, value:int|None=None, size:int=4):
    assert byte_addr >> 12 == 0 and bus >> 8 == 0 and dev >> 5 == 0 and fn >> 3 == 0
    fmt_type = (0x44 if value is not None else 0x4) | int(bus > 0)
    address = (bus << 24) | (dev << 19) | (fn << 16) | (byte_addr & 0xfff)
    return self.pcie_request(fmt_type, address, value, size)

  def pcie_mem_req(self, address:int, value:int|None=None, size:int=4):
    return self.pcie_request(0x60 if value is not None else 0x20, address, value, size)

  def pcie_mem_write(self, address:int, values:list[int], size:int):
    """Streaming PCIe memory write via 0xF0 mode 1 + bulk OUT. Data is little-endian dwords on the wire."""
    if not values: return
    self._f0_out(0x60, 0x0F, address, len(values), mode=1)
    self.usb._bulk_out(0x02, struct.pack(f'<{len(values)}I', *values))

  def pcie_mem_read(self, address:int, nbytes:int) -> bytes:
    """Streaming PCIe memory read via 0xF0 mode 2 + bulk IN. Returns little-endian bytes."""
    assert nbytes % 4 == 0, f"pcie_mem_read requires 4-byte aligned size, got {nbytes}"
    self._f0_out(0x20, 0x0F, address, nbytes // 4, mode=2)
    return self.usb._bulk_in(0x81, nbytes, timeout=30000)

  # === XDATA read/write (0xE4/0xE5 vendor control transfers) ===

  def read(self, base_addr:int, length:int, **kwargs) -> bytes:
    """Read from chip XDATA via vendor control IN (bRequest=0xE4). wValue=addr, wLength=size."""
    result = b''
    for off in range(0, length, 0xFF):
      chunk = min(0xFF, length - off)
      ret = libusb.libusb_control_transfer(self.usb.handle, 0xC0, 0xE4, base_addr + off, 0, self._f0_out_buf, chunk, 1000)
      assert ret == chunk, f"read(0x{base_addr + off:04X}, {chunk}) failed: {ret}"
      result += bytes(self._f0_out_buf[:ret])
    return result[:length]

  def write(self, base_addr:int, data:bytes, **kwargs):
    """Write to chip XDATA via vendor control OUT (bRequest=0xE5). wValue=addr, wIndex=val."""
    for off, val in enumerate(data):
      checked(libusb.libusb_control_transfer,
              f"write(0x{base_addr + off:04X}, 0x{val:02X}) failed")(self.usb.handle, 0x40, 0xE5, base_addr + off, val, None, 0, 1000)

  def scsi_write(self, buf:bytes, lba:int=0):
    """Write to SRAM via 0xF2 vendor command + bulk OUT."""
    buf_padded = buf + b'\x00' * (round_up(len(buf), 512) - len(buf))
    sectors = len(buf_padded) // 512
    num_slots = round_up(len(buf_padded), 0x4000) // 0x4000  # 16KB per slot
    # 0xF2 OUT: wValue=sectors, wIndex=start_slot|(num_slots<<8)
    windex = (num_slots & 0xFF) << 8
    checked(libusb.libusb_control_transfer, "F2 setup failed")(self.usb.handle, 0x40, 0xF2, sectors, windex, None, 0, 1000)
    self.usb._bulk_out(0x02, buf_padded)

  def scsi_read_arm(self, size:int):
    windex = (ceildiv(size, 0x4000) & 0xFF) << 8
    checked(libusb.libusb_control_transfer,
            "F2 read arm failed")(self.usb.handle, 0x40, 0xF2, (ceildiv(size, 512) & 0x7FFF) | 0x8000, windex, None, 0, 1000)

  def scsi_read(self, size:int) -> memoryview: return self.usb._bulk_in(0x81, round_up(size, 512), timeout=10000)[:size]

class ASM24Controller:
  def __init__(self, usb:USB3|None=None):
    if not usb:
      devs = USB3.list_devices(0xADD1, 0x0001)
      assert len(devs), "no ASM24 controller found"
      self.usb = USB3(devs[0][0], 0x81, 0x83, 0x02, 0x04, use_bot=bool(getenv("USE_BOT", 0)))
    else: self.usb = usb
    self._cache: dict[int, int|None] = {}
    self._pci_cacheable: list[tuple[int, int]] = []
    self._pci_cache: dict[int, int|None] = {}

    # Init controller.
    self.exec_ops([WriteOp(0x54b, b' '), WriteOp(0x54e, b'\x04'), WriteOp(0x5a8, b'\x02'), WriteOp(0x5f8, b'\x04'),
      WriteOp(0x7ec, b'\x01\x00\x00\x00'), WriteOp(0xc422, b'\x02'), WriteOp(0x0, b'\x33')])

  def exec_ops(self, ops:Sequence[WriteOp|ReadOp|ScsiWriteOp]):
    cdbs:list[bytes] = []
    idata:list[int] = []
    odata:list[bytes|None] = []

    def _add_req(cdb:bytes, i:int, o:bytes|None):
      nonlocal cdbs, idata, odata
      cdbs, idata, odata = cdbs + [cdb], idata + [i], odata + [o]

    for op in ops:
      if isinstance(op, WriteOp):
        for off, value in enumerate(op.data):
          addr = ((op.addr + off) & 0x1FFFF) | 0x500000
          if not op.ignore_cache and self._cache.get(addr) == value: continue
          _add_req(struct.pack('>BBBHB', 0xE5, value, addr >> 16, addr & 0xFFFF, 0), 0, None)
          self._cache[addr] = value
      elif isinstance(op, ReadOp):
        assert op.size <= 0xff
        addr = (op.addr & 0x1FFFF) | 0x500000
        _add_req(struct.pack('>BBBHB', 0xE4, op.size, addr >> 16, addr & 0xFFFF, 0), op.size, None)
        for i in range(op.size): self._cache[addr + i] = None
      elif isinstance(op, ScsiWriteOp):
        sectors = round_up(len(op.data), 512) // 512
        _add_req(struct.pack('>BBQIBB', 0x8A, 0, op.lba, sectors, 0, 0), 0, op.data+b'\x00'*((sectors*512)-len(op.data)))

    return self.usb.send_batch(cdbs, idata, odata)

  def write(self, base_addr:int, data:bytes, ignore_cache:bool=True): return self.exec_ops([WriteOp(base_addr, data, ignore_cache)])

  def scsi_write(self, buf:bytes, lba:int=0):
    if len(buf) > 0x4000: buf += b'\x00' * (round_up(len(buf), 0x10000) - len(buf))

    for i in range(0, len(buf), 0x10000):
      self.exec_ops([ScsiWriteOp(buf[i:i+0x10000], lba), WriteOp(0x171, b'\xff\xff\xff', ignore_cache=True)])
      self.exec_ops([WriteOp(0xce6e, b'\x00\x00', ignore_cache=True)])

    if len(buf) > 0x4000:
      for i in range(4): self.exec_ops([WriteOp(0xce40 + i, b'\x00', ignore_cache=True)])

  def read(self, base_addr:int, length:int, stride:int=0xff) -> bytes:
    parts = self.exec_ops([ReadOp(base_addr + off, min(stride, length - off)) for off in range(0, length, stride)])
    return b''.join(p or b'' for p in parts)[:length]

  def _is_pci_cacheable(self, addr:int) -> bool: return any(x <= addr <= x + sz for x, sz in self._pci_cacheable)
  def pcie_prep_request(self, fmt_type:int, address:int, value:int|None=None, size:int=4) -> list[WriteOp]:
    if fmt_type == 0x60 and size == 4 and self._is_pci_cacheable(address) and self._pci_cache.get(address) == value: return []

    assert fmt_type >> 8 == 0 and size > 0 and size <= 4, f"Invalid fmt_type {fmt_type} or size {size}"
    if DEBUG >= 5: print("pcie_request", hex(fmt_type), hex(address), value, size)

    masked_address, offset = address & 0xFFFFFFFC, address & 0x3
    assert size + offset <= 4 and (value is None or value >> (8 * size) == 0)
    self._pci_cache[address] = value if size == 4 and fmt_type == 0x60 else None

    return ([WriteOp(0xB220, struct.pack('>I', value << (8 * offset)), ignore_cache=False)] if value is not None else []) + \
      [WriteOp(0xB218, struct.pack('>I', masked_address), ignore_cache=False), WriteOp(0xB21c, struct.pack('>I', address>>32), ignore_cache=False),
       WriteOp(0xB217, bytes([((1 << size) - 1) << offset]), ignore_cache=False), WriteOp(0xB210, bytes([fmt_type]), ignore_cache=False),
       WriteOp(0xB254, b"\x0f", ignore_cache=True), WriteOp(0xB296, b"\x04", ignore_cache=True)]

  def pcie_request(self, fmt_type, address, value=None, size=4, cnt=10):
    self.exec_ops(self.pcie_prep_request(fmt_type, address, value, size))

    # Fast path for write requests
    if ((fmt_type & 0b11011111) == 0b01000000) or ((fmt_type & 0b10111000) == 0b00110000): return

    while (stat:=self.read(0xB296, 1)[0]) & 2 == 0:
      if stat & 1:
        self.write(0xB296, bytes([0x01]))
        if cnt > 0: return self.pcie_request(fmt_type, address, value, size, cnt=cnt-1)
    assert stat == 2, f"stat read 2 was {stat}"

    # Retrieve completion data from Link Status (0xB22A, 0xB22B)
    b284 = self.read(0xB284, 1)[0]
    completion = struct.unpack('>H', self.read(0xB22A, 2))

    # Validate completion status based on PCIe request typ
    # Completion TLPs for configuration requests always have a byte count of 4.
    assert completion[0] & 0xfff == (4 if (fmt_type & 0xbe == 0x04) else size)

    # Extract completion status field
    status = (completion[0] >> 13) & 0x7

    # Handle completion errors or inconsistencies
    if status or ((fmt_type & 0xbe == 0x04) and (((value is None) and (not (b284 & 0x01))) or ((value is not None) and (b284 & 0x01)))):
      status_map = {0b001: f"Unsupported Request: invalid address/function (target might not be reachable): {address:#x}",
                    0b100: "Completer Abort: abort due to internal error", 0b010: "Configuration Request Retry Status: configuration space busy"}
      raise RuntimeError(f"TLP status: {status_map.get(status, 'Reserved (0b{:03b})'.format(status))}")

    if value is None: return (struct.unpack('>I', self.read(0xB220, 4))[0] >> (8 * (address & 0x3))) & ((1 << (8 * size)) - 1)

  def pcie_cfg_req(self, byte_addr, bus=1, dev=0, fn=0, value=None, size=4):
    assert byte_addr >> 12 == 0 and bus >> 8 == 0 and dev >> 5 == 0 and fn >> 3 == 0, f"Invalid byte_addr {byte_addr}, bus {bus}, dev {dev}, fn {fn}"

    fmt_type = (0x44 if value is not None else 0x4) | int(bus > 0)
    address = (bus << 24) | (dev << 19) | (fn << 16) | (byte_addr & 0xfff)
    return self.pcie_request(fmt_type, address, value, size)

  def pcie_mem_req(self, address, value=None, size=4): return self.pcie_request(0x60 if value is not None else 0x20, address, value, size)

  def pcie_mem_write(self, address, values, size):
    ops = [self.pcie_prep_request(0x60, address + i * size, value, size) for i, value in enumerate(values)]

    # Send in batches of 4 for OSX and 16 for Linux (benchmarked values)
    for i in range(0, len(ops), bs:=(4 if OSX else 16)): self.exec_ops(list(itertools.chain.from_iterable(ops[i:i+bs])))

class USBMMIOInterface(MMIOInterface):
  def __init__(self, usb, addr, size, fmt, pcimem=True): # pylint: disable=super-init-not-called
    self.usb, self.addr, self.nbytes, self.fmt, self.pcimem, self.el_sz = usb, addr, size, fmt, pcimem, struct.calcsize(fmt)

  def __getitem__(self, index): return self._access_items(index)
  def __setitem__(self, index, val): self._access_items(index, val)

  def _access_items(self, index, val=None):
    if isinstance(index, slice): return self._acc((index.start or 0) * self.el_sz, ((index.stop or len(self))-(index.start or 0)) * self.el_sz, val)
    return self._acc_one(index * self.el_sz, self.el_sz, val) if self.pcimem else self._acc(index * self.el_sz, self.el_sz, val)

  def view(self, offset:int=0, size:int|None=None, fmt=None):
    return USBMMIOInterface(self.usb, self.addr+offset, size or (self.nbytes - offset), fmt=fmt or self.fmt, pcimem=self.pcimem)

  def _acc_size(self, sz): return next(x for x in [('I', 4), ('H', 2), ('B', 1)] if sz % x[1] == 0)

  def _acc_one(self, off, sz, val=None):
    upper = 0 if sz < 8 else self.usb.pcie_mem_req(self.addr + off + 4, val if val is None else (val >> 32), 4)
    lower = self.usb.pcie_mem_req(self.addr + off, val if val is None else val & 0xffffffff, min(sz, 4))
    if val is None: return lower | (upper << 32)

  def _acc(self, off, sz, data=None):
    if data is None: # read op
      if not self.pcimem:
        if self.addr == 0xf000 and hasattr(self.usb, 'scsi_read'): return self.usb.scsi_read(sz)
        return int.from_bytes(self.usb.read(self.addr + off, sz), "little") if sz == self.el_sz else self.usb.read(self.addr + off, sz)

      # Fast path: streaming PCIe read if controller supports it
      if hasattr(self.usb, 'pcie_mem_read') and sz >= 4 and sz % 4 == 0:
        return self.usb.pcie_mem_read(self.addr + off, sz)

      acc, acc_size = self._acc_size(sz)
      return bytes(array.array(acc, [self._acc_one(off + i * acc_size, acc_size) for i in range(sz // acc_size)]))

    # write op
    data = struct.pack(self.fmt, data) if isinstance(data, int) else bytes(data)

    if not self.pcimem:
      # Fast path for writing into buffer 0xf000
      use_cache = 0xa800 <= self.addr <= 0xb000
      return self.usb.scsi_write(bytes(data)) if self.addr == 0xf000 else self.usb.write(self.addr + off, bytes(data), ignore_cache=not use_cache)

    _, acc_sz = self._acc_size(len(data) * struct.calcsize(self.fmt))
    self.usb.pcie_mem_write(self.addr+off, [int.from_bytes(data[i:i+acc_sz], "little") for i in range(0, len(data), acc_sz)], acc_sz)

if DEV.interface.startswith("MOCK"): from test.mockgpu.usb import MockUSB3 as USB3  # type: ignore  # noqa: F811
