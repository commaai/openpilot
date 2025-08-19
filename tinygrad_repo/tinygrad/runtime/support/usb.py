import ctypes, struct, dataclasses, array, itertools
from typing import Sequence
from tinygrad.runtime.autogen import libusb
from tinygrad.helpers import DEBUG, to_mv, round_up, OSX
from tinygrad.runtime.support.hcq import MMIOInterface

class USB3:
  def __init__(self, vendor:int, dev:int, ep_data_in:int, ep_stat_in:int, ep_data_out:int, ep_cmd_out:int, max_streams:int=31):
    self.vendor, self.dev = vendor, dev
    self.ep_data_in, self.ep_stat_in, self.ep_data_out, self.ep_cmd_out = ep_data_in, ep_stat_in, ep_data_out, ep_cmd_out
    self.max_streams = max_streams
    self.ctx = ctypes.POINTER(libusb.struct_libusb_context)()

    if libusb.libusb_init(ctypes.byref(self.ctx)): raise RuntimeError("libusb_init failed")
    if DEBUG >= 6: libusb.libusb_set_option(self.ctx, libusb.LIBUSB_OPTION_LOG_LEVEL, 4)

    self.handle = libusb.libusb_open_device_with_vid_pid(self.ctx, self.vendor, self.dev)
    if not self.handle: raise RuntimeError(f"device {self.vendor:04x}:{self.dev:04x} not found. sudo required?")

    # Detach kernel driver if needed
    if libusb.libusb_kernel_driver_active(self.handle, 0):
      libusb.libusb_detach_kernel_driver(self.handle, 0)
      libusb.libusb_reset_device(self.handle)

    # Set configuration and claim interface
    if libusb.libusb_set_configuration(self.handle, 1): raise RuntimeError("set_configuration failed")
    if libusb.libusb_claim_interface(self.handle, 0): raise RuntimeError("claim_interface failed. sudo required?")
    if libusb.libusb_set_interface_alt_setting(self.handle, 0, 1): raise RuntimeError("alt_setting failed")

    # Clear any stalled endpoints
    all_eps = (self.ep_data_out, self.ep_data_in, self.ep_stat_in, self.ep_cmd_out)
    for ep in all_eps: libusb.libusb_clear_halt(self.handle, ep)

    # Allocate streams
    stream_eps = (ctypes.c_uint8 * 3)(self.ep_data_out, self.ep_data_in, self.ep_stat_in)
    if (rc:=libusb.libusb_alloc_streams(self.handle, self.max_streams * len(stream_eps), stream_eps, len(stream_eps))) < 0:
      raise RuntimeError(f"alloc_streams failed: {rc}")

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
    for tr in cmds: libusb.libusb_submit_transfer(tr)

    running = len(cmds)
    while running:
      libusb.libusb_handle_events(self.ctx)
      running = len(cmds)
      for tr in cmds:
        if tr.contents.status == libusb.LIBUSB_TRANSFER_COMPLETED: running -= 1
        elif tr.contents.status != 0xFF: raise RuntimeError(f"EP 0x{tr.contents.endpoint:02X} error: {tr.contents.status}")

  def send_batch(self, cdbs:list[bytes], idata:list[int]|None=None, odata:list[bytes|None]|None=None) -> list[bytes|None]:
    idata, odata = idata or [0] * len(cdbs), odata or [None] * len(cdbs)
    results, tr_window, op_window = [], [], []

    for idx, (cdb, rlen, send_data) in enumerate(zip(cdbs, idata, odata)):
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

class ASM24Controller:
  def __init__(self):
    self.usb = USB3(0xADD1, 0x0001, 0x81, 0x83, 0x02, 0x04)
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
  def __init__(self, usb, addr, size, fmt, pcimem=True):
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
        return int.from_bytes(self.usb.read(self.addr + off, sz), "little") if sz == self.el_sz else self.usb.read(self.addr + off, sz)

      acc, acc_size = self._acc_size(sz)
      return bytes(array.array(acc, [self._acc_one(off + i * acc_size, acc_size) for i in range(sz // acc_size)]))
    else: # write op
      data = struct.pack(self.fmt, data) if isinstance(data, int) else bytes(data)

      if not self.pcimem:
        # Fast path for writing into buffer 0xf000
        use_cache = 0xa800 <= self.addr <= 0xb000
        return self.usb.scsi_write(bytes(data)) if self.addr == 0xf000 else self.usb.write(self.addr + off, bytes(data), ignore_cache=not use_cache)

      _, acc_sz = self._acc_size(len(data) * struct.calcsize(self.fmt))
      self.usb.pcie_mem_write(self.addr+off, [int.from_bytes(data[i:i+acc_sz], "little") for i in range(0, len(data), acc_sz)], acc_sz)
