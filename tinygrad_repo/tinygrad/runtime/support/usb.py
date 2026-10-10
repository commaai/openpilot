import ctypes, struct, time, functools, itertools
from tinygrad.runtime.autogen import libusb, libc
from tinygrad.helpers import DEBUG, DEV, HCQ_RUNTIME_DEV, to_mv, round_up, ceildiv, flatten
from tinygrad.dtype import dtypes, DType, AddrSpace
from tinygrad.uop.ops import UOp, UPat, Ops, PatternMatcher, uopfunc
from tinygrad.device import Buffer, BufferSpec, Compiled
from tinygrad.runtime.support.hcq2 import HCQ_DEVS, CDTYPE, ccall, patch, unwrap_view, all_devices_in, to_name, get_time_ms, ins, chunks
from tinygrad.runtime.support.memory import MMIOInterface
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

  def __init__(self, dev:c.POINTER[libusb.struct_libusb_device], *args, **kwargs):
    self._tags, self._transferred = itertools.count(1), ctypes.c_int(0)
    self._bulk_buf, self._bulk_mv = alloc_cbuffer(4 << 20)
    self._ctrl_buf, self._ctrl_mv = alloc_cbuffer(0x1000)

    self.handle = c.init_c_var(c.POINTER[libusb.struct_libusb_device_handle], lambda x: checked(libusb.libusb_open)(dev, x))

    # Read product string descriptor
    _buf = (ctypes.c_ubyte * 256)()
    _desc = libusb.struct_libusb_device_descriptor()
    checked(libusb.libusb_get_device_descriptor)(libusb.libusb_get_device(self.handle), ctypes.byref(_desc))
    _ret = checked(libusb.libusb_get_string_descriptor_ascii)(self.handle, _desc.iProduct, _buf, 256)
    self.product = bytes(_buf[:_ret]).decode("ascii", errors="replace")
    assert self.product.startswith("custom") or self.product.startswith("AS2462")

    # Detach kernel driver if needed
    if checked(libusb.libusb_kernel_driver_active)(self.handle, 0):
      checked(libusb.libusb_detach_kernel_driver)(self.handle, 0)
      checked(libusb.libusb_reset_device)(self.handle)

    # Set configuration and claim interface
    checked(libusb.libusb_set_configuration)(self.handle, 1)
    checked(libusb.libusb_claim_interface)(self.handle, 0)
    checked(libusb.libusb_set_interface_alt_setting)(self.handle, 0, 0)

  def control_write(self, request:int, value:int=0, index:int=0, data:bytes=b'', timeout:int=1000):
    assert len(data) <= len(self._ctrl_mv)
    self._ctrl_mv[:len(data)] = data
    assert checked(libusb.libusb_control_transfer)(self.handle, 0x40, request, value, index, self._ctrl_buf, len(data), timeout) == len(data)

  def control_read(self, request:int, length:int, value:int=0, index:int=0, timeout:int=1000) -> memoryview:
    assert length <= len(self._ctrl_mv)
    assert checked(libusb.libusb_control_transfer)(self.handle, 0xC0, request, value, index, self._ctrl_buf, length, timeout) == length
    return self._ctrl_mv[:length]

  def bulk_write(self, payload:bytes, timeout:int=1000):
    if len(payload) > len(self._bulk_mv): self._bulk_buf, self._bulk_mv = alloc_cbuffer(len(payload))
    self._bulk_mv[:len(payload)] = payload
    checked(libusb.libusb_bulk_transfer, "bulk OUT 0x02 failed") \
      (self.handle, 0x02, self._bulk_buf, len(payload), self._transferred, timeout)
    assert self._transferred.value == len(payload), f"bulk OUT short write: {self._transferred.value}/{len(payload)} bytes"

  def bulk_read(self, length:int, timeout:int=1000) -> memoryview:
    if length > len(self._bulk_mv): self._bulk_buf, self._bulk_mv = alloc_cbuffer(length)
    checked(libusb.libusb_bulk_transfer, "bulk IN 0x81 failed")(self.handle, 0x81, self._bulk_buf, length, self._transferred, timeout)
    if self._transferred.value != length: raise RuntimeError(f"bulk IN short read: {self._transferred.value}/{length} bytes")
    return self._bulk_mv[:length]

  # NOTE: keep it for flash.py
  def send_batch(self, cdbs:list[bytes], odata:list[bytes|None]|None=None):
    for cdb, data in zip(cdbs, odata or [None] * len(cdbs)):
      self.bulk_write(struct.pack("<IIIBBB16s", 0x43425355, tag:=next(self._tags), len(data) if data is not None else 0, 0, 0, len(cdb), cdb))
      if data is not None: self.bulk_write(data)
      sig, rtag, _, status = struct.unpack("<IIIB", self.bulk_read(13, timeout=2000))
      assert (sig, rtag, status) == (0x53425355, tag, 0)

class CustomASM24Controller:
  def __init__(self, usb:USB3):
    self.usb = usb

    # Custom firmware now boots with PCIe off. Power it on before probing the link.
    ltssm = self.read(0xB450, 1)[0]
    if ltssm != 0x78: self.set_pcie_power(True)
    ltssm = self.read(0xB450, 1)[0]
    if ltssm != 0x78: raise RuntimeError(f"PCIe link not up (LTSSM=0x{ltssm:02X}), custom firmware not ready")

  def set_pcie_power(self, enabled:bool, timeout:int=10000): self.usb.control_write(0xF3, value=int(enabled), timeout=timeout)

  def _f0_out(self, fmt_type:int, byte_en:int, address:int, value:int, mode:int=0):
    self.usb.control_write(0xF0, fmt_type | (byte_en << 8), mode & 0x03, struct.pack('<III', address & 0xFFFFFFFF, address >> 32, value), 5000)

  def _f0_in(self) -> tuple[int, int, int]:
    data = self.usb.control_read(0xF0, 8, timeout=5000)
    return struct.unpack_from('<I', data)[0], (data[4] >> 5) & 0x7, data[7]

  def pcie_request(self, fmt_type:int, address:int, value:int|None=None, size:int=4, cnt:int=10):
    assert size > 0 and size <= 4, f"Invalid size {size}"
    if DEBUG >= 5: print("pcie_request", hex(fmt_type), hex(address), value, size)

    offset = address & 0x3
    byte_en = ((1 << size) - 1) << offset
    self._f0_out(fmt_type, byte_en, address & ~0x3, (value << (8 * offset)) if value is not None else 0)

    # Fast path: memory writes and messages don't return completions.
    if ((fmt_type & 0b11011111) == 0b01000000) or ((fmt_type & 0b10111000) == 0b00110000): return

    # Read TLPs and config writes: read completion via 0xF0 IN. Retry on error/timeout.
    data, cpl_status, ret_status = self._f0_in()
    if ret_status != 0:
      time.sleep(0.001)  # TODO: this sleep is very picky
      if cnt > 0: return self.pcie_request(fmt_type, address, value, size, cnt=cnt-1)
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

  def pcie_mem_write(self, address:int, data:bytes):
    """Streaming PCIe memory write via 0xF0 mode 1 + bulk OUT. Data is little-endian dwords on the wire."""
    if not data: return
    assert len(data) % 4 == 0, f"pcie_mem_write requires 4-byte aligned size, got {len(data)}"
    for off in range(0, len(data), STREAM):
      chunk = data[off:off+STREAM]
      self._f0_out(0x60, 0x0F, address + off, len(chunk) // 4, mode=1)
      self.usb.bulk_write(chunk)

  def pcie_mem_read(self, address:int, nbytes:int) -> memoryview:
    """Streaming PCIe memory read via 0xF0 mode 2 + bulk IN. Returns little-endian bytes."""
    assert nbytes % 4 == 0, f"pcie_mem_read requires 4-byte aligned size, got {nbytes}"
    self._f0_out(0x20, 0x0F, address, nbytes // 4, mode=2)
    return self.usb.bulk_read(nbytes, timeout=30000)

  def read(self, base_addr:int, length:int) -> bytes:
    """Read from chip XDATA via vendor control IN (bRequest=0xE4). wValue=addr, wLength=size."""
    result = b''
    for off in range(0, length, 0xFF):
      chunk = min(0xFF, length - off)
      result += self.usb.control_read(0xE4, chunk, value=base_addr + off)
    return result

  def write(self, base_addr:int, data:bytes):
    """Write to chip XDATA via vendor control OUT (bRequest=0xE5). wValue=addr, wIndex=val."""
    for off, val in enumerate(data): self.usb.control_write(0xE5, value=base_addr + off, index=val)

  def scsi_write(self, buf:bytes, slot_start:int=0):
    """Write to SRAM via 0xF2 vendor command + bulk OUT."""
    buf_padded = buf + b'\x00' * (round_up(len(buf), 512) - len(buf))
    self.usb.control_write(0xF2, value=len(buf_padded) // 512, index=(slot_start & 0xFF) | (ceildiv(len(buf_padded), 0x4000) << 8))
    self.usb.bulk_write(buf_padded)

class USBMMIOInterface(MMIOInterface):
  def __init__(self, usb, addr, size, fmt, pcimem=True): # pylint: disable=super-init-not-called
    self.usb, self.addr, self.nbytes, self.fmt, self.el_sz, self.pcimem = usb, addr, size, fmt, struct.calcsize(fmt), pcimem

  def _off_from_index(self, index):
    if isinstance(index, slice): return ((index.start or 0) * self.el_sz, ((index.stop or len(self))-(index.start or 0)) * self.el_sz)
    return (index * self.el_sz, self.el_sz)

  def __getitem__(self, index):
    off, sz = self._off_from_index(index)
    if self.pcimem:
      assert sz % 4 == 0 and off % 4 == 0, f"pcie_mem_read requires 4-byte aligned access, got off={off}, sz={sz}"
      data = self.usb.pcie_mem_read(self.addr + off, sz)
    else: data = self.usb.read(self.addr + off, sz)
    return data if isinstance(index, slice) else int.from_bytes(data, "little")

  def __setitem__(self, index, data):
    off, _ = self._off_from_index(index)
    data = struct.pack(self.fmt, data) if isinstance(data, int) else bytes(data)
    if not self.pcimem: self.usb.scsi_write(data) if self.addr == 0xf000 else self.usb.write(self.addr + off, data)
    else:
      # writes are whole dwords
      assert len(data) % 4 == 0 and off % 4 == 0, f"pcie_mem_write requires 4-byte aligned access, got off={off}, sz={len(data)}"
      self.usb.pcie_mem_write(self.addr+off, data)

  def view(self, offset:int=0, size:int|None=None, fmt=None):
    return USBMMIOInterface(self.usb, self.addr+offset, self.nbytes-offset if size is None else size, fmt=fmt or self.fmt, pcimem=self.pcimem)

# *****************
# 1. placeholders

HALF, CHUNK, SLOT, STREAM = 0x40000, 0x40000 - 512, 0x4000, 1 << 20 # sram half, payload, slot, stream
HOST_SIZE = 64 + 2 * HALF + STREAM # link, staging, zeros

def usb_host(dev:str) -> UOp: return UOp.alloc((HOST_SIZE,), dtypes.uint8, 0, device=HCQ_RUNTIME_DEV.value).rtag(to_name(dev, "usb_host"))
def usb_link(dev:str) -> UOp: return usb_host(dev)[:48].bitcast(dtypes.uint64) # handle, context, prev chunks, two transfers, error state
def usb_stage(dev:str) -> UOp: return usb_host(dev)[64:64 + 2 * HALF]
def usb_zeros(dev:str) -> UOp: return usb_host(dev)[64 + 2 * HALF:]

def usb_go(dev:str) -> UOp: return UOp.alloc((1,), dtypes.uint32, 0, device=dev).rtag(to_name(dev, "usb_go")) # chunk + 1: read armed

def usb_asm24(dev:str) -> UOp: return UOp.alloc((0x85000,), dtypes.uint8, 0, device=dev).rtag(to_name(dev, "usb_asm24")) # sys, cq, sram
def usb_fence(dev:str) -> UOp: return usb_asm24(dev)[0x800:0x804].bitcast(dtypes.uint32) # gpu chunks done
def usb_cq(dev:str) -> UOp: return usb_asm24(dev)[0x100c:0x1010].bitcast(dtypes.uint32) # completion queue
def usb_sram(dev:str) -> UOp: return usb_asm24(dev)[0x5000:0x5000 + 2 * HALF]

# *****************
# 2. transfers

def usb_word(v:UOp|int, dt:DType) -> UOp: return v.cast(dt) if isinstance(v, UOp) else UOp.const(v, dt)
def usb_stack(dt:DType, *vals:UOp|int) -> UOp:
  return (r:=UOp.placeholder((len(vals) or 1,), dt, addrspace=AddrSpace.REG)).after(*[r.index(i).store(usb_word(v, dt)) for i, v in enumerate(vals)])

def usb_fail(link:UOp, code:UOp) -> UOp: return link.index(UOp.const(5).valid(link.after(code).index(5).load().eq(0))).store(code.cast(dtypes.uint64))
def usb_ctrl(link:UOp, rtype:int, req:int, val:UOp|int, idx:UOp|int, data:UOp, n:UOp|int, timeout:int=1000) -> UOp:
  return usb_fail(link, ccall(libusb.libusb_control_transfer, link.index(0).load(), rtype, req, val, idx, data, n, timeout).minimum(0))
def usb_bulk(link:UOp, ep:int, data:UOp, n:UOp|int, timeout:int=1000) -> UOp: # shorter transfer fails
  rc = ccall(libusb.libusb_bulk_transfer, link.index(0).load(), ep, data, n, (got:=usb_stack(dtypes.int32, 0)).index(0), timeout)
  return usb_fail(link, rc.minimum(0).minimum(-got.after(rc).index(0).load().ne(n).cast(dtypes.int)))

@uopfunc
def usb_poke(link:UOp, addr:UOp, val:UOp) -> UOp: # 0xF0 mode 0: a dword
  return usb_ctrl(link, 0x40, 0xF0, 0x60 | 0x0F00, 0, usb_stack(dtypes.uint64, addr, val.cast(dtypes.uint64)).index(0), 12, 5000).sink()

@uopfunc
def usb_stream(link:UOp, addr:UOp, data:UOp, n:UOp, write:bool, fill:bool=False) -> UOp: # 0xF0 mode 1/2: header, then bulk per STREAM
  i = UOp.range(((n + (STREAM - 1)) // STREAM).after(link), next(UOp.unique_num), dtype=dtypes.int)
  off, cnt = i * STREAM, (n - i * STREAM).minimum(STREAM)
  payload = usb_stack(dtypes.uint64, addr + off.cast(dtypes.uint64), cnt // 4)
  live = UOp.range(link.after(i).index(5).load().eq(0).cast(dtypes.int).after(link), next(UOp.unique_num), dtype=dtypes.int) # no trips once failed
  header = usb_ctrl(link.after(live), 0x40, 0xF0, (0x60 if write else 0x20) | 0x0F00, 1 if write else 2, payload.index(0), 12, 5000)
  return usb_bulk(link.after(header), 0x02 if write else 0x81, data.index(0 if fill else off // data.dtype.itemsize), cnt).end(live, i).sink()

def usb_poke_word(link:UOp, addr:UOp, val:UOp) -> UOp:
  low = usb_poke(link, addr, val.cast(dtypes.uint32))
  return usb_poke(link.after(low), addr + 4, (val >> 32).cast(dtypes.uint32)) if val.dtype.itemsize == 8 else low

@uopfunc
def usb_patch(link:UOp, addr:UOp, val:UOp, slot:UOp) -> UOp: # poke only on change
  changed = UOp.range(slot.index(0).load().ne(val).cast(dtypes.int).after(link), next(UOp.unique_num), dtype=dtypes.int)
  return slot.after(usb_poke_word(link.after(changed), addr, val).end(changed)).index(0).store(val).sink()

# *****************
# 3. lower

def is_host(b:UOp) -> bool: return b.device is None or not all_devices_in(b.device, HCQ_DEVS - {"CPU"}) # register or host memory
def is_remote(b:UOp) -> bool: return (p:=unwrap_view(b)[0]).op in (Ops.PARAM, Ops.ALLOC) and not is_host(p)
def usb_addr(b:UOp, idx:UOp, dt:DType) -> UOp: return b.getaddr("CPU") + (idx * dt.itemsize).cast(dtypes.uint64) # byte address of b[idx]
def usb_deps(b:UOp) -> tuple[UOp, ...]: # deps through views
  return (b.src[1:] if b.op is Ops.AFTER else ()) + (usb_deps(b.src[0]) if b.op in (Ops.BITCAST, Ops.SHRINK, Ops.AFTER) else ())
def usb_affine(idx:UOp, r:UOp) -> UOp|None: # base of base + r
  if idx is r: return UOp.const(0, r.dtype)
  if idx.op is not Ops.ADD or r not in idx.src: return None
  base = idx.src[1] if idx.src[0] is r else idx.src[0]
  return base if r not in base.ranges else None

def usb_copy(dst:UOp, di:UOp, v:UOp, r:UOp) -> UOp|None: # ranged store: one stream
  if not is_remote(dst): return None
  if v.op is Ops.LOAD and not is_remote(sb:=v.src[0].src[0]): s0, deps, fill = usb_affine(v.src[0].src[1], r), usb_deps(sb), False
  elif v.vmin == v.vmax == 0: sb, s0, deps, fill = usb_zeros(dst.device), UOp.const(0, dtypes.int), (), True
  else: return usb_store(dst, di, v).end(r)
  if s0 is None or (d0:=usb_affine(di, r)) is None: return usb_store(dst, di, v).end(r)
  link = usb_link(dst.device).after(*usb_deps(dst), *deps, *r.src[1:])
  return usb_stream(link, usb_addr(dst, d0, v.dtype), sb.index(s0), r.src[0] * v.dtype.itemsize, True, fill)

def usb_store(b:UOp, idx:UOp, v:UOp) -> UOp: # kernargs: poke on change
  if idx.op is Ops.STACK: # word by word
    stored = usb_store(b, idx.src[0], v.src[0])
    for i, w in zip(idx.src[1:], v.src[1:]): stored = usb_store(b.after(stored), i, w)
    return stored
  link, addr, v = usb_link(b.device).after(*usb_deps(b)), usb_addr(b, idx, v.dtype), v.bitcast(CDTYPE[v.dtype.itemsize])
  if str(unwrap_view(b)[0].tag).startswith("kernargs"):
    cache = UOp.alloc((int(idx.vmax - idx.vmin) + 1,), v.dtype, device=HCQ_RUNTIME_DEV.value).rtag("usb_arg_cache")
    return usb_patch(link, addr, v, patch(cache, [], bytes(v.dtype.itemsize * cache.max_numel())).index(idx - idx.vmin))
  return usb_poke_word(link, addr, v)

def usb_load(b:UOp, idx:UOp, ld:UOp) -> UOp: # all ones once the link failed, like a dead pcie device, so polls end
  link, slot, n = usb_link(b.device).after(*usb_deps(b)), usb_stack(ld.dtype), UOp.const(ld.dtype.itemsize, dtypes.int)
  read = usb_stream(link, usb_addr(b, idx, ld.dtype), slot.index(0), n, False)
  return link.after(read).index(5).load().ne(0).where(UOp.const(ld.dtype.max, ld.dtype), slot.after(read).index(0).load())

pm_usb_lower = PatternMatcher([
  (UPat.var("dst").index(UPat.var("di")).store(UPat.var("v")).end(UPat(Ops.RANGE, name="r")), usb_copy),
  (UPat.var("b").index(UPat.var("idx")).store(UPat.var("v")), lambda b, idx, v: None if idx.ranges or not is_remote(b) else usb_store(b, idx, v)),
  (UPat.var("b").index(UPat.var("idx")).load(name="ld"), lambda b, idx, ld: usb_load(b, idx, ld) if is_remote(b) else None),
])

# *****************
# 4. copies

def usb_wire(size:UOp|int) -> UOp|int: return (size + 512 + SLOT - 1) // SLOT * SLOT # payload + sentinel block, slot aligned
def usb_sentinel(n:UOp) -> UOp: return ((n & 0xFFFFFF) | 0x51000000).cast(dtypes.uint32)
def usb_put(link:UOp, ptr:UOp, dt:DType, *vals:UOp|int) -> UOp: # store through a pointer
  return ccall(libc.memcpy, ptr, usb_stack(dt, *vals).after(link).index(0), dt.itemsize * len(vals)).cast(dtypes.void)

@uopfunc
def usb_reap(link:UOp, xfer:UOp) -> UOp: # poll while pending (0xff), any other status but completed fails the link
  loop, slot = UOp.range(UOp(Ops.NOOP).after(link), next(UOp.unique_num), dtype=dtypes.void), usb_stack(dtypes.uint32)
  events = ccall(libusb.libusb_handle_events_timeout, link.after(loop).index(1).load(), usb_stack(dtypes.uint64, 0, 0).index(0)) # zero timeout
  peeked = ccall(libc.memcpy, slot.after(events).index(0), xfer + 16, 4)
  ok = link.index(5).load().eq(0) & ((events >= 0) | events.eq(libusb.LIBUSB_ERROR_INTERRUPTED))
  return usb_fail(link, slot.after(events.backedge(loop, slot.after(peeked).index(0).load().eq(0xff) & ok)).index(0).load()).sink()

@uopfunc
def usb_drain(link:UOp, fence:UOp, need:UOp) -> UOp: # fence == need - 1 or need, mod 256
  loop, slot = UOp.range(UOp(Ops.NOOP).after(link, start:=get_time_ms(link)), next(UOp.unique_num), dtype=dtypes.void), usb_stack(dtypes.uint32, 0)
  read = usb_ctrl(link.after(loop), 0xC0, 0xE4, fence, 0, slot.index(0), 1)
  def behind(dep:UOp) -> UOp: return ((need - slot.after(dep).index(0).load().cast(dtypes.uint64)) & 0xff) > 1
  done = read.backedge(loop, link.after(read).index(5).load().eq(0) & behind(read) & (get_time_ms(read) - start < 1000))
  return usb_fail(link.after(done), behind(done).cast(dtypes.int) * libusb.LIBUSB_ERROR_TIMEOUT).sink()

@uopfunc
def usb_begin(link:UOp, fence:UOp, prev:UOp) -> UOp: # previous batch drained, count restarts
  return usb_ctrl(link.after(usb_drain(link, fence, prev + 1)), 0x40, 0xE5, fence, 0, UOp.const(0, dtypes.uint64), 0).sink()

@uopfunc
def usb_send(link:UOp, table:UOp, i:UOp, run:UOp, fence:UOp, stage:UOp) -> UOp: # chunk i -> sram half i & 1
  addr, size = table.index(2 * i).load(), table.index(2 * i + 1).load().cast(dtypes.int)
  n, half, wire = (i + run).cast(dtypes.uint64), i & 1, usb_wire(size)
  xfer, end = link.index(3 + half).load(), (half + 1) * HALF

  # host half free: fill it
  reaped = usb_reap(link, xfer)
  copied = ccall(libc.memcpy, stage.after(reaped).index(end - wire), addr, size.cast(dtypes.uint64))
  sealed = stage.after(copied).bitcast(dtypes.uint32).index(end // 4 - 1).store(usb_sentinel(n))

  # sram half free: arm, submit
  drained = usb_drain(link.after(sealed), fence, n)
  armed = usb_ctrl(link.after(drained), 0x40, 0xF2, wire // 512, ((end - wire) // SLOT) | (wire // SLOT << 8), UOp.const(0, dtypes.uint64), 0)
  status = usb_put(link.after(armed), xfer + 16, dtypes.uint32, 0xff, wire)
  buffer = usb_put(link.after(armed), xfer + 48, dtypes.uint64, stage.getaddr("CPU") + (end - wire).cast(dtypes.uint64))
  ret = ccall(libusb.libusb_submit_transfer, link.after(status, buffer).index(3 + half).load()) # read after the fields
  return usb_fail(link, ret.minimum(0)).sink()

@uopfunc
def usb_recv(link:UOp, table:UOp, i:UOp, run:UOp, go:UOp, stage:UOp) -> UOp: # chunk i <- both halves
  addr, size = table.index(2 * i).load(), table.index(2 * i + 1).load().cast(dtypes.int)
  first, second = size.minimum(CHUNK), (size - CHUNK).maximum(0) # bytes per half
  wire = (size + (second > 0).cast(dtypes.int) * 512 + 511) // 512 * 512 # the gap block is read too

  # arm, let the gpu copy, receive
  armed = usb_ctrl(link, 0x40, 0xF2, (wire // 512) | 0x8000, (wire + SLOT - 1) // SLOT << 8, UOp.const(0, dtypes.uint64), 0)
  released = usb_poke(link.after(armed), go, (i + run + 1).cast(dtypes.uint32))
  received = usb_bulk(link.after(released), 0x81, stage.index(0), wire)
  lower = ccall(libc.memcpy, addr, stage.after(received).index(0), first.cast(dtypes.uint64))
  return ccall(libc.memcpy, addr + CHUNK, stage.after(lower).index(HALF), second.cast(dtypes.uint64)).cast(dtypes.void).sink()

# *****************
# 5. batch and encode

def is_staged(call:UOp) -> bool: return call.op is Ops.CALL and call.body.op is Ops.STORE and is_host(call.src[1]) != is_host(call.src[2])
def usb_window(call:UOp) -> tuple[UOp, int]: return (call.src[2], CHUNK) if is_host(call.src[2]) else (call.src[1], 2 * CHUNK) # host, chunk bytes
def usb_hostaddr(host:UOp, dev:str) -> UOp:
  base, boff = unwrap_view(host)
  return base.bitcast(dtypes.uint8)[boff:boff + host.nbytes()].getaddr(dev)

def usb_table(hosts:list[tuple[UOp, int]], n:int, win:int, dev:str) -> UOp: # [address, bytes] per chunk
  rows = [(16 * (k + r) + o, w) for host, k in hosts for r, nb in chunks(host.nbytes(), win)
          for o, w in ((0, usb_hostaddr(host, dev) + usb_word(r, dtypes.uint64) * win), (8, UOp.const(nb, dtypes.uint64)))]
  return patch(UOp.alloc((2 * n,), dtypes.uint64, device=HCQ_RUNTIME_DEV.value).rtag("usb_table"), rows)

def usb_chunks(call:UOp, first:int, run:int) -> list[UOp]: # gpu side. first: chunk id of the copy and of its run
  dst, src = call.src[1:]
  vram, (host, win), ops = (dst if is_host(src) else src).bitcast(dtypes.uint8), usb_window(call), list[UOp]()
  sram = usb_sram(dev:=vram.device).getaddr(dev)

  for r, nb in chunks(host.nbytes(), win):
    n, va = (i:=usb_word(r, dtypes.uint64)) + first, vram.getaddr(dev) + i * win
    if is_host(src): # copyin: wait sentinel, copy, release
      end = sram + (((n - run) & 1) + 1) * HALF
      cmds = [ins("wait_eq", end - 4, usb_sentinel(n)), ins("copy", va, end - usb_wire(nb), nb), ins("store", end - 4, 0)]
    else: # copyout: wait go, fill, signal
      cmds = [ins("wait", usb_go(dev), n + 1), ins("store", usb_go(dev), 0)]
      cmds += [ins("copy", sram + wo, va + po, pb) for wo, po, pb in ((0, 0, min(nb, CHUNK)), (HALF, CHUNK, nb - CHUNK)) if pb > 0]
      cmds += [ins("store", usb_cq(dev), 0)]
    cmds += [ins("store", usb_fence(dev), n + 1)]
    ops += [UOp(Ops.LINEAR, src=tuple(cmds)).end(r)] if isinstance(r, UOp) else cmds # full chunks as one block
  return ops

def usb_copy_rewriter(s:UOp) -> UOp|None:
  lins = [submit.without_after.src[1].without_after for submit in s.src]
  if not (copies:=[call for lin in lins for call in lin.src if is_staged(call)]): return None
  dev = lins[0].arg[0][0]

  # a run: one direction, one table
  runs, nums, n = list[tuple[bool, int, UOp, int]](), list[tuple[int, int]](), 0
  for cin, grp in itertools.groupby(copies, key=lambda call: is_host(call.src[2])):
    hosts, k = list[tuple[UOp, int]](), 0 # (host view, first chunk)
    for host, win in map(usb_window, grp): nums, hosts, k = nums + [(n + k, n)], hosts + [(host, k)], k + ceildiv(host.nbytes(), win)
    runs, n = runs + [(cin, n, usb_table(hosts, k, win, dev), k)], n + k

  # gpu side
  chunks = (usb_chunks(call, *num) for call, num in zip(copies, nums))
  s = s.substitute({lin: lin.replace(src=tuple(flatten(next(chunks) if is_staged(c) else [c] for c in lin.src))) for lin in lins})

  # host side, after the submits
  link = usb_link(dev).after(s.src[-1])
  done = UOp.custom_function("usb_begin").call(link, usb_fence(dev), link.index(2).load())
  for cin, run, table, k in runs:
    if run: done = UOp.custom_function("usb_drain").call(link.after(done), usb_fence(dev), UOp.const(run + 1, dtypes.uint64)) # a run uses both halves
    i = UOp.range(k, next(UOp.unique_num), dtype=dtypes.int)
    word = usb_fence(dev) if cin else usb_go(dev)
    live = UOp.range(link.after(done, i).index(5).load().eq(0).cast(dtypes.int), next(UOp.unique_num), dtype=dtypes.int) # no trips once failed
    done = UOp.custom_function("usb_send" if cin else "usb_recv").call(link.after(live), table, i, UOp.const(run, dtypes.int), word).end(live, i)
  return s.replace(src=(*s.src, link.after(done).index(2).store(UOp.const(n, dtypes.uint64))))
pm_usb_batch = PatternMatcher([(UPat(Ops.SINK, name="s"), usb_copy_rewriter)])

pm_usb_encode = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat.custom_function("usb_begin"), UPat.var("link"), UPat.var("fence"), UPat.var("prev"))),
   lambda link, fence, prev: usb_begin(link, fence.getaddr("CPU"), prev)),
  (UPat(Ops.CALL, src=(UPat.custom_function("usb_drain"), UPat.var("link"), UPat.var("fence"), UPat.var("need"))),
   lambda link, fence, need: usb_drain(link, fence.getaddr("CPU"), need)),
  (UPat(Ops.CALL, src=(UPat.custom_function("usb_send"), UPat.var("link"), UPat.var("table"), UPat.var("i"), UPat.var("run"), UPat.var("fence"))),
   lambda link, table, i, run, fence: usb_send(link, table, i, run, fence.getaddr("CPU"), usb_stage(fence.device))),
  (UPat(Ops.CALL, src=(UPat.custom_function("usb_recv"), UPat.var("link"), UPat.var("table"), UPat.var("i"), UPat.var("run"), UPat.var("go"))),
   lambda link, table, i, run, go: usb_recv(link, table, i, run, go.getaddr("CPU"), usb_stage(go.device))),
])

# *****************
# 6. bufferize

@functools.cache
def _host_block(dev) -> Buffer:
  xfers = [libusb.libusb_alloc_transfer(0).contents for _ in range(2)]
  for t in xfers: t.dev_handle, t.endpoint, t.type, t.timeout = dev.iface.pci_dev.usb.usb.handle, 0x02, libusb.LIBUSB_TRANSFER_TYPE_BULK, 10000
  words = [ctypes.addressof(x.contents) for x in (dev.iface.pci_dev.usb.usb.handle, USB3.ctx())] + [0] + [ctypes.addressof(t) for t in xfers]
  return Buffer("CPU", HOST_SIZE, options=BufferSpec(nolru=True), initial_value=struct.pack('5Q', *words).ljust(HOST_SIZE, b'\0'))

@functools.cache
def _go(dev) -> Buffer:
  return Buffer(dev.device, 4, options=BufferSpec(uncached=True, cpu_access=True, nolru=True), initial_value=bytes(4))

def usb_reset(dev):
  for buf, off, n in ((dev.iface.ctrl, 0x800, 4), (dev.iface.ctrl, 0x5000, 0x80000), (_host_block(dev), 16, 8)): buf.host.view(off, n)[:] = bytes(n)
  for t in map(libusb.struct_libusb_transfer.from_address, _host_block(dev).host.view(fmt='Q')[3:5]): t.status = 0xff * (t.status == 0xff)

def setup_usb_rules(dev):
  dev.pm_batch, dev.pm_lower, dev.pm_encode = dev.pm_batch + pm_usb_batch, dev.pm_lower + pm_usb_lower, dev.pm_encode + pm_usb_encode
  Compiled.pm_bufferize += PatternMatcher([(UPat(Ops.ALLOC, tag=dev.tag("usb_host")), lambda d=dev: _host_block(d)), # placeholders the gpu owns
                                           (UPat(Ops.ALLOC, tag=dev.tag("usb_go")), lambda d=dev: _go(d)),
                                           (UPat(Ops.ALLOC, tag=dev.tag("usb_asm24")), lambda d=dev: d.iface.ctrl)])
  dev.error_state = _host_block(dev).view(8, 40) # the link's error word

if DEV.interface.startswith("MOCK"): from test.mockgpu.usb import MockUSB3 as USB3  # type: ignore  # noqa: F811
