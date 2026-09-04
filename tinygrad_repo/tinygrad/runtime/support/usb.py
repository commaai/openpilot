import ctypes, struct, time, functools, itertools
from typing import Any, cast
from tinygrad.runtime.autogen import libusb
from tinygrad.helpers import DEBUG, DEV, to_mv, from_mv, round_up, ceildiv, unwrap, to_tuple
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp, UPat, Ops, PatternMatcher
from tinygrad.device import Buffer, BufferSpec, Device
from tinygrad.runtime.support.hcq2 import HCQInfo, make_buf, make_submit, HCQ_RUNTIME_DEV
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

  def __init__(self, dev:c.POINTER[libusb.struct_libusb_device], *args, **kwargs):
    self._tags, self._transferred = itertools.count(1), ctypes.c_int(0)
    self._bulk_buf, self._bulk_mv = alloc_cbuffer(4 << 20)
    self._ctrl_buf, self._ctrl_mv = alloc_cbuffer(0x1000)
    # async bulk OUT state: tag -> (pooled transfer, keepalive payload mv); transfer errors latch into _async_err
    self._async_seq, self._async_err = itertools.count(1), 0
    self._async_pending: dict = {}
    self._async_pool: list = []
    self._async_cb = libusb.libusb_transfer_cb_fn(self._on_bulk_done)

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

  def _on_bulk_done(self, xfer):  # runs in libusb event handling; latch errors (exceptions here are unraisable)
    exp = xfer.contents.length - 8 if xfer.contents.type == libusb.LIBUSB_TRANSFER_TYPE_CONTROL else xfer.contents.length
    if xfer.contents.status != 0 or xfer.contents.actual_length != exp: self._async_err = xfer.contents.status or -1
    self._async_pool.append(self._async_pending.pop(int(xfer.contents.user_data or 0))[0])

  def _submit_async(self, endpoint:int, xtype:int, payload:bytes|bytearray|memoryview, timeout:int) -> int:  # payload kept alive till bulk_wait
    tr = self._async_pool.pop() if self._async_pool else libusb.libusb_alloc_transfer(0)
    tr.contents.dev_handle, tr.contents.endpoint, tr.contents.type = self.handle, endpoint, xtype
    tr.contents.timeout, tr.contents.length = timeout, len(payload)
    tr.contents.buffer = ctypes.cast(from_mv(memoryview(payload), ctypes.c_ubyte), ctypes.POINTER(ctypes.c_ubyte))
    tr.contents.callback, tr.contents.user_data = self._async_cb, (tag := next(self._async_seq))
    self._async_pending[tag] = (tr, payload)
    checked(libusb.libusb_submit_transfer, "async submit failed")(tr)
    return tag

  def bulk_write_async(self, payload:memoryview, timeout:int=10000) -> int:
    """Queue a bulk OUT transfer without blocking; payload is kept alive until bulk_wait(tag)."""
    return self._submit_async(0x02, libusb.LIBUSB_TRANSFER_TYPE_BULK, payload, timeout)

  def control_write_async(self, request:int, value:int=0, index:int=0, data:bytes=b"", timeout:int=1000) -> int:
    """Queue a vendor control OUT without blocking; completes via bulk_wait(tag) like bulk_write_async."""
    setup = bytearray(struct.pack('<BBHHH', 0x40, request, value, index, len(data)) + data)
    return self._submit_async(0, libusb.LIBUSB_TRANSFER_TYPE_CONTROL, setup, timeout)

  def control_read_async(self, request:int, length:int, value:int=0, index:int=0, timeout:int=1000) -> tuple[int, memoryview]:
    """Queue a vendor control IN without blocking; the data lands in the returned buffer by bulk_wait(tag)."""
    buf = bytearray(struct.pack('<BBHHH', 0xC0, request, value, index, length)) + bytearray(length)
    return self._submit_async(0, libusb.LIBUSB_TRANSFER_TYPE_CONTROL, buf, timeout), memoryview(buf)[8:]

  def bulk_wait(self, tag:int):
    """Block until the tagged transfer completes; raises if any async transfer failed. LIBUSB_ERROR_INTERRUPTED is retried."""
    while tag in self._async_pending:
      if (rc:=libusb.libusb_handle_events(None)) < 0 and rc != libusb.LIBUSB_ERROR_INTERRUPTED:
        raise RuntimeError(f"libusb_handle_events: {ctypes.string_at(libusb.libusb_strerror(rc)).decode()}")
    if self._async_err: raise RuntimeError(f"async bulk OUT failed: status={self._async_err}")

  def bulk_read(self, length:int, timeout:int=1000) -> memoryview:
    if length > len(self._bulk_mv): self._bulk_buf, self._bulk_mv = alloc_cbuffer(length)
    checked(libusb.libusb_bulk_transfer, "bulk IN 0x81 failed")(self.handle, 0x81, self._bulk_buf, length, self._transferred, timeout)
    return self._bulk_mv[:self._transferred.value]

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
    self._f0_out(0x60, 0x0F, address, len(data) // 4, mode=1)
    self.usb.bulk_write(data)

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

  def scsi_read_arm(self, size:int):
    windex = (ceildiv(size, 0x4000) & 0xFF) << 8
    self.usb.control_write(0xF2, value=(ceildiv(size, 512) & 0x7FFF) | 0x8000, index=windex)

  def scsi_read(self, size:int) -> memoryview: return self.usb.bulk_read(round_up(size, 512), timeout=10000)[:size]

class USBMMIOInterface(MMIOInterface):
  def __init__(self, usb, addr, size, fmt, pcimem=True): # pylint: disable=super-init-not-called
    self.usb, self.addr, self.nbytes, self.fmt, self.el_sz, self.pcimem = usb, addr, size, fmt, struct.calcsize(fmt), pcimem

  def _off_from_index(self, index):
    if isinstance(index, slice): return ((index.start or 0) * self.el_sz, ((index.stop or len(self))-(index.start or 0)) * self.el_sz)
    return (index * self.el_sz, self.el_sz)

  def __getitem__(self, index):
    Device[HCQ_RUNTIME_DEV.value].synchronize() # one driver on the link: drain the compiled submits before python touches it
    off, sz = self._off_from_index(index)
    if self.pcimem:
      assert sz % 4 == 0 and off % 4 == 0, f"pcie_mem_read requires 4-byte aligned access, got off={off}, sz={sz}"
      data = self.usb.pcie_mem_read(self.addr + off, sz)
    else: data = self.usb.scsi_read(sz) if self.addr == 0xf000 else self.usb.read(self.addr + off, sz)
    return data if isinstance(index, slice) else int.from_bytes(data, "little")

  def __setitem__(self, index, data):
    Device[HCQ_RUNTIME_DEV.value].synchronize()
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

def _libusb(devs, dep:tuple[UOp, ...], fn:str, *args) -> UOp:
  return make_buf(devs, tag=f"func:{fn}").after(*dep).index(0).load().call(make_buf(devs, tag="usb_handle").index(0).load(),
    *[UOp.const(a, dtypes.int) if isinstance(a, int) else a for a in args], ret_dtype=dtypes.void)

def usb_bulk(devs, dep, endpoint:int, data:UOp, length, timeout:int=1000) -> UOp: # NULL actual_length out param
  return _libusb(devs, dep, "libusb_bulk_transfer", endpoint, data, length, UOp.const(0, dtypes.uint64), timeout)

def usb_stream(devs, dep:tuple[UOp, ...], addr:UOp, data:UOp, nbytes:int, write:bool) -> UOp:
  hdr = UOp.placeholder((2,), dtypes.uint64, device=devs, tag="usb_scratch").after(*dep)
  arm = _libusb(devs, (hdr.index(0).store(addr), hdr.index(1).store(UOp.const(nbytes // 4, dtypes.uint64))), "libusb_control_transfer",
                0x40, 0xF0, (0x60 if write else 0x20) | (0x0F << 8), 1 if write else 2, hdr.index(0), 12, 5000)
  return usb_bulk(devs, (arm,), 0x02 if write else 0x81, data, nbytes)

def usb_load(b:UOp, idx:UOp, dt) -> UOp:
  got = UOp.placeholder((1,), dt, device=(devs:=to_tuple(b.device)), tag="usb_scratch")
  addr = b.getaddr((HCQ_RUNTIME_DEV.value,)) + (idx*dt.itemsize).cast(dtypes.uint64)
  return got.after(usb_stream(devs, b.src[1:] if b.op is Ops.AFTER else (), addr, got.index(0), dt.itemsize, False)).index(0).load()

def usb_write(b:UOp, idx:UOp, v:UOp) -> UOp:
  val = (s:=UOp.placeholder((1,), v.dtype, device=(devs:=to_tuple(b.device)), tag="usb_scratch")).after(s.index(0).store(v))
  addr = b.getaddr((HCQ_RUNTIME_DEV.value,)) + (idx*v.dtype.itemsize).cast(dtypes.uint64)
  return usb_stream(devs, b.src[1:] if b.op is Ops.AFTER else (), addr, val.index(0), v.dtype.itemsize, True)

def usb_idle(devs) -> UOp:
  v = usb_load(make_buf(devs, tag="timeline_signal").after(loop:=UOp.loop(0)), UOp.const(0, dtypes.int), dtypes.uint64)
  return v.end(loop, v + 1 < make_buf(devs, tag="timeline_value").index(0).load())

def usb_scsi(devs, read:bool, nbytes:int) -> UOp:
  return _libusb(devs, (usb_idle(devs),), "libusb_control_transfer", 0x40, 0xF2, ceildiv(nbytes, 512) | (0x8000 if read else 0),
                 (ceildiv(nbytes, 0x4000) & 0xFF) << 8, UOp.const(0, dtypes.uint64), 0, 1000)

def usb_stage_copy(dst:UOp, src:UOp) -> UOp|None:
  if (cin:=to_tuple(src.device)[0].startswith("CPU")) == to_tuple(dst.device)[0].startswith("CPU"): return None

  total, ops, win = dst.nbytes(), [], cast(Any, Device[(devs:=to_tuple((dst if cin else src).device))[0]]).iface.usb_sram
  for off in range(0, total, win.size): # off and nb are bytes, the two ends of the copy can have different dtypes
    sram = UOp.from_buffer(win)[0:(nb:=min(win.size, total - off))]
    s, d = src[off // src.dtype.itemsize:(off + nb) // src.dtype.itemsize], dst[off // dst.dtype.itemsize:(off + nb) // dst.dtype.itemsize]
    if cin:
      push = usb_bulk(devs, (usb_scsi(devs, False, nb),), 0x02, s.getaddr((HCQ_RUNTIME_DEV.value,)), round_up(nb, 512), 10000)
      ops += [UOp.custom_function("hcq", push.sink()).call(sram, s, name="hcq_copyin", aux=HCQInfo(devs)),
              sram.copy_to_device(d.device).call(d, sram)]
    else:
      pad = UOp.new_buffer("CPU", round_up(nb, 512), dtypes.uint8)[0:nb]
      submit = make_submit(s.copy_to_device(sram.device).call(sram, s), devs=devs, queue="COPY:0")
      pull = usb_bulk(devs, (submit,), 0x81, pad.getaddr((HCQ_RUNTIME_DEV.value,)), round_up(nb, 512), 10000)
      ops += [UOp.custom_function("hcq", pull.sink()).call(pad, sram, s, name="hcq_copyout", aux=HCQInfo(devs)),
              pad.copy_to_device("CPU").call(d, pad)]
  return UOp(Ops.LINEAR, src=tuple(ops))
pm_usb_stage = PatternMatcher([(UPat(Ops.CALL, src=(UPat(Ops.COPY), UPat(name="dst"), UPat(name="src"))), usb_stage_copy)])

USB_HOST_TAGS = {"signal", "timeline_signal"}
pm_usb_hostio = PatternMatcher([
  (UPat(Ops.LOAD, src=(UPat(Ops.INDEX, src=(UPat(Ops.PARAM, tag=USB_HOST_TAGS).or_after(name="b"), UPat(name="idx"))),),
        name="ld"), lambda b, idx, ld: usb_load(b, idx, ld.dtype)),
  (UPat(Ops.STORE, src=(UPat(Ops.INDEX, src=(UPat(Ops.PARAM, tag=USB_HOST_TAGS).or_after(name="b"), UPat(name="idx"))), UPat(name="v"))), usb_write)])

pm_usb_bufferize = PatternMatcher([
  (UPat(Ops.PARAM, tag={"systems", "runtime", "inputs", "usb_scratch"}, name="b"),
   lambda ctx, b: Buffer("CPU", b.max_numel(), b.dtype, options=BufferSpec(nolru=True), preallocate=True)),
  (UPat(Ops.PARAM, tag="usb_handle", name="b"), lambda ctx, b: ctx[0].signal(b.tag, ctx[0].iface.usb_handle, device="CPU")),
  (UPat(Ops.PARAM, name="b"), lambda ctx, b: None if not isinstance(b.tag, str) or not b.tag.startswith("func:") else
   ctx[0].signal(b.tag, unwrap(ctypes.cast(getattr(libusb.dll, b.tag[5:]), ctypes.c_void_p).value), device="CPU")),
])

if DEV.interface.startswith("MOCK"): from test.mockgpu.usb import MockUSB3 as USB3  # type: ignore  # noqa: F811
