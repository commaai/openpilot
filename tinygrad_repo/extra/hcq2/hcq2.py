from __future__ import annotations
from typing import cast, Callable, TypeVar, Generic, Any, TYPE_CHECKING
import struct, functools, time, collections
from dataclasses import replace
if TYPE_CHECKING: from tinygrad.engine.realize import ExecContext
from tinygrad.helpers import DEV, getenv, select_first_inited, select_by_name, suppress_finalizing, mv_address, round_up, DEBUG, dedup
from tinygrad.device import Device, Buffer, BufferSpec, Compiled, LRUAllocator, MultiBuffer
from tinygrad.uop.ops import Ops, sint, UOp, UPat, PatternMatcher, KernelInfo, graph_rewrite, track_rewrites
from tinygrad.uop.symbolic import symbolic_simple, symbolic
from tinygrad.dtype import dtypes, DType
from dataclasses import dataclass, field
from tinygrad.runtime.support.memory import BumpAllocator
from tinygrad.runtime.support.hcq import MMIOInterface
from tinygrad.renderer import Renderer, Estimates
from tinygrad.engine.realize import to_program, track_stats, get_call_arg_uops, resolve_params

HCQDeviceType = TypeVar('HCQDeviceType', bound='HCQ2Compiled')

class HCQ2Compiled(Compiled):
  """
  A base class for devices compatible with the HCQ (Hardware Command Queue) API.
  """
  timestamp_divider: float = 1000.0  # GPU timestamp counter ticks per microsecond; override per device

  def __init__(self, device:str, allocator:'HCQAllocator', compilers:list[type[Renderer]], runtime,
               kernargs_size=(16 << 20), can_recover:bool=False, arch=None):
    self.device_id:int = int(device.split(":")[1]) if ":" in device else 0

    from extra.hcq2.graph.hcq import HCQ2Graph
    super().__init__(device, allocator, compilers, lambda *a, **kw: None, HCQ2Graph, arch=arch)

    self.kernargs_size = kernargs_size
    self.kernargs_offset_allocator:BumpAllocator = BumpAllocator(kernargs_size, wrap=True)

  @functools.cached_property
  def kernargs_buf(self) -> Buffer:
    return Buffer(self.device, self.kernargs_size, dtypes.uint8, options=BufferSpec(cpu_access=True), preallocate=True)

  @functools.cached_property
  def timeline_signal(self) -> Buffer:
    return Buffer(self.device, 0x100, dtypes.uint8, options=BufferSpec(host=True, uncached=True, cpu_access=True), preallocate=True)

  @functools.cached_property
  def timestamps_buf(self) -> Buffer:
    return Buffer(self.device, 0x100, dtypes.uint8, options=BufferSpec(cpu_access=True), preallocate=True)

  @functools.cached_property
  def timeline_value(self) -> Buffer:
    buf = Buffer("CPU", 1, dtypes.uint64, preallocate=True)
    buf.as_memoryview(force_zero_copy=True).cast('Q')[0] = 1
    return buf

  def synchronize(self, timeout:int|None=None):
    if not hasattr(self, 'iface'): return
    sig = self.timeline_signal._buf.cpu_view().mv.cast('Q')
    tl = self.timeline_value.as_memoryview(force_zero_copy=True).cast('Q')
    st = time.perf_counter()
    while sig[0] < tl[0] - 1:
      if time.perf_counter() - st > (timeout or 3000) / 1000: self.on_device_hang()

  def device_props(self) -> dict[str,Any]: return {} # to be overridden if needed. dict keys are backend dependent.

  def _realloc(self, oldbuf:HCQ2Buffer|None, new_size:int, options:BufferSpec|None=None, force=False) -> tuple[HCQ2Buffer, bool]:
    if oldbuf is not None: self.allocator.free(oldbuf, oldbuf.size, options=options)
    try: buf, realloced = self.allocator.alloc(new_size, options=options), True
    except MemoryError:
      if force: raise
      buf, realloced = self.allocator.alloc(oldbuf.size if oldbuf is not None else new_size, options=options), False
    return buf, realloced

  def count(self) -> int: return self.iface.count if hasattr(self, 'iface') else 1

  def _select_iface(self):
    assert (v:=getenv(k:=f'{type(self).__name__[:-6].upper()}_IFACE', "")) == "",  \
      f"{k}={v} is deprecated, use DEV={replace(DEV.target(type(self).__name__[:-6]), interface=v)} instead"
    assert hasattr(self, "ifaces"), "must have ifaces to select an iface"
    t = DEV.target(dev:=type(self).__name__[:-6])
    filtered = select_by_name(self.ifaces, lambda i: i.__name__[:-5], t.interface, f"{dev} has no interface {t.interface!r}")
    filtered = [i for i in filtered if t.interface.startswith("MOCK") or not i.__name__[:-5].startswith("MOCK")] # never fall back to mock ifaces
    return select_first_inited([functools.partial(cast(Callable, iface), self, self.device_id) for iface in filtered],
                               f"No interface for {dev}:{self.device_id} is available")

  def _is_cpu(self) -> bool: return hasattr(self, 'device') and self.device.split(":")[0] == "CPU"

  def finalize(self):
    try: self.synchronize() # try to finalize the device in any case
    except RuntimeError as e: print(f"{self.device} synchronization failed before finalizing: {e}")

    # if the device has an interface, call device_fini to clean up resources
    if hasattr(self, 'iface') and hasattr(self.iface, 'device_fini'): self.iface.device_fini()

class HCQ2Buffer:
  def __init__(self, va_addr:sint, size:int, meta:Any=None, _base:HCQ2Buffer|None=None, view:MMIOInterface|None=None, owner:HCQ2Compiled|None=None):
    self.va_addr, self.size, self.meta, self._base, self.view, self.owner = va_addr, size, meta, _base, view, owner

  def offset(self, offset:int=0, size:int|None=None) -> HCQ2Buffer:
    return HCQ2Buffer(self.va_addr+offset, size or (self.size - offset), owner=self.owner, meta=self.meta,
      _base=self._base or self, view=(self.view.view(offset=offset, size=size) if self.view is not None else None))

  def cpu_view(self) -> MMIOInterface:
    assert self.view is not None, "buffer has no cpu_view"
    return self.view

  @property
  def base(self) -> HCQ2Buffer: return self._base or self

class HCQAllocator(LRUAllocator[HCQDeviceType], Generic[HCQDeviceType]):
  def _map(self, buf:HCQ2Buffer) -> HCQ2Buffer:
    if not hasattr(self, '_do_map'): raise NotImplementedError("map failed: no method implemented")
    return self._do_map(buf)

  @suppress_finalizing
  def _free(self, buf:HCQ2Buffer, options:BufferSpec|None=None):
    if options is not None and options.external_ptr is not None: return
    if hasattr(self, '_do_free'): self._do_free(buf, options)

  def _unmap(self, mb):
    self.dev.synchronize()
    self.dev.iface.dev_impl.mm.unmap_range(int(mb.va_addr), round_up(mb.size, 0x1000))

  def _offset(self, buf, size:int, offset:int) -> HCQ2Buffer: return buf.offset(offset=offset, size=size)

  def _wrap(self, dev:str, sz:int, opaque:HCQ2Buffer) -> Buffer:
    return Buffer(dev, sz, dtypes.uint8, opaque=opaque, options=BufferSpec(external_ptr=1))

  def _copy(self, dst:Buffer, src:Buffer):
    from tinygrad.engine.realize import run_linear
    su = UOp.from_buffer(src)
    run_linear(UOp(Ops.LINEAR, dtypes.void, (su.copy_to_device(dst.device).call(UOp.from_buffer(dst), su),)), jit=True, update_stats=False)

  def _copyin(self, dest:HCQ2Buffer, src:memoryview):
    s = Buffer(self.dev.device, len(src), dtypes.uint8, options=BufferSpec(host=True), preallocate=True)
    s._buf.cpu_view()[:len(src)] = src
    self._copy(self._wrap(self.dev.device, len(src), dest), s)

  def _copyout(self, dest:memoryview, src:HCQ2Buffer):
    d = Buffer(self.dev.device, len(dest), dtypes.uint8, options=BufferSpec(host=True), preallocate=True)
    self._copy(d, self._wrap(self.dev.device, len(dest), src))
    self.dev.synchronize()
    dest[:] = d._buf.cpu_view()[:len(dest)]

  # def _as_buffer(self, buf): return buf.cpu_view().mv

# **************** lower context ****************

def unwrap_after(uop):
  while uop.op is Ops.AFTER: uop = uop.src[0]
  return uop

@dataclass
class HCQ2DeviceCtx:
  device:str                       # device name; resolve to instance via Device[device]
  kernargs_host:UOp                # UOp whose .buffer is dev.kernargs_buf (BUFFER UOp in runtime, PARAM in graph)
  kernargs_gpu:UOp                 # va_addr const of dev.kernargs_buf
  kernargs_allocator:BumpAllocator = field(default_factory=lambda: BumpAllocator(2 << 20, wrap=False))

@dataclass
class HCQ2LowerCtx:
  name:str
  inputs:list[Buffer|MultiBuffer] = field(default_factory=list)
  holds:list[UOp] = field(default_factory=list)
  dev_ctx:dict[str, HCQ2DeviceCtx] = field(default_factory=dict)
  addr_table:UOp|None = None
  next_slot:int = 0

class HCQEncoder:
  def __init__(self): self.blob, self.patches = b'', []

  def get_dev_addr(self, uop:UOp) -> UOp:
    return UOp(Ops.GETADDR, dtypes.uint64, src=(uop,)) if unwrap_after(uop).op in (Ops.BUFFER, Ops.BUFFER_VIEW, Ops.BINARY, Ops.MSTACK, Ops.MSELECT) else uop

  def append(self, *data, dtype=dtypes.uint32):
    for d in data:
      if isinstance(d, int): self.blob += struct.pack(f'<{dtype.fmt}', d)
      else:
        self.patches.append((len(self.blob), self.get_dev_addr(d), dtype))
        self.blob += struct.pack(f'<{dtype.fmt}', 0)

  def q(self, *values): self.append(*values)

  def uop(self, dev:str|tuple[str, ...], tag:str|None=None) -> UOp:
    buf = UOp.new_buffer(dev, len(self.blob), dtypes.uint8)
    if tag: buf = buf.rtag(tag)
    blob_uop = UOp(Ops.BINARY, dtypes.void, src=(), arg=self.blob)
    stores = [buf.index(UOp.const(dtypes.int, off)).cast(dt.ptr()).store(val.cast(dt)) for off, val, dt in self.patches]
    return buf.after(buf.store(blob_uop), *stores)

# **************** prepare runtime ****************

def _devices(x) -> tuple[str, ...]:
  return tuple(b.device for b in x.bufs) if isinstance(x, MultiBuffer) else (x.device,) if isinstance(x, Buffer) else x if isinstance(x, tuple) else (x,)

def rebind_program_dev(c:UOp, p:UOp) -> UOp:
  devs = _devices(c.src[1].buffer)
  p = p.replace(src=p.src[:1] + (UOp(Ops.DEVICE, arg=devs),) + p.src[2:])
  return c.replace(src=(Device[devs[0]].pm_lower.rewrite(p),) + c.src[1:])

def lower_kernargs(call:UOp, prg:UOp) -> UOp:
  data, info = prg.arg
  enc = HCQEncoder()
  for gi in info.globals: enc.append(call.src[1+gi], dtype=dtypes.uint64)
  for v in info.vars: enc.append(v, dtype=dtypes.uint32)

  enc.blob += b'\x00' * (data.kernargs_alloc_size - len(enc.blob)) # pad blob
  kernargs = enc.uop(_devices(call.src[1].buffer), tag="kernargs")
  return call.replace(src=(prg.replace(src=prg.src + (kernargs,), arg=(data, info)),) + call.src[1:])

pm_prep_runtime = PatternMatcher([
  # bind generic PROGRAM device to the call's actual dev(s), then run device-specific lowering
  (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, src=(UPat(), UPat(Ops.DEVICE), UPat(), UPat(), UPat(Ops.BINARY)), name="p"),), name="c", allow_any_len=True),
    rebind_program_dev),

  # lower kernargs (PROGRAM.src[0] is now AFTER(BUFFER, COPY) — the lowered program image)
  (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, src=(UPat(Ops.AFTER),), name="prg"),), name="call", allow_any_len=True), lower_kernargs),
])

# **************** lower ops ****************

def lower_program(call:UOp, prg:UOp) -> UOp:
  q = UOp(Ops.LINEAR, dtypes.void, (prg,), arg=(_devices(call.src[1].buffer), "COMPUTE"))
  return UOp(Ops.LINEAR, dtypes.void, (q,), tag=call.tag)

def lower_copy(call:UOp, copy:UOp) -> UOp:
  dst, src = call.src[1], call.src[2]
  q = UOp(Ops.LINEAR, dtypes.void, (UOp(Ops.COPY, dtypes.void, src=(dst, src), arg=src.buffer.nbytes),), arg=(_devices(dst.buffer), "COPY"))
  return UOp(Ops.LINEAR, dtypes.void, (q,), tag=call.tag)

pm_lower_ops = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, src=(UPat(Ops.AFTER), UPat()), name="prg"),), name="call", allow_any_len=True), lower_program),
  (UPat(Ops.CALL, src=(UPat(Ops.COPY, name="copy"),), name="call", allow_any_len=True), lower_copy),
])

def split_into_queues(outer:UOp) -> UOp:
  groups:dict[tuple, list[UOp]] = collections.defaultdict(list)
  for child in outer.src:
    wrapper = child.src[0] if child.op is Ops.AFTER else child
    for q in wrapper.src: groups[q.arg].extend(q.src)
  return outer.replace(src=tuple(UOp(Ops.LINEAR, dtypes.void, tuple(cmds), arg=k) for k, cmds in groups.items()))
pm_split_into_queues = PatternMatcher([(UPat(Ops.LINEAR, src=UPat(Ops.LINEAR, src=UPat(Ops.LINEAR)).or_after(), name="outer"), split_into_queues)])

def add_signals(q:UOp) -> UOp:
  sig = UOp.new_buffer(q.arg[0], 0x100, dtypes.uint8).rtag("timeline_signal")
  tl = UOp.new_buffer(q.arg[0], 1, dtypes.uint64).rtag("timeline_value").index(UOp.const(dtypes.int, 0))
  return q.replace(src=(sig.wait(tl-1), *q.src, sig.store(tl)), arg=q.arg)
pm_add_signals = PatternMatcher([(UPat(Ops.LINEAR, src=UPat(Ops.LINEAR), name="outer"),
  lambda outer: outer.replace(src=tuple(add_signals(q) for q in outer.src)))])

pm_add_barriers = PatternMatcher([(UPat(Ops.LINEAR, src=UPat(Ops.LINEAR), name="outer"),
  lambda outer: outer.replace(src=tuple(q.replace(src=(UOp(Ops.BARRIER, dtypes.void), *q.src)) for q in outer.src)))])

def add_timeline_inc(q:UOp) -> UOp:
  tl = UOp.new_buffer(q.arg[0], 1, dtypes.uint64).rtag("timeline_value")
  done = tl.after(UOp(Ops.BARRIER, dtypes.void, src=(q,)))
  return done.index(UOp.const(dtypes.int, 0), dtype=tl.dtype.ptr()).store(tl.index(UOp.const(dtypes.int, 0)) + 1)
pm_add_timeline_inc = PatternMatcher([(UPat(Ops.LINEAR, src=UPat(Ops.LINEAR), name="outer"),
  lambda outer: outer.replace(src=tuple(add_timeline_inc(q) for q in outer.src)))])

# **************** build host program ****************

def calc_kernargs_sizes(ctx:dict[str,int], u:UOp) -> None:
  if u.tag != "kernargs": return
  for d in _devices(u.src[1].arg): ctx[d] = ctx.get(d, 0) + round_up(u.arg, 16)
pm_calc_kernargs_sizes = PatternMatcher([(UPat(Ops.BUFFER, name="u"), calc_kernargs_sizes)])

# bufferize

def _maybe_mstack(srcs:tuple[UOp, ...], tag=None) -> UOp: return srcs[0] if len(srcs) == 1 else UOp(Ops.MSTACK, srcs[0].dtype, srcs, tag=tag)

def _lower_stores(host_buf:UOp, buf_node:UOp, stores:tuple[UOp, ...]) -> list[UOp]:
  def lower(s:UOp) -> UOp:
    if s.src[1].op is Ops.BINARY: return s.substitute({buf_node: host_buf})
    idx = s.src[0].src[0]
    return s.substitute({idx: host_buf.index(UOp.const(dtypes.int, idx.src[1].arg // host_buf.dtype.base.itemsize), dtype=host_buf.dtype.ptr())})
  return [lower(s) for s in stores]

_program_uop_cache:dict[tuple[bytes,str], tuple[UOp,UOp]] = {}
def bufferize_program(ctx:HCQ2LowerCtx, target:UOp, buf_node:UOp) -> UOp:
  blob, addrs = target.src[1].src[1].arg, []
  for dev in _devices(buf_node.src[1].arg):
    if (cached:=_program_uop_cache.get((blob, dev))) is None:
      lib_gpu = Buffer(dev, round_up(len(blob), 0x1000), dtypes.uint8, options=BufferSpec(nolru=True, cpu_access=True), preallocate=True)
      lib_gpu._buf.cpu_view()[:len(blob)] = memoryview(blob)
      cached = _program_uop_cache[(blob, dev)] = (UOp.from_buffer(lib_gpu, dev), UOp.const(dtypes.uint64, lib_gpu._buf.va_addr, device=dev))
    if cached[0] not in ctx.holds: ctx.holds.append(cached[0])
    addrs.append(cached[1])
  return _maybe_mstack(tuple(addrs))

def bufferize_kernargs(ctx:HCQ2LowerCtx, target:UOp, buf_node:UOp) -> UOp:
  hbufs, addrs = [], []
  for dev in _devices(buf_node.src[1].arg):
    dctx = ctx.dev_ctx[dev]
    isz = dctx.kernargs_host.dtype.base.itemsize
    off = dctx.kernargs_allocator.alloc(buf_node.arg, 16)
    hbufs.append(UOp(Ops.BUFFER_VIEW, dctx.kernargs_host.dtype, src=(dctx.kernargs_host,), arg=(buf_node.arg // isz, off // isz)))
    addrs.append(dctx.kernargs_gpu + off)
  return _maybe_mstack(tuple(addrs)).after(*_lower_stores(_maybe_mstack(tuple(hbufs)), buf_node, target.src[1:]))

def bufferize_cmdbuf(ctx:HCQ2LowerCtx, target:UOp, buf_node:UOp) -> UOp:
  hbufs = tuple(UOp.from_buffer(Buffer("CPU", buf_node.arg // dtypes.uint32.itemsize, dtypes.uint32,
                                       options=BufferSpec(cpu_access=True, nolru=True), preallocate=True), dev)
                for dev in _devices(buf_node.src[1].arg))
  hbuf = _maybe_mstack(hbufs)
  return hbuf.after(*_lower_stores(hbuf, buf_node, target.src[1:]), tag=buf_node.tag)

def bufferize_binary(ctx:HCQ2LowerCtx, target:UOp, buf_node:UOp) -> UOp|None:
  if buf_node.tag == "program": return bufferize_program(ctx, target, buf_node)
  if buf_node.tag == "kernargs": return bufferize_kernargs(ctx, target, buf_node)
  if buf_node.tag in ("compute", "copy"): return bufferize_cmdbuf(ctx, target, buf_node)
  return None

# TODO: merge with bufferize_binary
def resolve_buffer(b:UOp) -> UOp|None:
  devs = _devices(b.src[1].arg)
  if b.tag in ("timeline_signal", "timeline_value"):
    return _maybe_mstack(tuple(UOp.from_buffer(getattr(Device[d], b.tag), d) for d in devs), b.tag)
  if b.tag == "scratch":
    return _maybe_mstack(tuple(UOp.from_buffer(Buffer(d, (s:=Device[d].scratch).size, dtypes.uint8, opaque=s, options=BufferSpec(external_ptr=1)), d)
                               for d in devs), b.tag)
  if isinstance(b.tag, tuple): # (compute_queue|sdma_queue, ring|write_ptr|doorbell|put_value)
    return _maybe_mstack(tuple(UOp.from_buffer(getattr(Device[d].compute_queue if b.tag[0] == "compute_queue" else Device[d].sdma_queue(0), b.tag[1]), d)
                            for d in devs), b.tag)
  if isinstance(b.device, tuple): return _maybe_mstack(tuple(UOp.from_buffer(buf, buf.device) for buf in b.buffer.bufs))
  return None

pm_bufferize = PatternMatcher([
  (UPat(Ops.AFTER, src=(UPat(Ops.BUFFER, name="buf_node"),), allow_any_len=True, name="target"), bufferize_binary),
  (UPat(Ops.BUFFER, name="b"), resolve_buffer), # TODO: cleanup
])

def lift_patches_to_cmdbuf(ctx:HCQ2LowerCtx, cmdbuf:UOp) -> UOp|None:
  if cmdbuf.tag not in ("compute", "copy"): return None
  patches = dedup(u for store in cmdbuf.src[1:] for u in store.toposort() if u.op is Ops.AFTER)
  deps = tuple(d for p in patches for d in p.src[1:])
  return cmdbuf.replace(src=cmdbuf.src+deps, tag=None).substitute({p:p.src[0] for p in patches})
pm_lift_patches_to_cmdbuf = PatternMatcher([(UPat(Ops.AFTER, name="cmdbuf", allow_any_len=True), lift_patches_to_cmdbuf)])

# resolve patches

def fold_const_store(ctx:HCQ2LowerCtx, buf:UOp, off:UOp, val:UOp) -> UOp:
  bufs = buf.src if buf.op is Ops.MSTACK else (buf,)
  vals = val.src if val.op is Ops.MSTACK else (val,) * len(bufs)
  for b, v in zip(bufs, vals):
    struct.pack_into(f'<{v.dtype.fmt}', b.buffer.ensure_allocated()._buf.cpu_view().mv.cast('B'), off.arg * b.dtype.base.itemsize, v.arg)
  return UOp(Ops.NOOP)

def fold_blob_store(ctx:HCQ2LowerCtx, buf:UOp, blob:UOp) -> UOp:
  for b in (buf.src if buf.op is Ops.MSTACK else (buf,)):
    b.buffer.ensure_allocated()._buf.cpu_view().mv.cast('B')[:len(blob.arg)] = blob.arg
  return UOp(Ops.NOOP)

def _new_addr_table(n:int) -> UOp:
  return _maybe_mstack(tuple(UOp.from_buffer(Buffer("CPU", 256, dtypes.uint64, preallocate=True), "CPU") for _ in range(n)))

def resolve_getaddr(ctx:HCQ2LowerCtx, m:UOp) -> UOp:
  srcs = m.src if m.op is Ops.MSTACK else (m,)
  for s in srcs:
    if s.op in (Ops.BUFFER, Ops.BUFFER_VIEW) and s not in ctx.holds: ctx.holds.append(s)
  addrs = [s.arg if s.op is Ops.CONST else s.buffer.get_buf(s.device).va_addr for s in srcs]

  # fast-path: all per-dev VAs equal -> just a const
  if all(v == addrs[0] for v in addrs): return UOp.const(dtypes.uint64, addrs[0])

  if ctx.addr_table is None: ctx.addr_table = _new_addr_table(len(srcs)) # TODO: move
  table, slot_const = ctx.addr_table, UOp.const(dtypes.int, (slot:=ctx.next_slot))
  ctx.next_slot = slot + 1

  patch = table.index(slot_const, dtype=table.dtype.ptr()).store(_maybe_mstack(tuple(UOp.const(dtypes.uint64, va) for va in addrs)))
  return table.after(patch).index(slot_const, dtype=table.dtype.ptr()).load(dtype=dtypes.uint64)

pm_resolve_patches = symbolic + PatternMatcher([
  # resolve getaddrs
  (UPat(Ops.GETADDR, src=(UPat(Ops.BUFFER_VIEW, name="bv"),)), # getaddr(buffer_view(x)) -> offset+getaddr(x)
    lambda ctx, bv: UOp(Ops.GETADDR, dtypes.uint64, src=(bv.src[0],)) + UOp.const(dtypes.uint64, bv.arg[1] * bv.dtype.itemsize)),
  (UPat(Ops.GETADDR, src=(UPat((Ops.BUFFER, Ops.MSTACK), name="m"),)), resolve_getaddr), # getaddr(buffer|mstack) -> addr_table load|const
  (UPat(Ops.GETADDR, src=(UPat.cvar("const"),)), lambda ctx, const: const), # getaddr(const) -> const

  # write consts and binaries directly into the buffer (BUFFER or MSTACK of BUFFERs)
  (UPat((Ops.BUFFER, Ops.BUFFER_VIEW, Ops.MSTACK), name="buf").store(UPat(Ops.BINARY, name="blob")), fold_blob_store),
  (UPat((Ops.BUFFER, Ops.BUFFER_VIEW, Ops.MSTACK), name="buf").index(UPat.cvar("off")).or_casted()
    .store(UPat.any(UPat.cvar("val"), UPat(Ops.MSTACK, src=UPat.cvar(), name="val"))), fold_const_store),
])

def parametrize_host_buffer(ctx:HCQ2LowerCtx, buf:UOp) -> UOp:
  # register a host buffer as a launcher input and return its placeholder
  if buf.op is Ops.AFTER:
    p = parametrize_host_buffer(ctx, buf.src[0])
    return p.after(*(s.substitute({buf.src[0]: p}) for s in buf.src[1:]))
  if (b:=buf.buffer) not in ctx.inputs: ctx.inputs.append(b)
  return UOp.placeholder((b.size,), b.dtype, ctx.inputs.index(b))

pm_parametrize_host_buffers = PatternMatcher([
  # resolve buffer views to parametrize only root buffers
  (UPat(Ops.INDEX, src=(UPat(Ops.BUFFER_VIEW, name="bv"), UPat.var("idx")), name="bi"),
    lambda bv, idx, bi: bi.replace(src=(bv.src[0], idx + bv.arg[1]))),

  # parametrize host buffers
  (UPat(Ops.AFTER, src=(UPat((Ops.BUFFER, Ops.BUFFER_VIEW, Ops.MSTACK)),), allow_any_len=True, name="buf"), parametrize_host_buffer),
  (UPat((Ops.BUFFER, Ops.BUFFER_VIEW, Ops.MSTACK), name="buf"), parametrize_host_buffer),

  # remove UNIQUE/DEVICE to dedup CONST
  (UPat(Ops.CONST, name="c"), lambda c: c.replace(src=()) if len(c.src) else None),
])

def hcq_callify(ctx:HCQ2LowerCtx, l:UOp) -> UOp:
  sink = UOp.sink(*l.src, arg=KernelInfo(name=ctx.name, estimates=Estimates()), tag=1)
  inputs = [UOp.from_buffer(b, tuple(x.device for x in b.bufs) if isinstance(b, MultiBuffer) else "CPU") for b in ctx.inputs]
  call = to_program(sink, Device["CPU"].renderer).call(*inputs)
  return call.replace(src=call.src + (UOp(Ops.BIND, dtypes.void, src=tuple(ctx.holds)),)) if ctx.holds else call
pm_callify = PatternMatcher([(UPat(Ops.LINEAR, name="l", allow_any_len=True), hcq_callify)])

# **************** schedule ****************

@track_rewrites(name=lambda linear,ast,**kw: f"hcq schedule {getattr(ast.arg, 'name', ast.op.name.lower())}")
def hcq_schedule(linear:UOp, ast:UOp) -> UOp:
  # runtime preparation: device-specific program, kernargs for each program
  linear = graph_rewrite(linear, pm_prep_runtime, name="hcq: prepare runtime")

  # lower ops into hcq style per-device operations
  linear = graph_rewrite(linear, pm_lower_ops, name="hcq: lower ops")

  # split ops into logical queues
  linear = graph_rewrite(linear, pm_split_into_queues, name="hcq: split into queues")

  # runtime-specific lowering
  linear = graph_rewrite(linear, pm_add_barriers, walk=True, name="hcq: add barriers")
  linear = graph_rewrite(linear, pm_add_signals, walk=True, name="hcq: add signals")
  linear = graph_rewrite(linear, pm_add_timeline_inc, walk=True, name="hcq: add submit")

  # encode cmdbuffers + submits
  # TODO: remove dev
  dev = Device["AMD"]
  return graph_rewrite(linear, dev.pm_lower, walk=True, name="hcq: encode cmdbuf")

@track_rewrites(name=lambda ctx,linear,ast,**kw: f"hcq realize {getattr(ast.arg, 'name', ast.op.name.lower())}")
def hcq_realize(ctx:HCQ2LowerCtx, linear:UOp, ast:UOp) -> UOp:
  # allocate lowering structs
  graph_rewrite(linear, pm_calc_kernargs_sizes, ctx=(sizes:={}), name=None)

  for dev_name, sz in sizes.items():
    dev = Device[dev_name]
    off = dev.kernargs_offset_allocator.alloc(sz, 16)
    ctx.dev_ctx[dev_name] = HCQ2DeviceCtx(dev_name, UOp.from_buffer(dev.kernargs_buf.view(sz, dtypes.uint8, off), dev_name),
                                          UOp.const(dtypes.uint64, dev.kernargs_buf.get_buf(dev_name).va_addr + off, device=dev_name))

  linear = graph_rewrite(linear, pm_bufferize, ctx=ctx, bottom_up=True, name="realize binaries")
  linear = graph_rewrite(linear, pm_lift_patches_to_cmdbuf, ctx=ctx, bottom_up=False, name="lift patches to cmdbuf")
  linear = graph_rewrite(linear, pm_resolve_patches, ctx=ctx, bottom_up=False, name="simplify patches")
  linear = graph_rewrite(linear, pm_parametrize_host_buffers, ctx=ctx, bottom_up=True, name="parametrize host buffers")
  return graph_rewrite(linear, pm_callify, ctx=ctx, name="hcq: callify")

def ensure_accessible(ctx:HCQ2LowerCtx, call:UOp, copy:UOp) -> UOp|None:
  src_buf = call.src[2].buffer # TODO: cleanup
  dev = call.src[1].buffer.device
  try: src_buf.get_buf(dev)
  except Exception:
    (cpubuf := Buffer("CPU", src_buf.nbytes, dtypes.uint8, preallocate=True)).copyin(src_buf.ensure_allocated().as_memoryview())
    ctx.holds.append(buf_uop:=UOp.from_buffer(cpubuf, dev))
    return call.replace(src=call.src[:2] + (buf_uop,) + call.src[3:])
pm_ensure_bufs_accessible = PatternMatcher([(UPat(Ops.CALL, src=(UPat(Ops.COPY, name="copy"),), name="call", allow_any_len=True), ensure_accessible)])

def hcq_exec(ctx:ExecContext, call:UOp, ast:UOp) -> float|None:
  from tinygrad.engine.realize import run_linear

  if ast.src[1].arg.split(":")[0] != "AMD": return None

  # TODO: this mess should gone
  resolved_call = call.replace(src=(ast,) + tuple(resolve_params(call, ctx.input_uops)) + tuple(s for s in call.src[1:] if s.op is Ops.BIND))
  bufs = [cast(Buffer, resolved_call.src[1+gi].buffer) for gi in ast.arg.globals] if ast.op is Ops.PROGRAM \
    else [cast(Buffer, resolved_call.src[i].buffer) for i in range(1, len(resolved_call.src))]
  hcq_ctx = HCQ2LowerCtx(name="submit")
  linear = graph_rewrite(UOp(Ops.LINEAR, dtypes.void, (resolved_call,)), pm_ensure_bufs_accessible, ctx=hcq_ctx)

  linear = hcq_schedule(linear, ast)
  host_call = hcq_realize(hcq_ctx, linear, ast)

  dev = Device["AMD"]
  with track_stats(ctx, call, dev.device, bufs, ctx.var_vals) as tm:
    st = time.perf_counter() if ctx.wait else 0.0
    run_linear(UOp(Ops.LINEAR, dtypes.void, (host_call,)), var_vals=ctx.var_vals, jit=True, update_stats=DEBUG>=3)
    if ctx.wait:
      dev.synchronize()
      tm[0] = time.perf_counter() - st
  return tm[0] if tm[0] is not None else 0.0

pm_hcq_exec = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat({Ops.PROGRAM, Ops.COPY}, name="ast"),), name="call", allow_any_len=True), hcq_exec),
])
