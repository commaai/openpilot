import inspect, functools, itertools
from tinygrad.device import BufferStorage, BufferSpec, Buffer, Compiled, HostAllocator, MMIOInterface, Program, ProfileGraphEntry, ProfileGraphEvent
from tinygrad.renderer import Renderer, cstyle, nir, ptx, llvmir, wgsl
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.uop.ops import UOp, Ops, UPat, PatternMatcher, uopfunc
from tinygrad.dtype import dtypes
from tinygrad.helpers import getenv, dedup, prod, panic, cpu_events, perf_counter_us, NULL_ALLOW_COPYOUT, PROFILE
from tinygrad.engine.realize import get_call_arg_uops, get_call_var_uops
from tinygrad.runtime.support.hcq2 import HWQueue, layout_args, pack_args

class NullRenderer(CStyleLanguage):
  has_local = False
  float4 = "float4"
  barrier = "// BARRIER"
  type_map = {**CStyleLanguage.type_map, **{dt:dt.name for dt in dtypes.fp8s}}
  code_for_op = {**CStyleLanguage.code_for_op, Ops.THREEFRY: lambda a,b,dtype: f"threefry({a},{b})", Ops.MAX: lambda a,b,dtype: f"max({a},{b})"}
  def asm(self, prg: UOp, lin: UOp) -> bytes:
    assert self.target.arch.startswith("gfx"), "only amd supports assembly"
    from tinygrad.renderer.amd.elf import assemble_linear
    return assemble_linear(prg, lin, self.target.arch)

EXEC, COPY, WAIT, STORE, TIMESTAMP = range(5)
null_events:dict[tuple[str, str, bytes|None], int] = {}

@uopfunc
def null_submit(cmdbuf:UOp, doorbell:UOp) -> UOp: return doorbell.index(0).store(cmdbuf.index(0).load()).sink()

class NullQueue(HWQueue):
  def cmd(self, op, *args): self.q(*[a.getaddr(self.devs) if isinstance(a, UOp) else UOp.const(a, dtypes.uint64) for a in (op, *args, 0, 0, 0)][:4])
  def event(self, device:str, name:str, key:bytes|None=None) -> int: return null_events.setdefault((device, name, key), len(null_events))
  def exec(self, call:UOp, prg:UOp):
    args = [a.getaddr(self.devs) for a in get_call_arg_uops(call)] + [v.cast(dtypes.uint64) for v in get_call_var_uops(call, prg)]
    kernargs = UOp(Ops.LINEAR, src=tuple(pack_args(layout_args(args), 8 * max(len(args), 1))), arg="kernargs")
    self.cmd(EXEC, kernargs, len(args), self.event(self.devs[0], prg.src[0].arg.function_name, prg.key))
  def copy(self, dst:UOp, src:UOp, sz:int): self.cmd(COPY, dst, src, self.event(f"{src.device}:SDMA:0", f"{src.device} -> {dst.device}"))
  def wait(self, signal:UOp, value:UOp, eq:bool=False): self.cmd(WAIT, signal, value, int(eq))
  def signal(self, signal:UOp, value:UOp): self.cmd(STORE, signal, value)
  def timestamp(self, signal:UOp): self.cmd(TIMESTAMP, signal.getaddr(self.devs) + UOp.const(8, dtypes.uint64))
  def submit(self, cmdbuf): return null_submit(cmdbuf, UOp.alloc((1,), dtypes.uint8, device=self.devs[0]).rtag("doorbell"))

class NullProgram(Program['NullDevice']):
  def __init__(self, dev, obj): self.streams = [(i, prod(s)) for i, (n, _, _, s) in enumerate(obj.signature) if (n or "").startswith("cmdbuf")]
  def __call__(self, *bufs, **kwargs):
    st, words = perf_counter_us(), [w for i, n in self.streams for w in MMIOInterface(bufs[i], n, fmt='Q')[:]]
    # timestamps are emulated: every exec and copy takes 1us
    for op, addr, done in zip(words[0::4], words[1::4], itertools.accumulate(op in (EXEC, COPY) for op in words[0::4])):
      if op == TIMESTAMP: MMIOInterface(addr, 8, fmt='Q')[0] = int((st + done) * 1000)
    descs = [list(null_events)[event] for op, event in zip(words[0::4], words[3::4]) if op in (EXEC, COPY)] if PROFILE else []
    sigs = [st + sum(x[0] == d[0] for x in descs[:i]) + k for i, d in enumerate(descs) for k in (0, 1)]
    if descs: cpu_events.append(ProfileGraphEvent([ProfileGraphEntry(d, n, 2*i, 2*i+1, k) for i, (d, n, k) in enumerate(descs)], [], sigs))

class NullAllocator(HostAllocator):
  def _alloc(self, size:int, options, va=itertools.count(1 << 40, 1 << 32)) -> BufferStorage: return BufferStorage(next(va))
  def _copyin(self, dest, src:memoryview): pass
  def _copyout(self, dest:memoryview, src): NULL_ALLOW_COPYOUT or panic(RuntimeError, "no copyout on NULL")
  def _map(self, buf:Buffer) -> BufferStorage: return BufferStorage(buf._buf if buf.device.startswith("NULL") else buf.host.addr)

class NullDevice(Compiled):
  pm_encode = PatternMatcher([(UPat(Ops.CALL, src=(UPat.custom_function(f"submit_null_{q}"), UPat()), name="submit"),
                               lambda submit: NullQueue(submit).encode()) for q in ("compute", "copy")])
  host = property(lambda self: self.device)
  timeline = functools.cached_property(lambda self: self.link_buffer(2, dtypes.uint64))
  error_state = functools.cached_property(lambda self: self.link_buffer(1, dtypes.int64))

  def __init__(self, device:str):
    assert (emu:=getenv("EMULATE", "")) == "", \
      "EMULATE is deprecated, use DEV=NULL:HIP:"+{"AMD":"gfx1100", "AMD_RDNA4":"gfx1201", "AMD_CDNA4":"gfx950"}.get(emu, "<arch>")
    renderers = [NullRenderer] + [r for m in [cstyle, nir, ptx, llvmir, wgsl] for r in m.__dict__.values()
                                  if inspect.isclass(r) and issubclass(r, Renderer)]
    super().__init__(device, NullAllocator(self), dedup(renderers), NullProgram)
    Compiled.pm_bufferize += PatternMatcher([ # its memory is fake: every placeholder on it is a link buffer
      (UPat(Ops.ALLOC, name="b"), lambda b, d=self: d.link_buffer(b.max_numel(), b.dtype) if b.device == d.device else None)])

  def link_buffer(self, n, dt):
    return Buffer(self.device, n * dt.itemsize, opaque=memoryview(bytearray(n * dt.itemsize)), options=BufferSpec(external_ptr=1))
