import inspect, functools
from tinygrad.device import Compiled, Allocator, ProfileGraphEntry, ProfileGraphEvent
from tinygrad.engine.jit import MultiGraphRunner
from tinygrad.renderer import Renderer, cstyle, nir, ptx, llvmir, wgsl
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.uop.ops import UOp, Ops
from tinygrad.helpers import cpu_profile, getenv, dedup, NULL_ALLOW_COPYOUT, PROFILE, cpu_events, perf_counter_us

class NullRenderer(CStyleLanguage):
  has_local = False
  float4 = "float4"
  barrier = "// BARRIER"
  code_for_op = {**CStyleLanguage.code_for_op, Ops.THREEFRY: lambda a,b,dtype: f"threefry({a},{b})", Ops.MAX: lambda a,b,dtype: f"max({a},{b})"}
  def asm(self, prg: UOp, lin: UOp) -> bytes:
    assert self.target.arch.startswith("gfx"), "only amd supports assembly"
    from tinygrad.renderer.amd.elf import assemble_linear
    return assemble_linear(prg, lin, self.target.arch)

class NullProgram:
  def __init__(self, device:str, name:str, lib:bytes, *args, **kwargs): self.device, self.name = device, name
  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
    with cpu_profile(self.name, self.device): return 1e-3

class NullAllocator(Allocator['NullDevice']):
  def _alloc(self, size, options): pass
  def _copyin(self, dest, src:memoryview): pass
  def _copyout(self, dest:memoryview, src):
    if not NULL_ALLOW_COPYOUT: raise RuntimeError("no copyout on NULL")
  def _transfer(self, dest, src, sz:int, src_dev, dest_dev):
    with cpu_profile(f"{src_dev.device} -> {dest_dev.device}", f"{src_dev.device}:SDMA:0"): pass
  def _offset(self, buf, offset:int, size:int): pass

class NullGraph(MultiGraphRunner):
  def __call__(self, input_uops:tuple[UOp, ...], var_vals:dict[str, int], wait=False) -> float|None:
    # description based on command, copied from HCQ graph
    if PROFILE: cpu_events.append(ProfileGraphEvent(ents:=[ProfileGraphEntry(runtime.device if runtime is not None else f"{bufs[1].device}:SDMA:0", \
        runtime.name if runtime is not None else f"{bufs[1].device} -> {bufs[0].device}", i, i+1) \
        for i,((_,_,bufs,_),runtime) in enumerate(zip(self.calls, self.runtimes))], [], [perf_counter_us() for _ in range(len(ents)+1)]))
    return 1e-1

class NullDevice(Compiled):
  def __init__(self, device:str):
    assert (emu:=getenv("EMULATE", "")) == "", \
      "EMULATE is deprecated, use DEV=NULL:HIP:"+{"AMD":"gfx1100", "AMD_RDNA4":"gfx1201", "AMD_CDNA4":"gfx950"}.get(emu, "<arch>")
    renderers = [NullRenderer] + [r for m in [cstyle, nir, ptx, llvmir, wgsl] for r in m.__dict__.values()
                                  if inspect.isclass(r) and issubclass(r, Renderer)]
    super().__init__(device, NullAllocator(self), dedup(renderers), functools.partial(NullProgram, device), NullGraph)
