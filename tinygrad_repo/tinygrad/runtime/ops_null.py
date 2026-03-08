import functools
from tinygrad.device import Compiled, Compiler, Allocator, CompilerSet, CompilerPair
from tinygrad.engine.jit import MultiGraphRunner
from tinygrad.renderer.cstyle import Renderer, CStyleLanguage, AMDHIPRenderer
from tinygrad.uop.ops import Ops
from tinygrad.helpers import cpu_profile, EMULATE, NULL_IR3, NULL_NAK, NULL_ALLOW_COPYOUT
from tinygrad.renderer.nir import IR3Renderer, NAKRenderer

class NullRenderer(CStyleLanguage):
  device = "NULL"
  has_local = False
  float4 = "float4"
  barrier = "// BARRIER"
  code_for_op = {**CStyleLanguage.code_for_op, Ops.THREEFRY: lambda a,b,dtype: f"threefry({a},{b})", Ops.MAX: lambda a,b,dtype: f"max({a},{b})"}

class NullProgram:
  def __init__(self, device:str, name:str, lib:bytes, *args, **kwargs): self.device, self.name = device, name
  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False):
    with cpu_profile(self.name, self.device): return 1e-3

class NullAllocator(Allocator['NullDevice']):
  def _alloc(self, size, options): pass
  def _copyin(self, dest, src:memoryview): pass
  def _copyout(self, dest:memoryview, src):
    if not NULL_ALLOW_COPYOUT: raise RuntimeError("no copyout on NULL")
  def _transfer(self, dest, src, sz:int, src_dev, dest_dev):
    with cpu_profile(f"{src_dev.device} -> {dest_dev.device}", self.dev.device): pass
  def _offset(self, buf, offset:int, size:int): pass

class NullGraph(MultiGraphRunner):
  def __call__(self, input_buffers, var_vals, wait=False) -> float|None: return 1e-1

class NullDevice(Compiled):
  def __init__(self, device:str):
    renderer:functools.partial|type[Renderer]
    match str(EMULATE.value):
      case "AMD": renderer = functools.partial(AMDHIPRenderer, "gfx1100")
      case "AMD_RDNA4": renderer = functools.partial(AMDHIPRenderer, "gfx1201")
      case "AMD_CDNA4": renderer = functools.partial(AMDHIPRenderer, "gfx950")
      case "": renderer = NullRenderer
      case _: raise RuntimeError(f"can't EMULATE device: {EMULATE.value}")
    compilers = CompilerSet([CompilerPair(renderer, Compiler), CompilerPair(functools.partial(IR3Renderer, 0x6030001), None, NULL_IR3), # adreno 630
                             CompilerPair(functools.partial(NAKRenderer, "sm_120", 48), None, NULL_NAK)]) # 5090
    super().__init__(device, NullAllocator(self), compilers, functools.partial(NullProgram, device), NullGraph)
