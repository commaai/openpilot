import functools
from tinygrad.device import Compiled, Compiler, Allocator
from tinygrad.engine.jit import MultiGraphRunner
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.uop.ops import Ops
from tinygrad.helpers import cpu_profile

class NullRenderer(CStyleLanguage):
  device = "NULL"
  has_local = False
  float4 = "float4"
  barrier = "// BARRIER"
  code_for_op = {**CStyleLanguage.code_for_op, Ops.THREEFRY: lambda a,b,dtype: f"threefry({a},{b})", Ops.MAX: lambda a,b,dtype: f"max({a},{b})"}

class NullProgram:
  def __init__(self, device:str, name:str, lib:bytes): self.device, self.name = device, name
  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False):
    with cpu_profile(self.name, self.device): return 1e-4

class NullAllocator(Allocator['NullDevice']):
  def _alloc(self, size, options): pass
  def _copyin(self, dest, src:memoryview): pass
  def _copyout(self, dest:memoryview, src): pass
  def _transfer(self, dest, src, sz:int, src_dev, dest_dev):
    with cpu_profile(f"{src_dev.device} -> {dest_dev.device}", self.dev.device): pass
  def _offset(self, buf, offset:int, size:int): pass

class NullGraph(MultiGraphRunner):
  def __call__(self, input_rawbuffers, var_vals, wait=False) -> float|None: return 1e-3

class NullDevice(Compiled):
  def __init__(self, device:str): super().__init__(device, NullAllocator(self), [(NullRenderer, Compiler)], functools.partial(NullProgram, device),
                                                   NullGraph)
