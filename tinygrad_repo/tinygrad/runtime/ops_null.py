from tinygrad.device import Compiled, Compiler, Renderer, Allocator

class NullRenderer(Renderer):
  def render(self, uops:list) -> str: return ""

class NullProgram:
  def __init__(self, name:str, lib:bytes): pass
  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False):
    return 1e-4

class NullAllocator(Allocator):
  def _alloc(self, size, options): pass
  def _copyin(self, dest, src:memoryview): pass
  def _copyout(self, dest:memoryview, src): pass

class NullDevice(Compiled):
  def __init__(self, device:str): super().__init__(device, NullAllocator(), NullRenderer(), Compiler(), NullProgram)
