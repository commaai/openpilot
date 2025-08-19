import numpy as np
from tinygrad.helpers import flat_mv
from tinygrad.device import Compiled, Allocator

class NpyAllocator(Allocator['NpyDevice']):
  def _alloc(self, size:int, options=None) -> np.ndarray: return np.empty(size, dtype=np.uint8)
  def _as_buffer(self, src:np.ndarray) -> memoryview: return flat_mv(np.require(src, requirements='C').data)
  def _copyout(self, dest:memoryview, src:np.ndarray): dest[:] = self._as_buffer(src)

class NpyDevice(Compiled):
  def __init__(self, device:str): super().__init__(device, NpyAllocator(self), None, None, None)
