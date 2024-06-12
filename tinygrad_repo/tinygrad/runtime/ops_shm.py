import os, mmap
try: import _posixshmem    # type: ignore
except Exception: pass
from typing import Callable, Dict
from tinygrad.helpers import DType, OSX
from tinygrad.runtime.lib import RawBufferMapped
from tinygrad.ops import Interpreted, Op, UnaryOps, MovementOps, BufferOps

SHM_CACHE: Dict[str, mmap.mmap] = {}
class RawShmBuffer(RawBufferMapped):
  def __init__(self, size, dtype:DType, device:str):
    device, self.cache_id = device.split(",")[0], None if "," not in device else device.split(",")[1]

    if self.cache_id is not None and self.cache_id in SHM_CACHE: shm = SHM_CACHE[self.cache_id]
    else:
      if OSX:
        with open(f"/tmp/shm_{device}", "w+b") as f:
          f.truncate(size * dtype.itemsize)
          shm = mmap.mmap(f.fileno(), size * dtype.itemsize, flags=mmap.MAP_SHARED)
      else:
        fd = _posixshmem.shm_open(device, os.O_RDWR, 0o600)
        # TODO: these flags are somewhat platform specific, but python doesn't expose the ones we need
        shm = mmap.mmap(fd, size * dtype.itemsize, flags=mmap.MAP_SHARED | 0x2000 | 0x008000)
        shm.madvise(mmap.MADV_HUGEPAGE)    # type: ignore
        os.close(fd)
      if self.cache_id is not None: SHM_CACHE[self.cache_id] = shm

    super().__init__(size, dtype, shm)
  def __del__(self):
    if self.cache_id is None: self._buf.close()
  def _buffer(self): return memoryview(self._buf)

# TODO: is this wrong?
shm_fxn_for_op: Dict[Op, Callable] = { BufferOps.MEM: lambda x: x, UnaryOps.NOOP: lambda x:x, MovementOps.RESHAPE: lambda x,_:x, MovementOps.AS_STRIDED: lambda x,_:x }
ShmBuffer = Interpreted(RawShmBuffer, shm_fxn_for_op, from_underlying=lambda x:x)
