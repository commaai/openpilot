import os
os.environ["PROFILE"] = "1"
os.environ["PMC"] = "1"

import unittest
import functools, contextlib
import numpy as np
from tinygrad import Tensor, Context, Device
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType
from tinygrad.runtime.ops_amd import ProfilePMCEvent
from tinygrad.engine.realize import get_runner
from tinygrad.viz.serve import unpack_pmc
from extra.sqtt.roc import print_pmc

def copy_kernel(B, A, stride=1):
  n_threads = 32
  assert A.size >= n_threads, f"{A.size} is too small, min size {n_threads}"
  g = UOp.range(A.size//n_threads, 0, AxisType.GLOBAL)
  l = UOp.range(n_threads, 1, AxisType.LOCAL)
  i = g * n_threads + l
  index = (i * stride) % A.size
  return B[index].store(A[index]).sink(arg=KernelInfo(name=f"copy_{A.size}_stride_{stride}", opts_to_apply=()))

def lds_kernel(offset:UOp, size:int, inst:str) -> UOp:
  tid = UOp.range(offset.size, 0, AxisType.LOCAL)
  dst = UOp.placeholder((size,), dtypes.float32, 1, AddrSpace.REG)
  #lds = UOp.placeholder((1024,), dtypes.float32, 2, AddrSpace.LOCAL)
  u = UOp(Ops.CUSTOM, arg='__builtin_amdgcn_s_waitcnt(0);')
  u = UOp(Ops.CUSTOM, arg='__builtin_amdgcn_s_barrier();', src=(u,))
  u = UOp(Ops.CUSTOM, arg='__builtin_amdgcn_sched_barrier(0);', src=(u,))
  u = UOp(Ops.CUSTOM, arg=f'asm volatile("{inst} '+'%0, %1" : "=v"({0}) : "v"({1}));', src=(dst, offset[tid], u))
  return UOp.sink(u, arg=KernelInfo(name="test_lds", opts_to_apply=()))

dev = Device[Device.DEFAULT]

@contextlib.contextmanager
def save_pmc():
  # clear the old traces
  dev.profile_events.clear()
  pmc:list[ProfilePMCEvent] = []
  yield pmc
  for e in dev.profile_events:
    if isinstance(e, ProfilePMCEvent): pmc.append(e)

@unittest.skipIf(dev.device != "AMD", "tests PMC counters on AMD")
class TestPMC(unittest.TestCase):
  @Context(IGNORE_OOB=0)
  def test_copy(self, stride:int=1):
    N = 1 << 25 # ~134MB
    a = Tensor(np.arange(N, dtype=np.uint32)+1).realize()
    b = Tensor(np.zeros(N, dtype=np.uint32)).realize()
    b = Tensor.custom_kernel(b, a, fxn=functools.partial(copy_kernel, stride=stride))[0]
    with save_pmc() as pmc:
      b.realize()
    print_pmc(pmc)
    np.testing.assert_equal(a.numpy(), b.numpy())

  def test_copy_uncoalesced(self): return self.test_copy(stride=17)

  # test with two threads issuing ds_reads at different offsets
  def test_ds_read(self, size=1, inst='ds_read_b32'):
    test_banks = 256
    offsets = [Tensor([0, b*4]) for b in range(1, test_banks)]
    with Context(DEBUG=0): Tensor.realize(*offsets)
    k = Tensor.custom_kernel(offsets[0], fxn=functools.partial(lds_kernel, size=size, inst=inst))[0]
    # sample all kernels
    with save_pmc() as pmc_events:
      runner = get_runner(Device.DEFAULT, k.schedule()[0].ast)
      # TODO: llvm eliminates lds definition from the ELF, is there another way to pin lds size?
      runner._prg.group_segment_size = 1024
      for offset in offsets: runner([offset.uop.buffer])
    # find read offsets that created bank conflicts from the pmc counters
    found:list[Tensor] = []
    for i,e in enumerate(pmc_events):
      pmc = unpack_pmc(e)["rows"]
      # SQ on gfx9, renamed to SQC after gfx10
      val = next(total for name,total,_all_instances in pmc if name in {"SQ_LDS_BANK_CONFLICT", "SQC_LDS_BANK_CONFLICT"})
      if val > 0: found.append(offsets[i])
    print("Found bank conflicts at offsets:", [s.numpy() for s in found])

  def test_ds_read_b64(self): self.test_ds_read(2, 'ds_read_b64')

  def test_ds_read_b128(self): self.test_ds_read(4, 'ds_read_b128')

if __name__ == "__main__":
  unittest.main()
