import unittest
from tinygrad import Tensor
from tinygrad.helpers import getenv, GlobalCounters
from tinygrad.engine.realize import lower_schedule_item, ProgramSpec, get_program
from tinygrad.renderer import Estimates
from tinygrad.codegen import full_rewrite
from tinygrad.uop.ops import Ops, UOp
from tinygrad.dtype import dtypes
from tinygrad.codegen.opt.kernel import Opt, OptOps, KernelOptError
from tinygrad.device import Device

def flops_mem(uops, ignore_indexing=False):
  est = Estimates.from_uops(uops, ignore_indexing)
  return est.ops, est.lds

# **************** new FlopCounter ****************

def get_stats(x:Tensor):
  si = x.schedule()[-1]
  ei = lower_schedule_item(si)
  return ei.prg.estimates.ops, ei.prg.estimates.mem

class TestMemoryCount(unittest.TestCase):
  def test_add(self):
    a = Tensor.empty(1024, 1024, dtype=dtypes.uint8)
    b = Tensor.empty(1024, 1024, dtype=dtypes.uint8)
    _, mem = get_stats(a+b)
    self.assertEqual(mem, 1024*1024*3)  # 2 reads + 1 write

  def test_add_const(self):
    a = Tensor.empty(1024, 1024, dtype=dtypes.uint8)
    _, mem = get_stats(a+3)
    self.assertEqual(mem, 1024*1024*2)  # 1 read + 1 write

  @unittest.skip("depends on subbuffer working")
  def test_add_slice(self):
    a = Tensor.empty(1024, 1024, dtype=dtypes.uint8)[:512]
    _, mem = get_stats(a+3)
    self.assertEqual(mem, 512*1024*2)  # 1 read + 1 write

  def test_expanded(self):
    a = Tensor.empty(1024, 1, dtype=dtypes.uint8).expand(1024, 1024)
    b = Tensor.empty(1024, 1024, dtype=dtypes.uint8)
    _, mem = get_stats(a+b)
    self.assertEqual(mem, 1024*1024*2 + 1024)  # 1 full read + 1 lil read + 1 write

  def test_both_expanded(self):
    # TODO: this probably should be a full write
    a = Tensor.empty(1024, 1, dtype=dtypes.uint8).expand(1024, 1024)
    b = Tensor.empty(1024, 1, dtype=dtypes.uint8).expand(1024, 1024)
    _, mem = get_stats(a+b)
    self.assertEqual(mem, 1024*1024 + 2*1024)  # 2 lil reads + 1 write

  def test_self_add(self):
    a = Tensor.empty(1024, 1024, dtype=dtypes.uint8)
    _, mem = get_stats(a+a)
    self.assertEqual(mem, 1024*1024*2)  # 1 read + 1 write

  def test_self_add_transposed(self):
    a = Tensor.empty(1024, 1024, dtype=dtypes.uint8)
    _, mem = get_stats(a+a.T)
    self.assertEqual(mem, 1024*1024*2)  # 1 read + 1 write

  def test_self_add_assign(self):
    a = Tensor.empty(1024, 1024, dtype=dtypes.uint8).realize()
    _, mem = get_stats(a.assign(a+a))
    self.assertEqual(mem, 1024*1024*2)  # 1 read + 1 write

  @unittest.skipIf(Device.DEFAULT == "CPU", "test copy to CPU from other device")
  def test_copyout(self):
    a = Tensor.empty(32, dtype=dtypes.uint8).to("CPU")
    _, mem = get_stats(a)
    self.assertEqual(mem, 32*1)
    a = Tensor.empty(32, dtype=dtypes.uint32).to("CPU")
    _, mem = get_stats(a)
    self.assertEqual(mem, 32*4)

# NOTE: this still isn't testing unroll using the acc
@unittest.skipUnless(getenv("PYTHON"), "only run test on emulated tensor cores")
class TestUOpsStatsMatmulHalf(unittest.TestCase):
  def test_simple_matmul_half(self, N=16):
    GlobalCounters.reset()
    a, b = Tensor.empty(N, N, dtype=dtypes.half), Tensor.empty(N, N, dtype=dtypes.half)
    c = a.matmul(b)
    c.realize()
    expected_ops = N ** 3 * 2
    self.assertEqual(expected_ops, GlobalCounters.global_ops)

  def test_bigger_matmul_half(self): self.test_simple_matmul_half(64)

  def test_batched_matmul_half(self, N=16):
    GlobalCounters.reset()
    a, b = Tensor.empty(4, N, N, dtype=dtypes.half), Tensor.empty(1, N, N, dtype=dtypes.half)
    c = a.matmul(b)
    c.realize()
    expected_ops = 4 * N ** 3 * 2
    self.assertEqual(expected_ops, GlobalCounters.global_ops)

class TestUOpsStats(unittest.TestCase):
  @unittest.skipIf(getenv("PTX"), "wrong in PTX")
  def test_simple_add(self):
    a = Tensor.empty(100,100)
    b = Tensor.empty(100,100)
    c = a+b
    ops, mem = get_stats(c)
    expected_ops = c.numel()
    expected_mem = a.nbytes() + b.nbytes() + c.nbytes()
    self.assertEqual(mem, expected_mem)
    # NOTE; ops also include indexing ops
    assert expected_ops <= ops and ops <= expected_ops * 2

  @unittest.skipIf(getenv("PTX"), "wrong in PTX")
  def test_simple_add_sq(self):
    a = Tensor.empty(100,100)
    b = Tensor.empty(100,100)
    c = (a+b)*(a+b)
    ops, mem = get_stats(c)
    expected_ops = c.numel()*2
    expected_mem = a.nbytes() + b.nbytes() + c.nbytes()
    self.assertEqual(mem, expected_mem)
    # NOTE; ops also include indexing ops
    assert expected_ops <= ops and ops <= expected_ops * 2

  def test_simple_matmul(self, M=1024, N=1024, K=1024):
    a = Tensor.empty(M,N)
    b = Tensor.empty(N,K)
    c = a@b
    ops, mem = get_stats(c)
    expected_ops = c.numel() * N * 2
    required_mem = a.nbytes() + b.nbytes() + c.nbytes()
    assert expected_ops <= ops and ops <= expected_ops * 1.2
    # NOTE: it's hard to assert on the memory here, all depends on caching
    assert required_mem <= mem

  def test_simple_matmul_8192(self): self.test_simple_matmul(8192, 8192, 8192)

  #MULACC should have the same stats as MUL + ADD
  def test_mulacc(self):
    globl = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), tuple())
    o1 = UOp(Ops.CONST, dtypes.int, tuple(), 1)
    o2 = UOp(Ops.CONST, dtypes.int, tuple(), 2)
    u1 = UOp(Ops.LOAD, dtypes.int, (globl.index(o1),))
    u2 = UOp(Ops.LOAD, dtypes.int, (globl.index(o2),))
    u3 = UOp(Ops.CONST, dtypes.int, tuple(), 3)
    u4 = UOp(Ops.MUL, dtypes.int, (u1,u2))
    u5 = UOp(Ops.ADD, dtypes.int, (u4,u3))
    uops = full_rewrite(u5.sink())

    globl = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), tuple())
    o1 = UOp(Ops.CONST, dtypes.int, tuple(), 1)
    o2 = UOp(Ops.CONST, dtypes.int, tuple(), 2)
    u1 = UOp(Ops.LOAD, dtypes.int, (globl.index(o1),))
    u2 = UOp(Ops.LOAD, dtypes.int, (globl.index(o2),))
    u3 = UOp(Ops.CONST, dtypes.int, tuple(), 3)
    u4 = UOp(Ops.MULACC, dtypes.int, (u1,u2,u3))
    uops_fma = full_rewrite(u4.sink())

    self.assertEqual(flops_mem(uops), flops_mem(uops_fma))

N = 64
@unittest.skipIf(getenv("PTX"), "wrong in PTX") # maybe?
class TestStatsOptimized(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.ast_gemm = (Tensor.empty(N, N) @ Tensor.empty(N, N)).schedule()[-1].ast
    cls.ast_reduce = (Tensor.empty(N*N).sum()).schedule()[-1].ast

  def check_gemm(self, p:ProgramSpec, extra_flops=0):
    #p.uops.print()
    #print(p.src)
    print(p.name, p.estimates.ops, p.estimates.mem, p.estimates.lds)
    self.assertEqual(p.estimates.ops, 2*N*N*N + extra_flops)  # N**3 mulaccs
    self.assertEqual(p.estimates.mem, 3*N*N*4) # 3 NxN mats with floats

  def test_gemm(self):
    p = get_program(self.ast_gemm, opts=[])
    self.check_gemm(p)
    self.assertEqual(p.estimates.lds, 2*N*N*N*4 + 4*N*N)

  def test_gemm_tc_unroll(self):
    try:
      p = get_program(self.ast_gemm, opts=[Opt(OptOps.TC, 0, (-1, 0, 1)), Opt(OptOps.UNROLL, 0, 2)])
    except KernelOptError:
      raise unittest.SkipTest("no tensor cores")
    print(p.src)
    self.check_gemm(p)

  # this is a good lesson about why UPCASTing is a good idea

  def test_gemm_one_upcasted(self):
    p = get_program(self.ast_gemm, opts=[Opt(OptOps.UPCAST, 0, 4)])
    self.check_gemm(p)
    self.assertEqual(p.estimates.lds, N*N*N*4 + N*N*N*4//4 + 4*N*N)

  def test_gemm_upcasted(self):
    p = get_program(self.ast_gemm, opts=[Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UNROLL, 0, 4)])
    self.check_gemm(p)
    self.assertEqual(p.estimates.lds, 2*N*N*N*4//4 + 4*N*N)

  def test_gemm_upcasted_locals(self):
    try:
      p = get_program(self.ast_gemm, opts=[Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 4),
                                           Opt(OptOps.LOCAL, 0, 4),  Opt(OptOps.LOCAL, 1, 4)])
    except KernelOptError:
      raise unittest.SkipTest("no locals")
    self.check_gemm(p)
    self.assertEqual(p.estimates.lds, 2*N*N*N*4//4 + 4*N*N)

  def test_gemm_group(self):
    try:
      p = get_program(self.ast_gemm, opts=[Opt(OptOps.GROUP, 0, 4)])
    except KernelOptError:
      raise unittest.SkipTest("no locals")
    SZ = N*N*4
    # NOTE: these are sort of wrong. they aren't honoring the IF statement
    self.check_gemm(p, extra_flops=SZ*4)
    self.assertEqual(p.estimates.lds, 2*N*N*N*4 + SZ*4 + (SZ*4 + 4*N*N)*4)

  def test_reduce(self):
    p = get_program(self.ast_reduce, opts=[])
    print(p.name, p.estimates.ops, p.estimates.mem, p.estimates.lds)
    self.assertEqual(p.estimates.ops, N*N)
    self.assertEqual(p.estimates.mem, N*N*4 + 4)

  def test_reduce_group(self):
    try:
      p = get_program(self.ast_reduce, opts=[Opt(OptOps.GROUP, 0, 50)])
    except KernelOptError:
      raise unittest.SkipTest("no locals")
    # NOTE: these are wrong, they don't respect the if statement
    print(p.name, p.estimates.ops, p.estimates.mem, p.estimates.lds)

if __name__ == '__main__':
  unittest.main(verbosity=2)
