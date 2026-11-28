import unittest, functools
from tinygrad import Tensor, UOp, GlobalCounters, Context, Device, Variable, TinyJit
from tinygrad.codegen.compile import compile_linear
from tinygrad.codegen import to_program
import numpy as np
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.uop.ops import KernelInfo, AxisType, Ops, StageOpts
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.renderer import TinyELF
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.renderer.llvmir import LLVMRenderer
from test.helpers import assert_kernel_count, run_schedule
from test.null.test_custom_kernel import custom_elementwise_add_kernel, custom_elementwise_addmul_kernel, custom_gemm

# **** kernels ****

def custom_arange_kernel(C:UOp) -> UOp:
  i = UOp.range(C.shape[0], 0)
  return C[i].store(i.cast(C.dtype)).end(i).sink(arg=KernelInfo(name=f"custom_arange_{C.shape[0]}"))

def custom_eye_kernel(C:UOp) -> UOp:
  i = UOp.range(C.shape[0], 0)
  j = UOp.range(C.shape[1], 1)
  return C[i, j].store((i.eq(j)).cast(C.dtype)).end(i, j).sink(arg=KernelInfo(name=f"custom_eye_{C.numel()}"))

def custom_add_one_kernel(B:UOp, A:UOp) -> UOp:
  A,B = A.flatten(), B.flatten()
  assert B.numel() == A.numel()
  i = UOp.range(A.numel(), 0)
  return B[i].store(A[i] + 1).end(i).sink(arg=KernelInfo(name=f"add_one_{A.numel()}"))

def custom_ignore_first_kernel(C:UOp, A:UOp, B:UOp) -> UOp:
  # A is unused on purpose: the kernel takes call buffers 0 and 2, not 0, 1, 2
  C, B = C.flatten(), B.flatten()
  i = UOp.range(C.numel(), 0)
  return C[i].store(B[i] + 1).end(i).sink(arg=KernelInfo(name=f"ignore_first_{C.numel()}"))

def custom_add_var_kernel(*srcs:UOp, n_slot:int) -> UOp:
  C, *Bs = [s.flatten() for i,s in enumerate(srcs) if i != n_slot]
  i = UOp.range(C.numel(), 0)
  out = functools.reduce(lambda a,b: a+b, [B[i] for B in Bs]) + srcs[n_slot]
  return C[i].store(out).end(i).sink(arg=KernelInfo(name=f"add_var_{n_slot}_{len(Bs)}"))

def run_kernel_args(out:Tensor, name:str) -> list[str|None]:
  linear, externs = out.linear_with_externs()
  sink = [u for u in linear.toposort(enter_calls=True) if u.op is Ops.SINK and isinstance(u.arg, KernelInfo) and u.arg.name == name][0]
  sig = TinyELF.from_program(to_program(sink, Device[Device.DEFAULT].renderer)).signature
  run_schedule(linear, externs)
  return [nm for nm, *_ in sig]

def custom_ignore_first_var_kernel(C:UOp, A:UOp, B:UOp, n:UOp) -> UOp:
  C, B = C.flatten(), B.flatten()
  k = UOp.variable("k", 0, 100, dtype=dtypes.int)
  i = UOp.range(C.numel(), 0)
  return C[i].store(B[i] + n + k).end(i).sink(arg=KernelInfo(name=f"ignore_first_var_{C.numel()}"))

def custom_sum(B:UOp, A:UOp) -> UOp:
  i = UOp.range(A.shape[0], 0)
  B = B[0].set(0.0)
  B = B[0].set(B.after(i)[0] + A[i], end=i)
  return B.sink(arg=KernelInfo(name=f"custom_sum_{A.shape[0]}", opts_to_apply=()))

def slice_sum_kernel(dest:UOp, src:UOp):
  G = UOp.range(src.shape[0], 0, dtype=dtypes.int)
  slice_src = src[G, :]
  reg = UOp.placeholder((1,), dest.dtype, 0, addrspace=AddrSpace.REG)
  reg = reg.after(G)[0].set(0)
  R = UOp.range(src.shape[1], 1)
  reg = reg[0].set(reg.after(R)[0] + slice_src[R], end=R)
  ast = dest[G].set(reg[0], end=G)
  return ast.sink(arg=KernelInfo(name=f"slice_sum_{src.shape[0]}_{src.shape[1]}", opts_to_apply=()))

def simple_qkv_kernel(O:UOp, Q:UOp, K:UOp, V:UOp) -> UOp:
  # attention without softmax
  N, d = Q.shape[0], Q.shape[1]

  i = UOp.range(N, 0)  # output row
  d_out = UOp.range(d, 1)  # output column
  j = UOp.range(N, 2)

  k_inner = UOp.range(d, 3)
  qk_acc = UOp.placeholder((1,), Q.dtype, 0, addrspace=AddrSpace.REG)
  qk_acc = qk_acc.after(i, j)[0].set(0.0)
  qk_acc = qk_acc[0].set(qk_acc.after(k_inner)[0] + Q[i, k_inner] * K[j, k_inner], end=k_inner)
  qk_score = qk_acc[0] / (d ** 0.5)

  out_acc = UOp.placeholder((1,), Q.dtype, 1, addrspace=AddrSpace.REG)
  out_acc = out_acc.after(i, d_out)[0].set(0.0)
  out_acc = out_acc[0].set(out_acc.after(j)[0] + qk_score * V[j, d_out], end=j)

  store = O[i, d_out].store(out_acc[0])
  return store.end(d_out).end(i).sink(arg=KernelInfo(name=f"simple_qkv_{N}_{d}", opts_to_apply=()))

# **** backward callbacks ****

def backward_gemm(gradient:UOp, kernel:UOp) -> tuple[UOp, UOp]:
  out, a, b = kernel.src[1:]
  grad_a = (Tensor(gradient) @ Tensor(b).T).uop
  grad_b = (Tensor(a).T @ Tensor(gradient)).uop
  return (None, grad_a, grad_b)

def backward_gemm_custom(gradient:UOp, kernel:UOp) -> tuple[UOp, UOp]:
  out, a, b = kernel.src[1:]
  grad_a = Tensor.empty_like(Tensor(a)).custom_kernel(Tensor(gradient), Tensor(b).T, fxn=custom_gemm)[0].uop
  grad_b = Tensor.empty_like(Tensor(b)).custom_kernel(Tensor(a).T, Tensor(gradient), fxn=custom_gemm)[0].uop
  return (None, grad_a, grad_b)

# **** tests ****

class TestCustomKernel(unittest.TestCase):
  def test_empty(self):
    a = Tensor.empty(1)
    a = Tensor.custom_kernel(a, fxn=lambda _: UOp.sink(arg=KernelInfo()))[0]
    a.realize()

  def test_multioutput(self):
    a = Tensor.full((16, 16), 3.).contiguous()
    b = Tensor.full((16, 16), 3.).contiguous()
    c = Tensor.empty(16, 16)
    d = Tensor.empty(16, 16)

    c,d = Tensor.custom_kernel(c,d,a,b, fxn=custom_elementwise_addmul_kernel)[:2]
    Tensor.realize(c,d)

    assert all(x == 6 for x in c.flatten().tolist()), "all 6"
    assert all(x == 9 for x in d.flatten().tolist()), "all 9"

  def test_arange(self):
    ref = Tensor.arange(100)
    tst = Tensor.empty_like(ref)
    tst = tst.custom_kernel(fxn=custom_arange_kernel)[0]
    self.assertTrue((ref == tst).all().item())

  def test_eye(self):
    ref = Tensor.eye(1024).clone().realize()
    tst = Tensor.empty_like(ref)
    tst = tst.custom_kernel(fxn=custom_eye_kernel)[0]
    self.assertTrue((ref == tst).all().item())

  def test_noncontig(self):
    a = Tensor.ones(16, 16).contiguous()
    tst = Tensor.empty_like(a)
    b = a+1
    b_p1 = Tensor.custom_kernel(tst, b, fxn=custom_add_one_kernel)[0]
    self.assertTrue((b_p1 == 3).all().item())

  def test_sum_int(self):
    a = Tensor([1, 2, 3, 4, 5])
    tst = Tensor.empty(1, dtype=a.dtype)
    b = Tensor.custom_kernel(tst, a, fxn=custom_sum)[0]
    self.assertEqual(b.item(), 15)

  def test_slice_sum(self):
    A = Tensor.randn(16, 16).contiguous()
    B = Tensor.empty(16)
    B = Tensor.custom_kernel(B, A, fxn=slice_sum_kernel)[0]
    self.assertTrue(B.allclose(A.sum(1)).item())

  def test_gemm(self):
    N = 16
    a = Tensor.randn(N, N)
    b = Tensor.randn(N, N)
    c = Tensor.empty(N, N)

    tst = Tensor.custom_kernel(c, a, b, fxn=custom_gemm)[0]
    self.assertTrue(tst.allclose(a@b, atol=1e-3).item())

  def test_range_merge_keeps_inner_reduce(self):
    # the inner reduce ends r1 alone, so r0 and r1 don't merge. the inner sum is 4, the outer adds it 4*4*4 times
    def kernel(C:UOp, A:UOp) -> UOp:
      r0, r1, r2 = [UOp.range(4, n, AxisType.UPCAST) for n in range(3)]
      return C[0].store(A[r2].reduce(r1, arg=Ops.ADD).reduce(r0, r1, r2, arg=Ops.ADD)).sink(arg=KernelInfo(opts_to_apply=()))
    self.assertEqual(Tensor.custom_kernel(Tensor.empty(1), Tensor.ones(4).contiguous(), fxn=kernel)[0].item(), 256)

  def test_reduce_over_singleton_range(self):
    def kernel(C:UOp, A:UOp) -> UOp:
      return C[0].store(A[1].reduce(UOp.range(1, 0), arg=Ops.ADD)).sink(arg=KernelInfo(opts_to_apply=()))
    self.assertEqual(Tensor.custom_kernel(Tensor.empty(1, dtype=dtypes.int), Tensor([2, 3]).contiguous(), fxn=kernel)[0].item(), 3)

  def test_reduce_axis_and_range(self):
    def kernel(C:UOp, A:UOp) -> UOp: return C[0].store(A.reduce(UOp.range(3, 0), arg=(Ops.ADD, 1))).sink()
    self.assertEqual(Tensor.custom_kernel(Tensor.empty(1, dtype=dtypes.int), Tensor([2, 3]), fxn=kernel)[0].item(), 15)

  def test_reduce_singleton_axis_and_range(self):
    assert Tensor(Tensor([[2], [3]]).uop.reduce(UOp.range(3, 0), arg=(Ops.ADD, 2))).item() == 15

  def test_reduce_as_index(self):
    def kernel(C:UOp, A:UOp, B:UOp) -> UOp:
      k = UOp.range(2, 0)
      idx = A[k].load().reduce(k, arg=Ops.ADD)
      return C[0].store(B[idx].load()).sink(arg=KernelInfo(opts_to_apply=()))
    with Context(CHECK_OOB=0):
      out = Tensor.custom_kernel(Tensor.empty(1, dtype=dtypes.int), Tensor([1, 2]), Tensor([10, 20, 30, 40]), fxn=kernel)[0]
      self.assertEqual(out.item(), 40)

  @unittest.skipIf(not Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  def test_group_reduce_split_range(self):
    # j%2 splits j into two ranges, both are still LOCAL
    def kernel(C:UOp, A:UOp) -> UOp:
      i, j = UOp.range(4, 0), UOp.range(8, 1, AxisType.LOCAL)
      return C[i].store((A[i, j] * (j%2).cast(A.dtype)).reduce(j, arg=Ops.ADD)).end(i).sink(arg=KernelInfo(opts_to_apply=()))
    a = Tensor.arange(32).reshape(4, 8).float().contiguous().realize()
    self.assertEqual(Tensor.custom_kernel(Tensor.empty(4), a, fxn=kernel)[0].tolist(), a[:, 1::2].sum(1).tolist())

  @unittest.skipIf(not Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  def test_nested_group_reduce(self):
    # the inner group's stage is indexed by the outer group's range, which is live at the inner reduce
    def kernel(C:UOp, B:UOp) -> UOp:
      i, g1, g2 = UOp.range(4, 0), UOp.range(4, 1, AxisType.LOCAL), UOp.range(8, 2, AxisType.LOCAL)
      return C[i].store(B[i, g1, g2].reduce(g2, arg=Ops.ADD).reduce(g1, arg=Ops.ADD)).end(i).sink(arg=KernelInfo(opts_to_apply=()))
    b = Tensor.arange(128).reshape(4, 4, 8).float().contiguous().realize()
    self.assertEqual(Tensor.custom_kernel(Tensor.empty(4), b, fxn=kernel)[0].tolist(), b.sum((1, 2)).tolist())

  @unittest.skipIf(not Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  def test_local_reduce(self):
    # a reduce over a thread range combines across threads
    a = Tensor.arange(32).reshape(4, 8).float().contiguous().realize()
    for at in (AxisType.LOCAL, AxisType.WARP):
      def kernel(C:UOp, A:UOp) -> UOp:
        i, j = UOp.range(4, 0), UOp.range(8, 1, at)
        return C[i].store(A[i, j].reduce(j, arg=Ops.ADD)).end(i).sink(arg=KernelInfo(opts_to_apply=()))
      self.assertEqual(Tensor.custom_kernel(Tensor.empty(4), a, fxn=kernel)[0].tolist(), a.sum(1).tolist())

  @unittest.skipIf(not Device[Device.DEFAULT].renderer.has_shared, "LOCAL STAGE needs shared memory")
  def test_stage_then_reduce(self):
    # the STAGE ends j, so the accumulator of the reduce over jj is initialized before the jj loop, not inside it
    def kernel(C:UOp, A:UOp) -> UOp:
      i, j, jj = UOp.range(4, 0), UOp.range(8, 1), UOp.range(8, 2)
      stage = (A[i, j] * 2).stage(j, arg=StageOpts(None, AddrSpace.LOCAL))
      return C[i].store(stage.index(jj).reduce(jj, arg=Ops.ADD)).end(i).sink(arg=KernelInfo(opts_to_apply=()))
    a = Tensor.arange(32).reshape(4, 8).float().contiguous().realize()
    self.assertEqual(Tensor.custom_kernel(Tensor.empty(4), a, fxn=kernel)[0].tolist(), (a*2).sum(1).tolist())

  @unittest.skipIf(not Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  def test_stage_in_thread_range(self):
    # the STAGE is inside the thread range i, so every thread stages its own row
    def kernel(C:UOp, A:UOp) -> UOp:
      i, j, jj = UOp.range(4, 0, AxisType.LOCAL), UOp.range(8, 1), UOp.range(8, 2)
      stage = (A[i, j] * 2).stage(j, arg=StageOpts(None, AddrSpace.LOCAL))
      return C[i].store(stage.index(jj).reduce(jj, arg=Ops.ADD)).end(i).sink(arg=KernelInfo(opts_to_apply=()))
    a = Tensor.arange(32).reshape(4, 8).float().contiguous().realize()
    self.assertEqual(Tensor.custom_kernel(Tensor.empty(4), a, fxn=kernel)[0].tolist(), (a*2).sum(1).tolist())

  @unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, PTXRenderer), "PTX does not support dynamic register indexing")
  def test_reg_stage_then_reduce(self):
    # the REG buffer of the STAGE and the accumulator of the reduce are different buffers
    def kernel(C:UOp, A:UOp) -> UOp:
      i, j, jj = UOp.range(4, 0), UOp.range(8, 1), UOp.range(8, 2)
      stage = (A[i, j] * 2).stage(j, arg=StageOpts(None, AddrSpace.REG))
      return C[i].store(stage.index(jj).reduce(jj, arg=Ops.ADD)).end(i).sink(arg=KernelInfo(opts_to_apply=()))
    a = Tensor.arange(32).reshape(4, 8).float().contiguous().realize()
    self.assertEqual(Tensor.custom_kernel(Tensor.empty(4), a, fxn=kernel)[0].tolist(), (a*2).sum(1).tolist())

  @unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, PTXRenderer), "PTX does not support dynamic register indexing")
  def test_reg_stage_of_padded_value(self):
    # the STAGE writes every lane, the padding included, though the padded value's index gates r (also once r%2 splits r)
    a = Tensor.arange(4, dtype=dtypes.int).contiguous().realize()
    for n, pad, split, want in ((6, (1, 1), False, [3, 5, 2]), (8, (0, 4), True, [1, 4, 3, 6])):
      def kernel(C:UOp, A:UOp) -> UOp:
        r, i = UOp.range(n, 1), UOp.range(n//2, 0)
        v = (A+1).pad((pad,)).index(r)
        s = (v + (r%2).cast(dtypes.int) if split else v).stage(r, arg=StageOpts(None, AddrSpace.REG))
        return C[i].store(s[i] + s[i+n//2]).end(i).sink(arg=KernelInfo(opts_to_apply=()))
      with self.subTest(split=split):
        self.assertEqual(Tensor.custom_kernel(Tensor.empty(n//2, dtype=dtypes.int), a, fxn=kernel)[0].tolist(), want)

  def test_reg_placeholder_then_reduce(self):
    # the accumulator of the reduce does not reuse the slot of a REG placeholder in the kernel
    def kernel(C:UOp, A:UOp) -> UOp:
      i, j = UOp.range(4, 0), UOp.range(8, 1)
      reg = UOp.placeholder((1,), dtypes.float, 0, addrspace=AddrSpace.REG)
      reg = reg.after(i)[0].set(A[i, 0])
      return C[i].store(A[i, j].reduce(j, arg=Ops.ADD) + reg[0]).end(i).sink(arg=KernelInfo(opts_to_apply=()))
    a = Tensor.arange(32).reshape(4, 8).float().contiguous().realize()
    self.assertEqual(Tensor.custom_kernel(Tensor.empty(4), a, fxn=kernel)[0].tolist(), (a.sum(1) + a[:, 0]).tolist())

  def test_explicit_stage_padded_slice(self):
    def kernel(out):
      r = UOp.range(2, 0)
      return out.store(r.expand(2).pad((1, 1)).stage(r, arg=StageOpts(Device.DEFAULT, removable=False))).sink()
    self.assertEqual(Tensor.empty(2, 4, dtype=dtypes.int).custom_kernel(fxn=kernel)[0].tolist(), [[0, 0, 0, 0], [0, 1, 1, 0]])

  def _test_explicit_stage_free_range(self, space:AddrSpace):
    def kernel(out):
      i, j = UOp.range(3, 0), UOp.range(4, 1)
      v = (i*10+j).cast(dtypes.int).stage(j, arg=StageOpts(None, space, removable=False))
      return out[i].store(v.index(2)).end(i).sink()
    self.assertEqual(Tensor.empty(3, dtype=dtypes.int).custom_kernel(fxn=kernel)[0].tolist(), [2, 12, 22])
  @unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, PTXRenderer), "PTX does not support dynamic register indexing")
  def test_explicit_stage_free_range_reg(self): self._test_explicit_stage_free_range(AddrSpace.REG)
  @unittest.skipIf(not Device[Device.DEFAULT].renderer.has_shared, "LOCAL STAGE needs shared memory")
  def test_explicit_stage_free_range_local(self): self._test_explicit_stage_free_range(AddrSpace.LOCAL)

  def _test_explicit_stage_no_range(self, space:AddrSpace):
    def kernel(out):
      i = UOp.range(3, 0)
      return out[i].store((i*10).cast(dtypes.int).stage(arg=StageOpts(None, space, removable=False))+1).end(i).sink()
    self.assertEqual(Tensor.empty(3, dtype=dtypes.int).custom_kernel(fxn=kernel)[0].tolist(), [1, 11, 21])
  def test_explicit_stage_no_range_reg(self): self._test_explicit_stage_no_range(AddrSpace.REG)
  @unittest.skipIf(not Device[Device.DEFAULT].renderer.has_shared, "LOCAL STAGE needs shared memory")
  def test_explicit_stage_no_range_local(self): self._test_explicit_stage_no_range(AddrSpace.LOCAL)

  def test_explicit_stage_scalar_index(self):
    for direct, want in ((True, [2]), (False, [30])):
      def kernel(out, direct=direct):
        r = UOp.range(4, 0, dtype=dtypes.int32)
        v = r.stage(r, arg=StageOpts(Device.DEFAULT, removable=False)).index(2).load()
        values = UOp.stack(*[UOp.const(x, dtypes.int32) for x in (10, 20, 30, 40)])
        return out.index(0).store(v if direct else values.index(v)).sink()
      with self.subTest(direct=direct):
        self.assertEqual(Tensor.empty(1, dtype=dtypes.int32).custom_kernel(fxn=kernel)[0].tolist(), want)

  def test_split_range_id_free_of_loop(self):
    # the UPCAST range minted by the split gets a fresh id, the while loop's id 1 is taken
    def kernel(C:UOp, A:UOp) -> UOp:
      r, l = UOp.range(4, 0, AxisType.WEAK), UOp.loop(1)
      cnt = UOp.placeholder((1,), dtypes.int, slot=0, addrspace=AddrSpace.REG)
      cnt = cnt.after(r)[0].set(0)
      cnt = cnt.after(cnt[0].store(nxt:=cnt.after(l)[0] + 1).backedge(l, nxt < 3))
      return C[r].set(A[r] + cnt[0].cast(C.dtype), end=r).sink(arg=KernelInfo(opts_to_apply=(Opt(OptOps.SPLIT, 0, (2, AxisType.UPCAST)),)))
    a = Tensor([1., 2, 3, 4])
    self.assertEqual(Tensor.custom_kernel(Tensor.empty(4), a, fxn=kernel)[0].tolist(), [4., 5, 6, 7])

  def test_default_range_runs_in_order(self):
    # a default range is a LOOP: each iteration loads what the previous one stored
    def kernel(A:UOp) -> UOp:
      i = UOp.range(4, 0)
      return A[i+1].store(A[i]).end(i).sink(arg=KernelInfo())
    self.assertEqual(Tensor.custom_kernel(Tensor([1, 2, 3, 4, 5]).contiguous(), fxn=kernel)[0].tolist(), [1, 1, 1, 1, 1])

  def test_range_bound_after_outer_range(self):
    def kernel(C:UOp) -> UOp:
      i = UOp.range(3, 0)
      j = UOp.range(UOp.const(2, dtypes.weakint).after(i), 1)
      return C[0].set(C.after(i, j)[0] + 1, end=(j, i)).sink(arg=KernelInfo())
    self.assertEqual(Tensor.custom_kernel(Tensor.zeros(1, dtype=dtypes.int).contiguous(), fxn=kernel)[0].tolist(), [6])

  def test_gemm_backward_custom(self): self.test_gemm_backward(True)
  def test_gemm_backward(self, custom_backward_gemm=False):
    N = 4
    a_rand = Tensor.randn(N, 8)
    b_rand = Tensor.randn(8, N)
    Tensor.realize(a_rand, b_rand)

    a, b = Tensor(a_rand.numpy()), Tensor(b_rand.numpy())
    c = Tensor.empty(N, N)
    tst = Tensor.custom_kernel(c, a, b, fxn=custom_gemm, grad_fxn=backward_gemm_custom if custom_backward_gemm else backward_gemm)[0]
    tst.sum().backward()
    grad_a, grad_b = a.grad, b.grad
    Tensor.realize(tst, grad_a, grad_b)

    a, b = Tensor(a_rand.numpy()), Tensor(b_rand.numpy())
    ref = (a@b)
    ref.sum().backward()
    real_grad_a, real_grad_b = a.grad, b.grad
    Tensor.realize(ref, real_grad_a, real_grad_b)

    self.assertTrue(tst.allclose(ref, atol=1e-3).item())
    self.assertTrue(grad_a.allclose(real_grad_a, atol=1e-3).item())
    self.assertTrue(grad_b.allclose(real_grad_b, atol=1e-3).item())

  def test_simple_qkv(self):
    N, d = 8, 4
    Q = Tensor.randn(N, d)
    K = Tensor.randn(N, d)
    V = Tensor.randn(N, d)
    O = Tensor.empty(N, d)

    O_custom = Tensor.custom_kernel(O, Q, K, V, fxn=lambda o,q,k,v: simple_qkv_kernel(o,q,k,v))[0]
    O_ref = ((Q @ K.T) / (d ** 0.5)) @ V

    Tensor.realize(O_custom, O_ref)
    self.assertTrue(O_custom.allclose(O_ref, atol=1e-3).item())

  def test_simple_reshape(self):
    a = Tensor.ones(2,3,4).realize()
    b = Tensor.custom_kernel(Tensor.empty_like(a), a, fxn=custom_add_one_kernel)[0]
    b2 = b.reshape(2,12)
    c = Tensor.custom_kernel(Tensor.empty_like(b2), b2, fxn=custom_add_one_kernel)[0]
    GlobalCounters.reset()
    c.realize()
    assert all(i == 3. for i in c.flatten().tolist()), f"all 3 {c.tolist()}"
    assert_kernel_count(2)

  @unittest.skipIf(Device.DEFAULT == "WEBGPU", "kernel timing not supported")
  def test_invalids_into_custom_kernel_with_beam(self):
    a = Tensor.full((4, 4), 3.).contiguous()
    b = Tensor.full((4, 4), 2.).contiguous()
    Tensor.realize(a, b)
    with Context(BEAM=1, IGNORE_BEAM_CACHE=1):
      out = Tensor.invalids(*a.shape, dtype=a.dtype)
      out, *_ = Tensor.custom_kernel(out, a, b, fxn=custom_elementwise_add_kernel)
      result = out.flatten().tolist()
    self.assertTrue(all(x == 5 for x in result), f"expected all 5.0, got {result}")

  @unittest.skip("what are anonymous buffers?")
  def test_anonymous_buffers_in_function(self):
    """Test that custom kernels with anonymous output buffers work inside @function."""
    a = Tensor.full((4, 4), 3.).contiguous()
    b = Tensor.full((4, 4), 2.).contiguous()
    Tensor.realize(a, b)

    def custom_add_with_tmp(o1:UOp, o2:UOp, A:UOp, B:UOp) -> UOp:
      o1,o2,A,B = o1.flatten(), o2.flatten(), A.flatten(), B.flatten()
      i = UOp.range(o1.numel(), 0)
      store_o1 = o1[i].store(A[i]+B[i])
      store_o2 = o2[i].store(A[i]+B[i]+2)
      return UOp.group(store_o1, store_o2).end(i).sink(arg=KernelInfo(name=f"add_with_tmp_{o1.numel()}")).simplify()

    from tinygrad import function
    @function(precompile=True)
    def run(x:Tensor, w:Tensor) -> Tensor:
      out = Tensor.invalids(*x.shape, dtype=x.dtype)
      tmp = Tensor.invalids(*x.shape, dtype=x.dtype)
      out, tmp = Tensor.custom_kernel(out, tmp, x, w, fxn=custom_add_with_tmp)[:2]
      return out+tmp

    result = run(a, b).flatten().tolist()
    expected = (3+2)*2+2
    assert all(x == expected for x in result), f"expected all {expected}, got {result}"

  def test_custom_kernel_sched(self, use_custom=False):
    x = Tensor.arange(32).reshape(8, 4).clone().realize()
    y = Tensor.empty_like(x)
    y = Tensor.custom_kernel(y, x, fxn=custom_add_one_kernel)[0]
    if use_custom:
      z = Tensor.empty_like(x)
      z = Tensor.custom_kernel(z, y.T.T, fxn=custom_add_one_kernel)[0]
    else: z = y.T.T+1
    GlobalCounters.reset()
    z.realize()
    assert_kernel_count(2)
    self.assertEqual(z.tolist(), x.add(2).tolist())

  def test_custom_kernel_sched_copy(self): self.test_custom_kernel_sched(use_custom=True)

  @unittest.skipIf(Device.DEFAULT == "CPU", "test needs to copy from CPU to another device")
  def test_call_to_program(self):
    t = Tensor([-1]).realize()
    # render a source string
    from tinygrad.codegen import do_to_program
    source:str = do_to_program(custom_add_one_kernel(t.uop.param_like(0), t.uop.param_like(1)), Device[t.device].renderer).src[2].arg
    assert isinstance(source, str)
    # call to PROGRAM
    def custom_source(out:UOp, inp:UOp) -> UOp:
      sink = UOp.sink(out.base, inp.base, arg=KernelInfo("add_one_1"))
      return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(sink.toposort())), UOp(Ops.SOURCE, arg=source)))
    cpu_src = Tensor([2], device="CPU").realize()
    t = Tensor.custom_kernel(t, cpu_src.to(t.device), fxn=custom_source)[0]
    cp = t.to("CPU").realize()
    self.assertEqual(t.tolist(), [3])
    self.assertEqual(cp.tolist(), [3])

  def test_sliced_buffer_function(self):
    x = Tensor.arange(32).reshape(8, 4).clone().realize()
    from tinygrad import function
    @function(precompile=True)
    def run(x:Tensor) -> Tensor:
      y = Tensor.invalids(*x.shape, dtype=x.dtype)
      return Tensor.custom_kernel(y, x, fxn=custom_add_one_kernel)[0]
    GlobalCounters.reset()
    y = run(x[0]).realize()
    # backends that support contiguous views don't launch extra kernels
    assert_kernel_count(2 if x[0].uop.contiguous_view() is None else 1)
    self.assertEqual(y.tolist(), [1, 2, 3, 4])

  def test_inplace_transpose(self):
    def custom_assign_row_max_kernel(A:UOp) -> UOp:
      row = UOp.range(A.shape[0], 0)
      col = UOp.range(A.shape[1], 1)
      return A[row, col].store(A[row].max(axis=0)).end(col).end(row).sink(arg=KernelInfo(name=f"assign_row_max_{A.numel()}", opts_to_apply=()))
    a = Tensor.arange(4).reshape(2, 2).clone().realize()
    a = Tensor.custom_kernel(a.T, fxn=custom_assign_row_max_kernel)[0]
    with self.assertRaises(AssertionError):
      self.assertEqual(a.flatten().tolist(), [2, 2, 3, 3])
    self.assertEqual(a.shape, (2, 2))

  @unittest.skipUnless((isinstance(Device[Device.DEFAULT].renderer, (CStyleLanguage, LLVMRenderer)) or Device.DEFAULT == "PYTHON") and
                       Device.DEFAULT != "WEBGPU", "binary not supported on this backend")
  def test_binary(self):
    payload = bytes(range(256))
    def view(x): return x.reshape((16, 16)).permute((1, 0)).reshape((256,))
    def kernel(out:UOp):
      i = UOp.range(len(payload), 0)
      data = view(UOp(Ops.BINARY, arg=payload))
      return out[i].store(data[i]).end(i).sink(arg=KernelInfo(name="binary", opts_to_apply=()))
    self.assertEqual(Tensor.empty(len(payload), dtype=dtypes.uint8).custom_kernel(fxn=kernel)[0].tolist(),
                     view(Tensor(payload, dtype=dtypes.uint8)).tolist())

  def test_one_lane_stack_runtime_index(self):
    def kernel(A:UOp): return A[0].store(UOp.stack(UOp.const(42, dtypes.int)).index(A[1].load())).sink(arg=KernelInfo(opts_to_apply=()))
    self.assertEqual(Tensor.zeros(2, dtype=dtypes.int).contiguous().custom_kernel(fxn=kernel)[0].tolist(), [42, 0])

  def test_stack_runtime_index_two_axes(self):
    def kernel(A:UOp):
      i, j = UOp.range(2, 0), UOp.range(2, 1)
      m = UOp.const(((1, 2), (3, 4)), dtypes.int)
      return A[i*2+j].store(m.index(i, j)).end(j, i).sink(arg=KernelInfo(opts_to_apply=()))
    self.assertEqual(Tensor.zeros(4, dtype=dtypes.int).contiguous().custom_kernel(fxn=kernel)[0].tolist(), [1, 2, 3, 4])

  def test_stack_runtime_index_then_getitem(self):
    def kernel(A:UOp, B:UOp):
      return B[0].store(UOp.const(((1, 2), (3, 4)), dtypes.int).index(A[0].load())[A[1].load()]).sink(arg=KernelInfo(opts_to_apply=()))
    self.assertEqual(Tensor.custom_kernel(Tensor([1, 0]).contiguous(), Tensor.zeros(1, dtype=dtypes.int).contiguous(), fxn=kernel)[1].tolist(), [3])

  def test_index_coordinate_read(self):
    def kernel(A:UOp, B:UOp, C:UOp): return B[0].store(C[A[0]]).sink(arg=KernelInfo(opts_to_apply=()))
    a, b, c = Tensor([2]).contiguous(), Tensor.zeros(1, dtype=dtypes.int).contiguous(), Tensor([10, 11, 12]).contiguous()
    with Context(CHECK_OOB=0):
      self.assertEqual(Tensor.custom_kernel(a, b, c, fxn=kernel)[1].tolist(), [12])

  def test_index_vector_bitcast_widen(self):
    def kernel(A:UOp, B:UOp): return B[0].store(A.index().load().bitcast(dtypes.int32)[1]).sink(arg=KernelInfo(opts_to_apply=()))
    _, b = Tensor.custom_kernel(Tensor([1, 2, -3, 4], dtype=dtypes.int16).contiguous(), Tensor.zeros(1, dtype=dtypes.int32).contiguous(), fxn=kernel)
    self.assertEqual(b.tolist(), [(-3 & 0xFFFF) + (4 << 16)])

  def test_index_vector_bitcast_narrow(self):
    def kernel(A:UOp, B:UOp): return B[0].store((A.index()+1).bitcast(dtypes.uint16)[3]).sink(arg=KernelInfo(opts_to_apply=()))
    _, b = Tensor.custom_kernel(Tensor([1, 5 << 16], dtype=dtypes.uint32).contiguous(), Tensor.zeros(1, dtype=dtypes.uint16).contiguous(), fxn=kernel)
    self.assertEqual(b.tolist(), [5])

  def test_index_shaped_value_after_range(self):
    def kernel(B:UOp):
      r = UOp.range(3, 0)
      value = UOp.stack(UOp.const(2.0, dtypes.float)).reciprocal().after(r)
      return B[0].store(value.index(UOp.const(0, dtypes.int).after(r)).reduce(r, arg=Ops.MAX)).sink(arg=KernelInfo(opts_to_apply=()))
    self.assertEqual(Tensor.zeros(1).contiguous().custom_kernel(fxn=kernel)[0].tolist(), [0.5])

  @unittest.skipUnless(isinstance(Device[Device.DEFAULT].renderer, CStyleLanguage), "custom renders in cstyle")
  def test_custom_uop(self):
    N = 4
    def kernel(x:UOp) -> UOp:
      y = UOp.native("__builtin_shufflevector({0}, {0}, 3, 2, 1, 0)", shape=(N,), dtype=x.dtype,
                     fmt=True).call(UOp.stack([x[i].load() for i in range(N)]))
      self.assertEqual(y.shape, (N,))
      return x.store(y).sink(arg=KernelInfo("kernel"))
    out = Tensor.custom_kernel(Tensor.arange(N).clone("CPU"), fxn=kernel)[0]
    self.assertEqual(out.tolist(), [3, 2, 1, 0])

  def test_alloc(self, N=1):
    def kernel(out:UOp, inp:UOp):
      i = UOp.range(N, 0, AxisType.LOOP)
      alloc = UOp.alloc((N,), dtypes.int, -1, addrspace=AddrSpace.REG)
      buf = UOp._buffer(N, dtypes.int, -1, addrspace=AddrSpace.REG)
      alloc = alloc[i].set(inp[i])
      buf = buf.after(alloc)[i].set(inp[i+N])
      return out[i].store(alloc.after(buf)[i] + buf[i]).end(i).sink(arg=KernelInfo(opts_to_apply=()))
    out = Tensor.custom_kernel(Tensor.empty(N, dtype=dtypes.int), Tensor.arange(2*N).clone().realize(), fxn=kernel)[0]
    self.assertEqual(out.tolist(), [2*i+N for i in range(N)])

  @unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, PTXRenderer), "PTX does not support dynamic register indexing")
  def test_alloc_vec(self): self.test_alloc(4)

  def test_store_value_broadcasts_to_target(self):
    def stored(shape:tuple[int, ...]) -> list:
      def kernel(out:UOp, inp:UOp):
        reg = UOp.alloc((2,), dtypes.int, addrspace=AddrSpace.REG)
        reg = reg.after(reg.store(inp.reshape(shape)))
        return UOp.sink(out[0].store(reg[0]), out[1].store(reg[1]), arg=KernelInfo(opts_to_apply=()))
      return Tensor.custom_kernel(Tensor.empty(2, dtype=dtypes.int), Tensor([2, 3]).realize(), fxn=kernel)[0].tolist()
    self.assertEqual(stored((1, 2)), [2, 3])
    with self.assertRaises(ValueError): stored((2, 1))

class TestCustomKernelArgs(unittest.TestCase):
  def test_duplicate_call_arg(self):
    x = Tensor.arange(4).clone().realize()
    x = Tensor.custom_kernel(x, x, fxn=custom_add_one_kernel)[0]
    # webgpu silently errors when a kernel has duplicate buffer args, so the list stays the same.
    # https://gpuweb.github.io/gpuweb/#abstract-opdef-encoder-bind-groups-alias-a-writable-resource
    self.assertEqual(x.tolist(), [1, 2, 3, 4] if Device.DEFAULT != "WEBGPU" else [0, 1, 2, 3])

  def test_unused_buffer_arg(self):
    a, b = Tensor([100.0, 200, 300, 400]), Tensor([1.0, 2, 3, 4])
    out = Tensor.custom_kernel(Tensor.empty(4), a, b, fxn=custom_ignore_first_kernel)[0]
    self.assertEqual(out.tolist(), [2, 3, 4, 5])

  def test_captured_scalar_grad(self):
    # the gradient callback gets the explicit inputs, not the Variable the kernel captures
    v = Variable("v", 1, 4, dtypes.int).bind(2)
    def kernel(C:UOp, A:UOp) -> UOp:
      i = UOp.range(C.numel(), 0)
      return C[i].store(A[i]*v.cast(A.dtype)).end(i).sink(arg=KernelInfo(name="mul_v"))
    a = Tensor([1.0, 2.0])
    Tensor.custom_kernel(Tensor.empty(2), a, fxn=kernel, grad_fxn=lambda g, k: (None, (Tensor(g)*2).uop))[0].sum().backward()
    assert a.grad is not None
    self.assertEqual(a.grad.tolist(), [2.0, 2.0])

  def test_scalar_arg_any_position(self):
    for n_slot in range(3):
      with self.subTest(n_slot=n_slot):
        srcs = [Tensor.empty(4, dtype=dtypes.int), Tensor([1, 2, 3, 4], dtype=dtypes.int).realize()]
        srcs.insert(n_slot, Tensor(Variable("n", 0, 100, dtypes.int).bind(5)))
        out = Tensor.custom_kernel(*srcs, fxn=functools.partial(custom_add_var_kernel, n_slot=n_slot))[1 if n_slot == 0 else 0]
        self.assertEqual(run_kernel_args(out, f"add_var_{n_slot}_1"), ["n" if i == n_slot else None for i in range(3)])
        self.assertEqual(out.tolist(), [6, 7, 8, 9])

  @unittest.skipIf(Device.DEFAULT == "PYTHON", "needs a device that isnt the default")
  def test_scalar_arg_first_other_device(self):
    srcs = [Tensor(Variable("n", 0, 100, dtypes.int).bind(5), device="PYTHON"), Tensor.empty(4, dtype=dtypes.int, device="PYTHON"),
            Tensor([1, 2, 3, 4], dtype=dtypes.int, device="PYTHON").realize()]
    out = Tensor.custom_kernel(*srcs, fxn=functools.partial(custom_add_var_kernel, n_slot=0))[1]
    self.assertEqual(out.tolist(), [6, 7, 8, 9])

  def test_scalar_arg_past_registers(self):
    # 9 inputs with the scalar between pointers, x86 passes the ones after the 6th (4th on windows) on the stack. every buffer differs
    srcs = [Tensor.empty(4, dtype=dtypes.int)] + [Tensor([k, 2*k, 3*k, 4*k], dtype=dtypes.int).realize() for k in range(1, 8)]
    srcs.insert(4, Tensor(Variable("n", 0, 100, dtypes.int).bind(5)))
    out = Tensor.custom_kernel(*srcs, fxn=functools.partial(custom_add_var_kernel, n_slot=4))[0]
    self.assertEqual(run_kernel_args(out, "add_var_4_7"), [None]*4 + ["n"] + [None]*4)
    self.assertEqual(out.tolist(), [33, 61, 89, 117])

  def test_unused_arg_then_var(self):
    srcs = (Tensor.empty(4, dtype=dtypes.int), Tensor.empty(4, dtype=dtypes.int), Tensor([1, 2, 3, 4], dtype=dtypes.int).realize(),
            Tensor(Variable("n", 0, 100, dtypes.int).bind(5)))
    out = Tensor.custom_kernel(*srcs, fxn=custom_ignore_first_var_kernel)[0]
    linear, externs = out.linear_with_externs()
    run_schedule(linear, externs, {"k": 10})
    self.assertEqual(out.tolist(), [16, 17, 18, 19])

  def test_scalar_arg_jit_fixed_bind(self):
    # m is bound inside the jitted function so it's not an input, a replay keeps m=10 while the input n changes
    kernel = functools.partial(custom_add_var_kernel, n_slot=0)
    def f(B, n):
      Bm = Tensor.custom_kernel(Tensor(Variable("m", 0, 100, dtypes.int).bind(10)), Tensor.empty(4, dtype=dtypes.int), B, fxn=kernel)[1]
      return Tensor.custom_kernel(Tensor(n), Tensor.empty(4, dtype=dtypes.int), Bm, fxn=kernel)[1].realize()
    jf, B, n = TinyJit(f), Tensor([1, 2, 3, 4], dtype=dtypes.int).realize(), Variable("n", 0, 100, dtypes.int)
    for k in range(1, 6): self.assertEqual(jf(B, n.bind(k)).tolist(), [11+k, 12+k, 13+k, 14+k])

  def test_variable_arg_is_not_copy(self):
    # a buffer and a Variable aren't a copy kernel, so the ranges merge in codegen after the split, and SPLIT sees one axis of 8
    def kernel(D:UOp, N:UOp) -> UOp:
      a, b = UOp.range(2, 0, AxisType.WEAK), UOp.range(4, 1, AxisType.WEAK)
      i = a*4+b
      return D[i].store((i%2+N).cast(dtypes.int)).end(a, b).sink(arg=KernelInfo(opts_to_apply=(Opt(OptOps.SPLIT, 0, (8, AxisType.UPCAST)),)))
    d = Tensor.custom_kernel(Tensor.empty(8, dtype=dtypes.int), Tensor(Variable("n", 1, 8, dtypes.int).bind(2)), fxn=kernel)[0]
    self.assertEqual(d.tolist(), [2, 3]*4)

class TestCustomKernelSharded(unittest.TestCase):
  def test_device_number_arg(self):
    # the kernel's DEVICE range is its device number arg, bound per device and validated on each
    def kernel(C:UOp) -> UOp: return C[0].store(UOp.range(2, 0, AxisType.DEVICE).cast(dtypes.int)).sink(arg=KernelInfo(name="store_dnum"))
    out = Tensor.custom_kernel(Tensor.zeros(2, dtype=dtypes.int, device="CPU").shard(("CPU:1", "CPU:2"), axis=0).contiguous(), fxn=kernel)[0]
    linear, externs = out.linear_with_externs()
    validated = compile_linear(linear, validate=True).toposort()
    self.assertEqual(len([c for c in validated if c.op is Ops.CALL and c.body.op is Ops.NATIVE and c.body.arg.code == "validate"
                          and c.body.src[-1].arg.name == "store_dnum"]), 2)
    run_schedule(linear, externs)
    self.assertEqual(out.tolist(), [0, 1])

  def test_sharded_add_one(self):
    # PYTHON backend explicitly checks for OOB access for wrong multi shape regression
    devs = ("PYTHON:0", "PYTHON:1")
    a = Tensor.ones(4, 4).contiguous().shard(devs, axis=0)
    c = Tensor(Tensor.empty(2, 4, device=devs).uop.unshard(0), device=devs)
    c = Tensor.custom_kernel(c, a, fxn=custom_add_one_kernel)[0]
    assert (c == 2).all().item()

  def test_gemm_multi(self):
    devs = ("CPU:0", "CPU:1")
    N = 16
    a = Tensor.randn(N, N).shard_(devs, axis=0)
    b = Tensor.randn(N, N).to(devs)
    c = Tensor(Tensor.empty(N//2, N, device=devs).uop.unshard(0), device=devs)
    tst = Tensor.custom_kernel(c, a, b, fxn=custom_gemm)[0]
    self.assertTrue(tst.allclose(a@b, atol=1e-3).item())

  def test_multi_invalids_custom_kernel_no_copy(self):
    devs = ("CPU:0", "CPU:1")
    a = Tensor.ones(4, 4).shard(devs, axis=0).realize()
    c = Tensor(Tensor.invalids(2, 4, dtype=dtypes.float, device=devs).uop.unshard(0), device=devs)
    c = Tensor.custom_kernel(c, a, fxn=custom_add_one_kernel)[0]
    GlobalCounters.reset()
    c.realize()
    assert_kernel_count(len(devs))
    self.assertTrue((c == 2).all().item())

  def test_reduce_device_range(self):
    def kernel(C:UOp, A:UOp, B:UOp):
      d = UOp.range(2, 0, AxisType.DEVICE)
      return C[0].store(UOp.stack(A, B)[d, 0].reduce(d, arg=Ops.ADD)).sink(arg=KernelInfo(opts_to_apply=()))
    devs = ('CPU:1', 'CPU:2')
    lanes = [Tensor([2]).to(devs[0]).realize(), Tensor([3]).to(devs[1]).realize()]
    out = Tensor.empty(2, dtype=lanes[0].dtype, device='CPU').shard(devs, axis=0).realize()
    with self.assertRaises(RuntimeError): Tensor.custom_kernel(out, *lanes, fxn=kernel)[0].realize()

  def test_reduce_device_range_not_copy(self):
    def kernel(C:UOp, A:UOp):
      d = UOp.range(2, 0, AxisType.DEVICE)
      return UOp(Ops.NOOP).after(d.after(C[0].store(A[0])).reduce(d, arg=Ops.ADD)).sink(arg=KernelInfo(opts_to_apply=()))
    out = Tensor.zeros(2, device='CPU').shard(('CPU:1', 'CPU:2'), axis=0).contiguous().realize()
    with self.assertRaises(RuntimeError): Tensor.custom_kernel(out, Tensor([7.]).realize(), fxn=kernel)[0].realize()

class TestUnshardIndex(unittest.TestCase):
  """Regression tests for INDEX on UNSHARD (fragment) resolution in codegen/multi.py.

  A fragment is a per-thread REG buffer wrapped in UNSHARD over LOCAL thread ranges.
  index_multi must resolve an INDEX on the UNSHARD view into an INDEX on the per-thread
  shard. Ownership is contiguous: idx = rng*shard_sz + local (thread rng owns [rng*shard_sz, ...)).
  Other placements are movement ops on the fragment.
  """
  def _run(self, kernel, shape=(8, 8), inputs=()):
    c = Tensor.empty(*shape)
    out = Tensor.custom_kernel(c, *inputs, fxn=kernel)[0]
    try: return out.numpy()
    except RuntimeError as e:
      if isinstance(Device[Device.DEFAULT].renderer, PTXRenderer) and "dynamic register indexing" in str(e):
        self.skipTest("PTX does not support dynamic register indexing")
      raise

  @unittest.skipIf(not Device[Device.DEFAULT].renderer.has_local, "fragment tests need LOCAL ranges")
  def test_contiguous_fragment_index(self):
    # thread ty owns rows [ty*8, ty*8+8) of a 64-row fragment -- contiguous ownership.
    # This is the pre-existing case that index_multi always handled.
    def kernel(C:UOp) -> UOp:
      ty = UOp.range(8, 0, AxisType.LOCAL)
      ir = UOp.range(8, 1)
      j = UOp.range(8, 2)
      # 8x8 fragment, 8 threads -> 64x8 full tile. thread ty owns rows [ty*8, ty*8+8).
      frag = UOp.placeholder((8, 8), dtypes.float32, 0, AddrSpace.REG).unshard((0,), (ty,))
      return C[ty*8 + ir, j].store(frag[ty*8 + ir, j]).end(j, ir, ty).sink(arg=KernelInfo(name="contig_frag"))
    out = self._run(kernel, (64, 8))
    assert out.shape == (64, 8)

  @unittest.skipIf(not Device[Device.DEFAULT].renderer.has_local, "fragment tests need LOCAL ranges")
  def test_strided_fragment_index(self):
    # thread ty writes rows {ty, ty+8, ..., ty+56} of C: the strided placement is a permute of the fragment's blocks
    def kernel(C:UOp, A:UOp) -> UOp:
      ty = UOp.range(8, 0, AxisType.LOCAL)
      ir = UOp.range(8, 1)
      j = UOp.range(8, 2)
      frag = UOp.placeholder((8, 8), dtypes.float32, 0, AddrSpace.REG).unshard((0,), (ty,))
      frag = frag.after(frag.store(A)).reshape(8, 8, 8).permute(1, 0, 2)
      return C[ty + ir*8, j].store(frag[ir, ty, j]).end(j, ir, ty).sink(arg=KernelInfo(name="strided_frag", opts_to_apply=()))
    a = Tensor.arange(512).float().reshape(64, 8).contiguous()
    np.testing.assert_equal(self._run(kernel, (64, 8), inputs=(a,)), a.numpy().reshape(8, 8, 8).transpose(1, 0, 2).reshape(64, 8))

  def test_strided_index_into_block_fragment(self):
    # owner + local*2 reads another owner's block, so index_multi must raise
    def kernel(C:UOp, A:UOp) -> UOp:
      owner = UOp.range(2, 0)
      local = UOp.range(2, 1)
      frag = UOp.placeholder((2,), dtypes.float32, 0, AddrSpace.REG).unshard(0, owner)
      frag = frag.after(frag.store(A))
      return C[owner + local*2].store(frag[owner + local*2]).end(local, owner).sink(arg=KernelInfo(opts_to_apply=()))
    with self.assertRaisesRegex(RuntimeError, "cannot shard index"): self._run(kernel, (4,), inputs=(Tensor.arange(4).float().contiguous(),))

  def test_fragment_index_cannot_shard(self):
    # thread ty indexing rows [ty, ty+8) overlaps with other threads' rows, so index_multi must raise.
    def kernel(C:UOp) -> UOp:
      ty = UOp.range(8, 0, AxisType.LOCAL)
      ir = UOp.range(8, 1)
      j = UOp.range(8, 2)
      frag = UOp.placeholder((8, 8), dtypes.float32, 0, AddrSpace.REG).unshard((0,), (ty,))
      return C[ty + ir, j].store(frag[ty + ir, j]).end(j, ir, ty).sink(arg=KernelInfo(name="bad_frag"))
    with self.assertRaisesRegex(RuntimeError, "cannot shard index"):
      self._run(kernel, (64, 8))

def _run_fragment_kernel(testcase, kernel, out_shape, inputs=()):
  c = Tensor.empty(*out_shape)
  out = Tensor.custom_kernel(c, *inputs, fxn=kernel)[0]
  try: return out.numpy()
  except RuntimeError as e:
    if isinstance(Device[Device.DEFAULT].renderer, PTXRenderer) and "dynamic register indexing" in str(e):
      testcase.skipTest("PTX does not support dynamic register indexing")
    raise

class TestUnshardAlu(unittest.TestCase):
  """Tests for ALU on (fragment) UNSHARD values in codegen/multi.py's alu_multi.

  An ALU with UNSHARD srcs lowers to per-shard ops when every src is one of:
    same sharding:  peel the UNSHARD, keep the layout
    scalar:         broadcast to every shard
    whole unsharded same-shape value: takes its per-shard sub-view (shard_subview)
  """
  @unittest.skipIf(not Device[Device.DEFAULT].renderer.has_local, "fragment tests need LOCAL ranges")
  def test_alu_scalar_broadcast(self):
    # scalar srcs broadcast to every shard: frag*2.0 where frag is 1.5 per thread -> 3.0 everywhere
    def kernel(C:UOp) -> UOp:
      ty = UOp.range(8, 0, AxisType.LOCAL)
      # 8 values per thread, 8 threads -> 64-value full view
      frag = UOp.placeholder((8,), dtypes.float32, 0, AddrSpace.LOCAL).unshard((0,), (ty,))
      v = frag.after(frag.store(1.5)) * 2.0
      return C.store(v).end(ty).sink(arg=KernelInfo(name="alu_scalar", opts_to_apply=()))
    out = _run_fragment_kernel(self, kernel, (64,))
    np.testing.assert_allclose(out, 3.0)

  @unittest.skipIf(not Device[Device.DEFAULT].renderer.has_local, "fragment tests need LOCAL ranges")
  def test_alu_whole_value_subview(self):
    # UNSHARD + whole unsharded same-shape value: each shard adds its own sub-view of A.
    def kernel(C:UOp, A:UOp) -> UOp:
      ty = UOp.range(8, 0, AxisType.LOCAL)
      frag = UOp.placeholder((8,), dtypes.float32, 0, AddrSpace.LOCAL).unshard((0,), (ty,))
      v = frag.after(frag.store(0.0)) + A
      return C.store(v).end(ty).sink(arg=KernelInfo(name="alu_subview", opts_to_apply=()))
    a = Tensor(np.arange(64, dtype=np.float32))
    out = _run_fragment_kernel(self, kernel, (64,), inputs=(a,))
    np.testing.assert_allclose(out, a.numpy(), atol=1e-4)

class TestUnshardStore(unittest.TestCase):
  """Tests for STORE of a sharded value into an unsharded dest (store_value_multi in codegen/multi.py).

  Every shard stores its value into its own contiguous sub-view of the dest, one SHRINK per sharded axis.
  """
  @unittest.skipIf(not Device[Device.DEFAULT].renderer.has_local, "fragment tests need LOCAL ranges")
  def test_store_unshard_value_2axis_reshape(self):
    # the reshape keeps both sharded axes, each with its own shard count (4 and 2)
    def kernel(C:UOp, A:UOp) -> UOp:
      ty = UOp.range(4, 0, AxisType.LOCAL)
      tx = UOp.range(2, 1, AxisType.LOCAL)
      frag = UOp.placeholder((2, 1, 1, 2), dtypes.float32, 0, AddrSpace.REG).unshard((1, 2), (ty, tx))
      v = (frag.after(frag.store(0.0)) + A).reshape(2, 4, 2, 2, 1)
      return C.store(v).end(tx, ty).sink(arg=KernelInfo(name="store_unshard_2axis_reshape", opts_to_apply=()))
    a = Tensor(np.arange(32, dtype=np.float32).reshape(2, 4, 2, 2))
    out = _run_fragment_kernel(self, kernel, (2, 4, 2, 2, 1), inputs=(a,))
    np.testing.assert_allclose(out, a.numpy().reshape(2, 4, 2, 2, 1), atol=1e-4)

  def _test_store_load_fragment(self, addrspace:AddrSpace):
    # thread ty stores A[ty*8:ty*8+8] into its fragment, then reads it back into the same slice of C
    def kernel(C:UOp, A:UOp) -> UOp:
      ty = UOp.range(8, 0, AxisType.LOCAL)
      frag = UOp.placeholder((8,), dtypes.float32, 0, addrspace).unshard((0,), (ty,))
      return C.store(frag.after(frag.store(A))).end(ty).sink(arg=KernelInfo(name="store_load_fragment", opts_to_apply=()))
    a = Tensor.arange(64, dtype=dtypes.float32)
    out = _run_fragment_kernel(self, kernel, (64,), inputs=(a,))
    np.testing.assert_equal(out, a.numpy())

  @unittest.skipIf(not Device[Device.DEFAULT].renderer.has_local, "fragment tests need LOCAL ranges")
  def test_store_load_reg_fragment(self): self._test_store_load_fragment(AddrSpace.REG)

  @unittest.skipIf(not Device[Device.DEFAULT].renderer.has_local, "fragment tests need LOCAL ranges")
  @unittest.expectedFailure  # TODO: should not fail silently
  def test_store_load_local_fragment(self): self._test_store_load_fragment(AddrSpace.LOCAL)

class TestUOpReduce(unittest.TestCase):
  def test_uop_sum(self):
    a = Tensor([1.0, 2, 3, 4, 5])
    self.assertAlmostEqual(Tensor(a.uop.sum(axis=0)).item(), 15.0)

  def test_uop_sum_2d(self):
    a = Tensor.arange(6).reshape(2, 3).float()
    result = Tensor(a.uop.sum(axis=1)).numpy()
    assert result[0] == 3 and result[1] == 12

  def test_uop_sum_all(self):
    a = Tensor.arange(6).reshape(2, 3).float()
    self.assertAlmostEqual(Tensor(a.uop.sum()).item(), 15.0)

  def test_uop_sum_negative_axis(self):
    a = Tensor.arange(6).reshape(2, 3).float()
    result = Tensor(a.uop.sum(axis=-1)).numpy()
    assert result[0] == 3 and result[1] == 12

  def test_uop_sum_dtype(self):
    a = Tensor([1.0, 2, 3], dtype=dtypes.float16)
    result = Tensor(a.uop.sum(axis=0, dtype=dtypes.float32))
    self.assertEqual(result.dtype, dtypes.float)
    self.assertAlmostEqual(result.item(), 6.0, places=2)

  def test_uop_prod(self):
    a = Tensor([1.0, 2, 3, 4, 5])
    self.assertAlmostEqual(Tensor(a.uop.prod(axis=0)).item(), 120.0)

  def test_uop_max(self):
    a = Tensor([1.0, 5, 3, 2, 4])
    self.assertAlmostEqual(Tensor(a.uop.max(axis=0)).item(), 5.0)

  def test_uop_max_2d(self):
    a = Tensor([[1, 5, 3], [4, 2, 6]]).float()
    result = Tensor(a.uop.max(axis=0)).numpy()
    assert result[0] == 4 and result[1] == 5 and result[2] == 6

  def test_uop_std(self):
    a = Tensor([2.0, 4, 4, 4, 5, 5, 7, 9])
    self.assertAlmostEqual(Tensor(a.uop.std()).item(), a.std().item(), places=5)

class TestUOpWhere(unittest.TestCase):
  def test_uop_where_both_const(self):
    cond = Tensor([True, False, True])
    result = Tensor(cond.uop.where(1, 0))
    self.assertEqual(result.tolist(), [1, 0, 1])

    result = Tensor(cond.uop.where(1.5, 0))
    self.assertEqual(result.tolist(), [1.5, 0, 1.5])

if __name__ == '__main__':
  unittest.main()
