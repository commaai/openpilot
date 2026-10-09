import unittest
from tinygrad import Tensor, UOp, GlobalCounters, Context, Device
import numpy as np
from tinygrad.dtype import AddrSpace, dtypes, Invalid
from tinygrad.helpers import getenv
from tinygrad.schedule.rangeify import BufferizeOpts
from tinygrad.uop.ops import KernelInfo, AxisType, Ops, uopfunc
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.renderer.llvmir import LLVMRenderer
from test.helpers import assert_kernel_count
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

def custom_sum(B:UOp, A:UOp) -> UOp:
  i = UOp.range(A.shape[0], 0, axis_type=AxisType.LOOP)
  B = B[0].set(0.0)
  B = B[0].set(B.after(i)[0] + A[i], end=i)
  return B.sink(arg=KernelInfo(name=f"custom_sum_{A.shape[0]}", opts_to_apply=()))

def flip_contract_kernel(dest:UOp, src:UOp):
  i = UOp.range(dest.shape[0], 0)
  j = UOp.range(dest.shape[1], 1, AxisType.UPCAST)
  vec = src[i, j].contract(j)
  store = UOp.group(*[dest[i, k].store(vec.index(3-k)) for k in range(4)])
  return store.end(i, j).sink(arg=KernelInfo(name=f"flip_contract_{dest.numel()}", opts_to_apply=()))

def slice_sum_kernel(dest:UOp, src:UOp):
  G = UOp.range(src.shape[0], 0, dtype=dtypes.int)
  slice_src = src[G, :]
  reg = UOp.placeholder((1,), dest.dtype, 0, addrspace=AddrSpace.REG)
  reg = reg.after(G)[0].set(0)
  R = UOp.range(src.shape[1], 1, AxisType.LOOP)
  reg = reg[0].set(reg.after(R)[0] + slice_src[R], end=R)
  ast = dest[G].set(reg[0], end=G)
  return ast.sink(arg=KernelInfo(name=f"slice_sum_{src.shape[0]}_{src.shape[1]}", opts_to_apply=()))

def simple_qkv_kernel(O:UOp, Q:UOp, K:UOp, V:UOp) -> UOp:
  # attention without softmax
  N, d = Q.shape[0], Q.shape[1]

  i = UOp.range(N, 0)  # output row
  d_out = UOp.range(d, 1)  # output column
  j = UOp.range(N, 2, axis_type=AxisType.LOOP)

  k_inner = UOp.range(d, 3, axis_type=AxisType.LOOP)
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

  def test_simple(self):
    a = Tensor.ones(16, 16).contiguous()
    b = Tensor.ones(16, 16).contiguous()
    c = Tensor.empty(16, 16)

    c = Tensor.custom_kernel(c,a,b, fxn=custom_elementwise_add_kernel)[0]

    out = c.flatten().tolist()
    assert all(x == 2 for x in out), "all 2"

  def test_duplicate_call_arg(self):
    x = Tensor.arange(4).clone().realize()
    x = Tensor.custom_kernel(x, x, fxn=custom_add_one_kernel)[0]
    # webgpu silently errors when a kernel has duplicate buffer args, so the list stays the same.
    # https://gpuweb.github.io/gpuweb/#abstract-opdef-encoder-bind-groups-alias-a-writable-resource
    self.assertEqual(x.tolist(), [1, 2, 3, 4] if Device.DEFAULT != "WEBGPU" else [0, 1, 2, 3])

  def test_simple_sharded(self):
    devs = ("CPU:0", "CPU:1")

    a = Tensor.ones(16, 16).contiguous().shard(devs, axis=0)
    b = Tensor.ones(16, 16).contiguous().shard(devs, axis=0)
    # ugly construction to get a sharded empty tensor
    c = Tensor(Tensor.empty(8, 16, device=devs).uop.unshard(0), device=devs)
    c = Tensor.custom_kernel(c,a,b, fxn=custom_elementwise_add_kernel)[0]
    out = c.flatten().tolist()
    assert all(x == 2 for x in out), "all 2"

  def test_sharded_add_one(self):
    # PYTHON backend explicitly checks for OOB access for wrong multi shape regression
    devs = ("PYTHON:0", "PYTHON:1")
    a = Tensor.ones(4, 4).contiguous().shard(devs, axis=0)
    c = Tensor(Tensor.empty(2, 4, device=devs).uop.unshard(0), device=devs)
    c = Tensor.custom_kernel(c, a, fxn=custom_add_one_kernel)[0]
    assert (c == 2).all().item()

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

  @unittest.skip("contract shouldn't be supported here")
  def test_flip_contract(self):
    a = Tensor.randn(10,4)
    b = Tensor.empty_like(a)
    b = b.custom_kernel(a, fxn=flip_contract_kernel)[0]
    self.assertTrue((a.flip(1) == b).all().item())

  def test_noncontig(self):
    a = Tensor.ones(16, 16).contiguous()
    tst = Tensor.empty_like(a)
    b = a+1
    b_p1 = Tensor.custom_kernel(tst, b, fxn=custom_add_one_kernel)[0]
    self.assertTrue((b_p1 == 3).all().item())

  def test_unused_buffer_arg(self):
    a, b = Tensor([100.0, 200, 300, 400]), Tensor([1.0, 2, 3, 4])
    out = Tensor.custom_kernel(Tensor.empty(4), a, b, fxn=custom_ignore_first_kernel)[0]
    self.assertEqual(out.tolist(), [2, 3, 4, 5])

  def test_sum(self):
    a = Tensor([1.0, 2, 3, 4, 5])
    tst = Tensor.empty(1)
    b = Tensor.custom_kernel(tst, a, fxn=custom_sum)[0]
    self.assertEqual(b.item(), 15)

  def test_sum_outside(self):
    a = Tensor([1.0, 2, 3, 4, 5])+1
    tst = Tensor.empty(1)
    b = Tensor.custom_kernel(tst, a, fxn=custom_sum)[0]
    self.assertEqual(b.item(), 20)

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
      i, j, jj = UOp.range(4, 0), UOp.range(8, 1, AxisType.LOOP), UOp.range(8, 2, AxisType.LOOP)
      stage = (A[i, j] * 2).bufferize(j, arg=BufferizeOpts(None, AddrSpace.LOCAL))
      return C[i].store(stage.index(jj).reduce(jj, arg=Ops.ADD)).end(i).sink(arg=KernelInfo(opts_to_apply=()))
    a = Tensor.arange(32).reshape(4, 8).float().contiguous().realize()
    self.assertEqual(Tensor.custom_kernel(Tensor.empty(4), a, fxn=kernel)[0].tolist(), (a*2).sum(1).tolist())

  @unittest.skipIf(not Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  def test_stage_in_thread_range(self):
    # the STAGE is inside the thread range i, so every thread stages its own row
    def kernel(C:UOp, A:UOp) -> UOp:
      i, j, jj = UOp.range(4, 0, AxisType.LOCAL), UOp.range(8, 1, AxisType.LOOP), UOp.range(8, 2, AxisType.LOOP)
      stage = (A[i, j] * 2).bufferize(j, arg=BufferizeOpts(None, AddrSpace.LOCAL))
      return C[i].store(stage.index(jj).reduce(jj, arg=Ops.ADD)).end(i).sink(arg=KernelInfo(opts_to_apply=()))
    a = Tensor.arange(32).reshape(4, 8).float().contiguous().realize()
    self.assertEqual(Tensor.custom_kernel(Tensor.empty(4), a, fxn=kernel)[0].tolist(), (a*2).sum(1).tolist())

  @unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, PTXRenderer), "PTX does not support dynamic register indexing")
  def test_reg_stage_then_reduce(self):
    # the REG buffer of the STAGE and the accumulator of the reduce are different buffers
    def kernel(C:UOp, A:UOp) -> UOp:
      i, j, jj = UOp.range(4, 0), UOp.range(8, 1, AxisType.LOOP), UOp.range(8, 2, AxisType.LOOP)
      stage = (A[i, j] * 2).bufferize(j, arg=BufferizeOpts(None, AddrSpace.REG))
      return C[i].store(stage.index(jj).reduce(jj, arg=Ops.ADD)).end(i).sink(arg=KernelInfo(opts_to_apply=()))
    a = Tensor.arange(32).reshape(4, 8).float().contiguous().realize()
    self.assertEqual(Tensor.custom_kernel(Tensor.empty(4), a, fxn=kernel)[0].tolist(), (a*2).sum(1).tolist())

  def test_reg_placeholder_then_reduce(self):
    # the accumulator of the reduce does not reuse the slot of a REG placeholder in the kernel
    def kernel(C:UOp, A:UOp) -> UOp:
      i, j = UOp.range(4, 0), UOp.range(8, 1)
      reg = UOp.placeholder((1,), dtypes.float, 0, addrspace=AddrSpace.REG)
      reg = reg.after(i)[0].set(A[i, 0])
      return C[i].store(A[i, j].reduce(j, arg=Ops.ADD) + reg[0]).end(i).sink(arg=KernelInfo(opts_to_apply=()))
    a = Tensor.arange(32).reshape(4, 8).float().contiguous().realize()
    self.assertEqual(Tensor.custom_kernel(Tensor.empty(4), a, fxn=kernel)[0].tolist(), (a.sum(1) + a[:, 0]).tolist())

  def test_split_range_id_free_of_loop(self):
    # the UPCAST range minted by the split gets a fresh id, the while loop's id 1 is taken
    def kernel(C:UOp, A:UOp) -> UOp:
      r, l = UOp.range(4, 0), UOp.loop(1)
      cnt = UOp.placeholder((1,), dtypes.int, slot=0, addrspace=AddrSpace.REG)
      cnt = cnt.after(r)[0].set(0)
      cnt = cnt.after(cnt[0].store(nxt:=cnt.after(l)[0] + 1).backedge(l, nxt < 3))
      return C[r].set(A[r] + cnt[0].cast(C.dtype), end=r).sink(arg=KernelInfo(opts_to_apply=(Opt(OptOps.SPLIT, 0, (2, AxisType.UPCAST)),)))
    a = Tensor([1., 2, 3, 4])
    self.assertEqual(Tensor.custom_kernel(Tensor.empty(4), a, fxn=kernel)[0].tolist(), [4., 5, 6, 7])

  def test_gemm_multi(self):
    devs = ("CPU:0", "CPU:1")
    N = 16
    a = Tensor.randn(N, N).shard_(devs, axis=0)
    b = Tensor.randn(N, N).to(devs)
    c = Tensor(Tensor.empty(N//2, N, device=devs).uop.unshard(0), device=devs)
    tst = Tensor.custom_kernel(c, a, b, fxn=custom_gemm)[0]
    self.assertTrue(tst.allclose(a@b, atol=1e-3).item())

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

  def test_multi_invalids_custom_kernel_no_copy(self):
    devs = ("CPU:0", "CPU:1")
    a = Tensor.ones(4, 4).shard(devs, axis=0).realize()
    c = Tensor(Tensor.invalids(2, 4, dtype=dtypes.float, device=devs).uop.unshard(0), device=devs)
    c = Tensor.custom_kernel(c, a, fxn=custom_add_one_kernel)[0]
    GlobalCounters.reset()
    c.realize()
    assert_kernel_count(len(devs))
    self.assertTrue((c == 2).all().item())

  def test_partial_invalid_store_keeps_uncovered_reads(self):
    x = Tensor([10., 20., 30., 40.]).realize()
    after = x.uop.after(x.uop.shrink(((0, 2),)).store(Invalid))
    self.assertEqual(Tensor(after).contiguous().tolist(), [10., 20., 30., 40.])

  def test_multi_after_invalid_store_dep_removed(self):
    x = Tensor.empty(4).uop
    self.assertEqual(Tensor(x.after(x.store(5), x.store(Invalid))).tolist(), [5]*4)

  def test_expand_view_invalid_assign_keeps_uncovered_reads(self):
    x = Tensor([[10., 11., 12., 13.], [20., 21., 22., 23.], [30., 31., 32., 33.], [40., 41., 42., 43.]]).realize()
    v = x[:1, :].expand(4, 4)
    v.assign(Tensor.invalids(4, 4, dtype=dtypes.float))
    self.assertEqual(v.contiguous().tolist(), [[10., 11., 12., 13.]]*4)

  @unittest.expectedFailure
  def test_gated_store_2d(self):
    # TODO: broken now, the valid is dropped. the valid on one index of the 2d C gates the whole store, only j == 0 writes C[i, 0]
    def kernel(C:UOp) -> UOp:
      i, j = UOp.range(4, 0), UOp.range(4, 1, AxisType.LOOP)
      return C[i.valid(j.eq(0)), 0].store((j+1).cast(C.dtype)).end(i, j).sink(arg=KernelInfo(opts_to_apply=()))
    self.assertEqual(Tensor.custom_kernel(Tensor.empty(4, 4), fxn=kernel)[0][:, 0].tolist(), [1.]*4)

  @unittest.skipIf(not Device[Device.DEFAULT].renderer.has_shared, "LOCAL buffer needs shared memory")
  @unittest.expectedFailure
  def test_gated_local_store_2d(self):
    # TODO: broken now, the valid is dropped. the valid on one index of the 2d LOCAL tmp gates the whole store, only j == 0 writes tmp[i, 0]
    def kernel(C:UOp) -> UOp:
      i, j = UOp.range(4, 0), UOp.range(4, 1, AxisType.LOOP)
      tmp = UOp.placeholder((4, 4), dtypes.float, slot=0, addrspace=AddrSpace.LOCAL)
      st = tmp[i.valid(j.eq(0)), 0].store((j+1).cast(dtypes.float)).end(j)
      return C[i].store(tmp.after(st)[i, 0]).end(i).sink(arg=KernelInfo(opts_to_apply=()))
    self.assertEqual(Tensor.custom_kernel(Tensor.empty(4), fxn=kernel)[0].tolist(), [1.]*4)

  @unittest.expectedFailure
  def test_gated_load_2d(self):
    # TODO: broken now, the valid is dropped. the valid on one index of the 2d A gates the whole load, it is 0 where j != 0
    def kernel(C:UOp, A:UOp) -> UOp:
      i, j = UOp.range(4, 0), UOp.range(4, 1)
      return C[i, j].store(A[i.valid(j.eq(0)), 0]).end(i, j).sink(arg=KernelInfo(opts_to_apply=()))
    a = Tensor.arange(16).reshape(4, 4).float().contiguous().realize()
    self.assertEqual(Tensor.custom_kernel(Tensor.empty(4, 4), a, fxn=kernel)[0].tolist(), a[:, :1].pad(((0, 0), (0, 3))).tolist())

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
  def test_custom_kernel_source_copy(self):
    from tinygrad.codegen import do_to_program
    def custom_source(out:UOp, inp:UOp) -> UOp:
      prg_uop = do_to_program(custom_add_one_kernel(out, inp), Device[out.device].renderer)
      # construct a plain Ops.PROGRAM
      sink = UOp.sink(out.base, inp.base, arg=KernelInfo("add_one_1"))
      return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(sink.toposort())),)+prg_uop.src[2:])
    out = Tensor([-1]).realize()
    cpu_src = Tensor([2], device="CPU").realize()
    out = Tensor.custom_kernel(out, cpu_src.to(out.device), fxn=custom_source)[0]
    cp = out.to("CPU").realize()
    self.assertEqual(out.tolist(), [3])
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

  @unittest.skip("this shouldn't be expected to work")
  def test_inplace_transpose(self):
    def custom_assign_row_max_kernel(A:UOp) -> UOp:
      row = UOp.range(A.shape[0], 0)
      col = UOp.range(A.shape[1], 1)
      return A[row, col].store(A[row].max(axis=0)).end(col).end(row).sink(arg=KernelInfo(name=f"assign_row_max_{A.numel()}"))
    a = Tensor.arange(4).clone().realize()
    a = Tensor.custom_kernel(a.reshape(2, 2).T, fxn=custom_assign_row_max_kernel)[0]
    self.assertEqual(a.flatten().tolist(), [2, 2, 3, 3])
    self.assertEqual(a.shape, (2, 2))

  @unittest.skipUnless(isinstance(Device[Device.DEFAULT].renderer, (CStyleLanguage, LLVMRenderer)) and Device.DEFAULT == "CPU",
                       "calls in kernels render on CPU")
  def test_call_in_kernel(self):
    def call_add(C:UOp, A:UOp) -> UOp:
      i = UOp.range(A.numel(), 0)
      return C[i].store(A[i] + 1).end(i)

    def call_sum(C:UOp, A:UOp) -> UOp:
      i = UOp.range(A.numel(), 0)
      return C[0].store(A[i].reduce(i, arg=Ops.ADD))

    def call_add_sum(C:UOp, A:UOp) -> UOp:
      tmp = UOp.alloc_like(A, addrspace=AddrSpace.REG)
      add_call = call_add(UOp.param(0, A.dtype, (N,), addrspace=AddrSpace.REG), A.param_like(1)).sink().call(tmp, A, name="add")
      tmp2 = UOp.alloc_like(A, addrspace=AddrSpace.REG)
      add_call2 = call_add(UOp.param(0, A.dtype, (N,), addrspace=AddrSpace.REG),
                           UOp.param(1, A.dtype, (N,), addrspace=AddrSpace.REG)).sink().call(tmp2, tmp.after(add_call), name="add2")
      sum_call = call_sum(C.param_like(0), UOp.param(1, A.dtype, (N,), addrspace=AddrSpace.REG)).sink().call(C, tmp2.after(add_call2), name="sum")
      return sum_call.sink(arg=KernelInfo(name="call_in_kernel"))

    N = getenv("N", 4)
    a = Tensor.arange(N).clone().realize()
    out = Tensor.custom_kernel(Tensor.empty_like(a), a, fxn=lambda C,A: call_add(C, A).sink(arg=KernelInfo(name="add")))[0]
    out = Tensor.custom_kernel(Tensor.empty(1, dtype=a.dtype), out, fxn=lambda C,A: call_sum(C, A).sink(arg=KernelInfo(name="sum")))[0]
    self.assertEqual(out.tolist(), [N*(N+1)//2])
    out = Tensor.custom_kernel(Tensor.empty(1, dtype=a.dtype), a, fxn=call_add_sum)[0]
    self.assertEqual(out.tolist(), [N*(N+3)//2])

  @unittest.skipUnless((isinstance(Device[Device.DEFAULT].renderer, CStyleLanguage) or Device.DEFAULT == "PYTHON") and
                       Device.DEFAULT != "WEBGPU", "binary not supported on this backend")
  def test_binary(self):
    payload = bytes(range(256))
    def kernel(out:UOp):
      i = UOp.range(len(payload), 0)
      data = UOp(Ops.BINARY, arg=payload)
      return out[i].store(data[i]).end(i).sink(arg=KernelInfo(name="binary", opts_to_apply=()))
    self.assertEqual(Tensor.empty(len(payload), dtype=dtypes.uint8).custom_kernel(fxn=kernel)[0].tolist(), list(payload))

@unittest.skipUnless(Device.DEFAULT == "CPU" and isinstance(Device[Device.DEFAULT].renderer, (CStyleLanguage, LLVMRenderer)),
                     "calls in kernels render on CPU")
class TestCallInKernel(unittest.TestCase):
  def test_nested_call(self):
    @uopfunc
    def incr(out:UOp, A:UOp):
      i = UOp.range(A.shape[0], 0)
      return out[i].store(A[i]+1).end(i).sink()

    @uopfunc
    def square(out:UOp, A:UOp):
      tmp0, tmp1 = (UOp.alloc_like(A, addrspace=AddrSpace.REG) for _ in range(2))
      call0 = incr(tmp0, A)
      call1 = incr(tmp1, A.after(call0))
      assert call0.body is call1.body
      i = UOp.range(A.shape[0], 1)
      return out[i].store(tmp1.after(call1)[i]**2).end(i).sink()

    def kernel(C:UOp, A:UOp):
      tmp = UOp.alloc_like(A, addrspace=AddrSpace.REG)
      call = square(tmp, A)
      i = UOp.range(A.shape[0], 2)
      return C[i].store(tmp.after(call)[i]+10).end(i).sink(arg=KernelInfo(name="nested_calls"))

    a = Tensor.arange(4).clone().realize()
    out = Tensor.custom_kernel(Tensor.empty_like(a), a, fxn=kernel)[0]
    self.assertEqual(out.tolist(), [11, 14, 19, 26])

  def test_call_loop_mini(self, opts=()):
    @uopfunc
    def mul(p:UOp, q:UOp): return p[0].store(q[0]*3).sink()

    def kernel(C:UOp, A:UOp):
      i = UOp.range(4, 0)
      call = mul(C[i], A[i])
      return call.end(i).sink(arg=KernelInfo(name="call_loop_mini", opts_to_apply=opts))
    a = Tensor([1, 2, 3, 4], dtype=dtypes.int).realize()
    out = Tensor.custom_kernel(Tensor.zeros(4, dtype=dtypes.int).clone().realize(), a, fxn=kernel)[0]
    self.assertEqual(out.tolist(), [3, 6, 9, 12])

  def test_call_with_gated_store(self):
    @uopfunc
    def gated_store(out:UOp, idx:UOp):
      i = idx[0]
      return out[i.valid((i >= 0) & (i < out.shape[0]))].store(7).sink()

    def kernel(C:UOp, A:UOp):
      tmp = UOp.alloc_like(C, addrspace=AddrSpace.REG)
      i = UOp.range(C.shape[0], 0)
      init = tmp[i].store(0).end(i)
      call = gated_store(tmp.after(init), A)
      j = UOp.range(C.shape[0], 1)
      return C[j].store(tmp.after(call)[j] + 1).end(j).sink(arg=KernelInfo(name="call_with_gated_store", opts_to_apply=()))

    a = Tensor([2], dtype=dtypes.int).realize()
    out = Tensor.custom_kernel(Tensor.zeros(4, dtype=dtypes.int).clone().realize(), a, fxn=kernel)[0]
    self.assertEqual(out.tolist(), [1, 1, 8, 1])

  @unittest.expectedFailure
  def test_call_loop_mini_opts(self): self.test_call_loop_mini(opts=None)

  @unittest.expectedFailure
  def test_call_loop_split(self): self.test_call_loop_mini((Opt(OptOps.SPLIT, 0, (4, AxisType.UPCAST)),))

  @unittest.expectedFailure
  def test_call_loop_pad(self):
    @uopfunc
    def add(p:UOp, q:UOp): return p[0].store(p[0]+q[0]).sink()

    def kernel(C:UOp, A:UOp):
      i = UOp.range(3, 0)
      call = add(C[i], A[i])
      return call.end(i).sink(arg=KernelInfo(name="call_loop_padto", opts_to_apply=(Opt(OptOps.PADTO, 0, 4),)))

    a = Tensor([1, 2, 3, 4], dtype=dtypes.int).realize()
    out = Tensor.custom_kernel(Tensor.zeros(4, dtype=dtypes.int).clone().realize(), a, fxn=kernel)[0]
    self.assertEqual(out.tolist(), [1, 2, 3, 0])

  def test_uopfunc(self):
    @uopfunc
    def axpy(out:UOp, x:UOp, a:int) -> UOp: # a is a trace-time constant: two bodies, two C functions
      i = UOp.range(x.shape[0], 0)
      return out[i].store(out[i] + x[i] * a).end(i).sink()

    def kernel(C:UOp, A:UOp) -> UOp:
      C = C.after(axpy(C.after(axpy(C, A, 2)), A, 3))
      i = UOp.range(A.shape[0], 1) # a loop after the calls
      return C[i].store(C[i] + 1).end(i).sink(arg=KernelInfo(name="uopfunc", opts_to_apply=()))

    a = Tensor([1, 2, 3, 4], dtype=dtypes.int).realize()
    out = Tensor.custom_kernel(Tensor.zeros(4, dtype=dtypes.int).contiguous().realize(), a, fxn=kernel)[0]
    self.assertEqual(out.tolist(), [6, 11, 16, 21])

class TestCustomKernelInput(unittest.TestCase):
  def _test_mop(self, mop_fxn, max_kernels):
    # default: input is BUFFER
    x = mop_fxn(Tensor.arange(32).clone().realize())
    y = Tensor.custom_kernel(Tensor.empty_like(x), x, fxn=custom_add_one_kernel)[0]
    GlobalCounters.reset()
    y.realize()
    assert_kernel_count(max_kernels)
    self.assertEqual(y.tolist(), x.add(1).tolist())
    # same test with @function, input is PARAM
    from tinygrad import function
    x0 = Tensor.arange(32).clone().realize()
    @function(precompile=True)
    def run(a:Tensor) -> Tensor:
      xv = mop_fxn(a)
      y = Tensor.invalids(*xv.shape, dtype=xv.dtype, device=a.device)
      return Tensor.custom_kernel(y, xv, fxn=custom_add_one_kernel)[0]
    GlobalCounters.reset()
    y = run(x0).realize()
    assert_kernel_count(max_kernels)
    self.assertEqual(y.tolist(), mop_fxn(x0).add(1).tolist())

  def test_reshape(self): self._test_mop(lambda x: x.reshape(16, 2), max_kernels=1)
  def test_permute(self): self._test_mop(lambda x: x.reshape(4, 8).T, max_kernels=2)
  def test_double_permute(self): self._test_mop(lambda x: x.reshape(4, 8).T.T, max_kernels=1)
  def test_shrink(self): self._test_mop(lambda x: x[:4], max_kernels=1)
  def test_pad(self): self._test_mop(lambda x: x[:4].pad(((0, 4),)), max_kernels=2)
  def test_flip(self): self._test_mop(lambda x: x.flip(0), max_kernels=2)
  def test_offset_shrink(self): self._test_mop(lambda x: x[4:8], max_kernels=2)
  def test_2d_shrink(self): self._test_mop(lambda x: x.reshape(4, 8)[:, 2:6], max_kernels=2)
  def test_expand(self): self._test_mop(lambda x: x.reshape(16, 2)[:, :1].expand(16, 2), max_kernels=2)

class TestUnshardIndex(unittest.TestCase):
  """Regression tests for INDEX on UNSHARD (fragment) resolution in schedule/multi.py.

  A fragment is a per-thread REG buffer wrapped in UNSHARD over LOCAL thread ranges.
  index_multi must resolve an INDEX on the UNSHARD view into an INDEX on the per-thread
  shard. Two ownership patterns must work:
    contiguous: idx = rng*shard_sz + local   (thread rng owns [rng*shard_sz, ...))
    strided:    idx = rng + ir*shard_sz      (thread rng owns {rng, rng+shard_sz, ...})
  """
  def _run(self, kernel, shape=(8, 8)):
    c = Tensor.empty(*shape)
    out = Tensor.custom_kernel(c, fxn=kernel)[0]
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
      ir = UOp.range(8, 1, AxisType.LOOP)
      j = UOp.range(8, 2, AxisType.LOOP)
      # 8x8 fragment, 8 threads -> 64x8 full tile. thread ty owns rows [ty*8, ty*8+8).
      frag = UOp.placeholder((8, 8), dtypes.float32, 0, AddrSpace.REG).unshard((0,), (ty,))
      return C[ty*8 + ir, j].store(frag[ty*8 + ir, j]).end(j, ir, ty).sink(arg=KernelInfo(name="contig_frag"))
    out = self._run(kernel, (64, 8))
    assert out.shape == (64, 8)

  @unittest.skipIf(not Device[Device.DEFAULT].renderer.has_local, "fragment tests need LOCAL ranges")
  def test_strided_fragment_index(self):
    # thread ty owns rows {ty, ty+8, ty+16, ty+24, ..., ty+56} of a 64-row fragment --
    # strided ownership. idx = ty + ir*8 where shard_sz=8 (8 threads, shard rows=8).
    # The contiguous check (idx - rng*shard_sz) fails; the strided check
    # (idx-rng) % shard_sz == 0 must succeed. This is the pattern the index_multi fix adds.
    def kernel(C:UOp) -> UOp:
      ty = UOp.range(8, 0, AxisType.LOCAL)
      ir = UOp.range(8, 1, AxisType.LOOP)
      j = UOp.range(8, 2, AxisType.LOOP)
      # 8x8 fragment, 8 threads -> 64x8 full tile. thread ty owns rows {ty, ty+8, ..., ty+56}.
      frag = UOp.placeholder((8, 8), dtypes.float32, 0, AddrSpace.REG).unshard((0,), (ty,))
      return C[ty + ir*8, j].store(frag[ty + ir*8, j]).end(j, ir, ty).sink(arg=KernelInfo(name="strided_frag"))
    out = self._run(kernel, (64, 8))
    assert out.shape == (64, 8)

  def test_fragment_index_cannot_shard(self):
    # thread ty indexing rows [ty, ty+8) overlaps with other threads' rows -- this matches neither
    # the contiguous nor the strided ownership pattern, so index_multi must raise.
    def kernel(C:UOp) -> UOp:
      ty = UOp.range(8, 0, AxisType.LOCAL)
      ir = UOp.range(8, 1, AxisType.LOOP)
      j = UOp.range(8, 2, AxisType.LOOP)
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
  """Tests for ALU on (fragment) UNSHARD values in schedule/multi.py's alu_multi.

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
  """Tests for STORE of a sharded value into an unsharded dest (store_value_multi in schedule/multi.py).

  Every shard stores its value into its own contiguous sub-view of the dest, one SHRINK per sharded axis.
  """
  @unittest.skipIf(not Device[Device.DEFAULT].renderer.has_local, "fragment tests need LOCAL ranges")
  def test_store_unshard_value(self):
    # single-axis: 8 threads each own 8 values of the 64-value output tile
    def kernel(C:UOp) -> UOp:
      ty = UOp.range(8, 0, AxisType.LOCAL)
      frag = UOp.placeholder((8,), dtypes.float32, 0, AddrSpace.LOCAL).unshard((0,), (ty,))
      v = frag.after(frag.store(0.0)) + 2.5
      return C.store(v).end(ty).sink(arg=KernelInfo(name="store_unshard", opts_to_apply=()))
    out = _run_fragment_kernel(self, kernel, (64,))
    np.testing.assert_allclose(out, 2.5)

  @unittest.skipIf(not Device[Device.DEFAULT].renderer.has_local, "fragment tests need LOCAL ranges")
  def test_store_unshard_value_2axis(self):
    # two sharded axes (the gemm fragment layout): thread (ty, tx) owns the (2, 1, 1, 2) sub-view of the
    # (2, 4, 2, 2) output tile; the store must SHRINK dest on both sharded axes
    def kernel(C:UOp, A:UOp) -> UOp:
      ty = UOp.range(4, 0, AxisType.LOCAL)
      tx = UOp.range(2, 1, AxisType.LOCAL)
      frag = UOp.placeholder((2, 1, 1, 2), dtypes.float32, 0, AddrSpace.REG).unshard((1, 2), (ty, tx))
      v = frag.after(frag.store(0.0)) + A
      return C.store(v).end(tx, ty).sink(arg=KernelInfo(name="store_unshard_2axis", opts_to_apply=()))
    a = Tensor(np.arange(32, dtype=np.float32).reshape(2, 4, 2, 2))
    out = _run_fragment_kernel(self, kernel, (2, 4, 2, 2), inputs=(a,))
    np.testing.assert_allclose(out, a.numpy(), atol=1e-4)

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

  def test_uop_sum_multi_axis(self):
    a = Tensor.arange(24).reshape(2, 3, 4).float()
    ref = a.sum(axis=(0, 2)).numpy()
    result = Tensor(a.uop.sum(axis=(0, 2))).numpy()
    for i in range(3): self.assertAlmostEqual(result[i], ref[i])

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
