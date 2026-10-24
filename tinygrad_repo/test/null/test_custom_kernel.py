import unittest
from tinygrad import Tensor, UOp, GlobalCounters, Device
from tinygrad.uop.ops import KernelInfo, AxisType, Ops
from tinygrad.codegen.opt import Opt, OptOps, KernelOptError
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.renderer import Target
from tinygrad.renderer.llvmir import AMDLLVMRenderer
from tinygrad.codegen import to_program
from test.helpers import assert_kernel_count

def custom_elementwise_add_kernel(C:UOp, A:UOp, B:UOp) -> UOp:
  C,A,B = C.flatten(), A.flatten(), B.flatten()
  i = UOp.range(C.numel(), 0)
  return C[i].store(A[i]+B[i]).end(i).sink(arg=KernelInfo(name=f"custom_add_kernel_{C.numel()}")).simplify()

def custom_elementwise_addmul_kernel(C:UOp, D:UOp, A:UOp, B:UOp) -> UOp:
  C,D,A,B = C.flatten(), D.flatten(), A.flatten(), B.flatten()
  assert C.numel() == D.numel()
  i = UOp.range(C.numel(), 0)
  store_c = C[i].store(A[i]+B[i])
  store_d = D[i].store(A[i]*B[i])
  return UOp.group(store_c, store_d).end(i).sink(arg=KernelInfo(name=f"custom_addmul_kernel_{C.numel()}")).simplify()

def custom_gemm(C:UOp, A:UOp, B:UOp) -> UOp:
  assert A.shape[1] == B.shape[0]
  i, j, k = UOp.range(C.shape[0], 0), UOp.range(C.shape[1], 1), UOp.range(A.shape[1], 2, axis_type=AxisType.LOOP)
  C = C[i, j].set(0.0)
  prog = C[i, j].store(C.after(k)[i, j] + A[i, k] * B[k, j]).end(k).end(i, j)
  return prog.sink(arg=KernelInfo(name=f"custom_gemm_{C.shape[0]}_{C.shape[1]}_{A.shape[1]}", opts_to_apply=()))

class TestCustomKernel(unittest.TestCase):
  def test_gemm_group_refused(self):
    # k is a serial LOOP: custom_gemm has no Ops.REDUCE
    a, b, c = Tensor.empty(16, 16), Tensor.empty(16, 16), Tensor.empty(16, 16)
    ast = Tensor.custom_kernel(c, a, b, fxn=custom_gemm)[0].schedule_linear().src[-1].src[0]
    with self.assertRaises(KernelOptError):
      Scheduler(ast, AMDLLVMRenderer(Target("AMD", arch="gfx1100"))).apply_opt(Opt(OptOps.SPLIT, 2, (4, AxisType.LOCAL)))

  def test_gemm_unroll_refused(self):
    # k is a serial LOOP, so the expander has nothing to contract the stores back with
    a, b, c = Tensor.empty(16, 16), Tensor.empty(16, 16), Tensor.empty(16, 16)
    ast = Tensor.custom_kernel(c, a, b, fxn=custom_gemm)[0].schedule_linear().src[-1].src[0]
    with self.assertRaises(KernelOptError):
      Scheduler(ast, Device[Device.DEFAULT].renderer).apply_opt(Opt(OptOps.SPLIT, 2, (4, AxisType.UPCAST)))

  def test_upcast_split_range(self):
    # j%2 splits j into two UPCAST ranges, the expander expands both, so no loop is left
    def kernel(C:UOp, A:UOp) -> UOp:
      i, j = UOp.range(4, 0), UOp.range(8, 1, AxisType.UPCAST)
      return C[i, j].store(A[i, j] + (j%2).cast(A.dtype)).end(i, j).sink(arg=KernelInfo(opts_to_apply=()))
    ast = Tensor.custom_kernel(Tensor.empty(4, 8), Tensor.empty(4, 8), fxn=kernel)[0].schedule_linear().src[-1].src[0]
    uops = to_program(ast, AMDLLVMRenderer(Target("AMD", arch="gfx1100"))).src[1].src
    self.assertEqual(len([u for u in uops if u.op is Ops.RANGE]), 0)

  def test_loop_acc_gemm_tc_refused(self):
    # ACC[j] += A[t,:] @ B[:,j] over t: the recurrence on ACC makes t a serial LOOP, so no tensor core may split it
    ren = AMDLLVMRenderer(Target("AMD", arch="gfx1100"))
    i, tc = next((i, tc) for i, tc in enumerate(ren.tensor_cores) if tc.dtype_in is dtypes.half and tc.dtype_out is dtypes.float)
    def kernel(ACC:UOp, A:UOp, B:UOp) -> UOp:
      t, j, k = UOp.range(A.shape[0], 0, AxisType.LOOP), UOp.range(B.shape[1], 1), UOp.range(A.shape[1], 2)
      mm = (A[t, k] * B[k, j]).cast(dtypes.float).reduce(k, arg=Ops.ADD)
      return ACC[j].store(ACC.after(t)[j] + mm).end(t).end(j).sink(arg=KernelInfo(opts_to_apply=(Opt(OptOps.TC, 0, (i, 0, 1)),)))
    N, M, K = tc.dims
    a, b, acc = Tensor.empty(M, K, dtype=dtypes.half), Tensor.empty(K, N, dtype=dtypes.half), Tensor.empty(N, dtype=dtypes.float)
    ast = Tensor.custom_kernel(acc, a, b, fxn=kernel)[0].schedule_linear().src[-1].src[0]
    with self.assertRaises(KernelOptError): to_program(ast, ren)

  def test_split_loop_local_barrier(self):
    # t%2 splits the t loop. tmp is stored and loaded in the loop, so the end of the loop still needs a barrier
    def kernel(C:UOp, A:UOp) -> UOp:
      l, t = UOp.range(4, 0, AxisType.LOCAL), UOp.range(8, 1, AxisType.LOOP)
      tmp = UOp.placeholder((4,), dtypes.float, slot=0, addrspace=AddrSpace.LOCAL)
      v = tmp.after(tmp[l].store(A[t%2, l]))[(l+1)%4]
      return C[l].store(C.after(t)[l] + v).end(t).end(l).sink(arg=KernelInfo(opts_to_apply=()))
    ast = Tensor.custom_kernel(Tensor.empty(4), Tensor.empty(2, 4), fxn=kernel)[0].schedule_linear().src[-1].src[0]
    uops = to_program(ast, AMDLLVMRenderer(Target("AMD", arch="gfx1100"))).src[1].src
    self.assertEqual(len([u for u in uops if u.op is Ops.BARRIER]), 2)

  def test_loop_local_barrier_inner_loop_load(self):
    # tmp is loaded inside the k loop. the end of the t loop still needs a barrier, and it leaves no range open
    def kernel(C:UOp, A:UOp) -> UOp:
      l, t, k = UOp.range(4, 0, AxisType.LOCAL), UOp.range(8, 1, AxisType.LOOP), UOp.range(4, 2)
      tmp = UOp.placeholder((4,), dtypes.float, slot=0, addrspace=AddrSpace.LOCAL)
      v = tmp.after(tmp[l].store(A[t, l]))[k].reduce(k, arg=Ops.ADD)
      return C[l].store(C.after(t)[l] + v).end(t).end(l).sink(arg=KernelInfo(opts_to_apply=()))
    ast = Tensor.custom_kernel(Tensor.empty(4), Tensor.empty(8, 4), fxn=kernel)[0].schedule_linear().src[-1].src[0]
    prg = to_program(ast, AMDLLVMRenderer(Target("AMD", arch="gfx1100")))
    self.assertEqual(prg.ranges, {})
    self.assertEqual(len([u for u in prg.src[1].src if u.op is Ops.BARRIER]), 2)

  def test_local_barrier_after_ended_loop(self):
    # tmp is read after the k loop that stored it. the barrier before the read leaves no range open
    def kernel(C:UOp, A:UOp) -> UOp:
      k = UOp.range(4, 0, AxisType.LOOP)
      tmp = UOp.placeholder((4,), dtypes.float, slot=0, addrspace=AddrSpace.LOCAL)
      tmp = tmp.after(k)[k].set(A[k], end=k)
      return C[0].store(tmp[0]).sink(arg=KernelInfo(opts_to_apply=()))
    ast = Tensor.custom_kernel(Tensor.empty(1), Tensor.empty(4), fxn=kernel)[0].schedule_linear().src[-1].src[0]
    prg = to_program(ast, AMDLLVMRenderer(Target("AMD", arch="gfx1100")))
    self.assertEqual(prg.ranges, {})
    self.assertEqual(len([u for u in prg.src[1].src if u.op is Ops.BARRIER]), 1)

  def test_gemm_qkv(self):
    B, N, K_DIM, H_KV, REP, D = 2, 7, 6, 2, 2, 6
    H, QKV = H_KV * REP, H_KV * (REP + 2) * D

    x = Tensor.empty(B*N, K_DIM)
    w = Tensor.empty(K_DIM, QKV)
    qkv = Tensor.empty(B*N, QKV)

    qkv = Tensor.custom_kernel(qkv, x, w, fxn=custom_gemm)[0]
    qkv = qkv.reshape(B, N, H_KV, REP + 2, D)

    q = qkv[:, :, :, :REP, :].reshape(B, N, H, D).transpose(1, 2)
    k = qkv[:, :, :, REP, :].transpose(1, 2)
    v = qkv[:, :, :, REP + 1, :].transpose(1, 2)

    out = q.scaled_dot_product_attention(k, v, enable_gqa=True)

    GlobalCounters.reset()
    out.realize()
    assert_kernel_count(5)

  def test_multi_after_schedule_order(self):
    """Test correct scheduling order when custom_kernel has multiple outputs.

    custom_kernel with 4 arguments creates 4 AFTERs from the same kernel.
    The custom_kernel depends on both A2 and B2, so it must be scheduled after both.
    E only depends on A2, so E can run before custom_kernel finishes waiting for B2.

    Expected schedule order: [A2, B2, E, custom_addmul, final_sum]
    The custom_addmul kernel should be at index 3.
    """

    A, B = Tensor.empty(4, 4), Tensor.empty(4, 4)
    A2 = (A + 1).contiguous()                      # kernel 0: depends on A
    B2 = (B * 2).contiguous()                      # kernel 1: depends on B
    C, D = Tensor.empty(4, 4), Tensor.empty(4, 4)
    C, D, _, _ = Tensor.custom_kernel(C, D, A2, B2, fxn=custom_elementwise_addmul_kernel)  # depends on A2 AND B2
    E = (A2 * 3).contiguous()                      # kernel 2: depends only on A2
    result = (C + D + E).sum()                     # kernel 3: custom_addmul, then kernel 4: sum
    schedule = result.schedule_linear().src

    # Find the custom_addmul kernel position
    custom_idx = next((i for i, item in enumerate(schedule)
                       if hasattr(item.src[0], "arg") and hasattr(item.src[0].arg, "name")
                       and "custom_addmul" in item.src[0].arg.name), None)

    self.assertIsNotNone(custom_idx, "custom_addmul kernel not found in schedule")
    self.assertEqual(custom_idx, 3, f"custom_addmul should be at index 3, got {custom_idx}")

  def test_invalids_into_custom_kernel_no_empty_kernel(self):
    from tinygrad.engine.realize import compile_linear
    a = Tensor.full((4, 4), 3.).contiguous()
    b = Tensor.full((4, 4), 2.).contiguous()
    Tensor.realize(a, b)
    out = Tensor.invalids(*a.shape, dtype=a.dtype)
    out, *_ = Tensor.custom_kernel(out, a, b, fxn=custom_elementwise_add_kernel)
    compiled = compile_linear(out.schedule_linear())
    for call in compiled.src:
      prg = call.src[0]
      if prg.op is not Ops.PROGRAM: continue
      self.assertTrue(len(prg.arg.globals) > 0, f"empty kernel compiled (no globals): name={prg.src[0].arg.name}")

class TestUOpReduce(unittest.TestCase):
  def test_uop_sum_keepdim(self):
    a = Tensor.arange(6).reshape(2, 3).float()
    result = Tensor(a.uop.sum(axis=1, keepdim=True))
    assert result.shape == (2, 1)

if __name__ == '__main__':
  unittest.main()
