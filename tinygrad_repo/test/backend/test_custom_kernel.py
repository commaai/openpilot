import unittest
from tinygrad import Tensor, UOp, GlobalCounters, Context, Device
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.uop.ops import KernelInfo, AxisType, Ops

# **** kernels ****

def custom_arange_kernel(C:UOp) -> UOp:
  i = UOp.range(C.shape[0], 0)
  return C[i].store(i.cast(C.dtype.base)).end(i).sink(arg=KernelInfo(name=f"custom_arange_{C.shape[0]}"))

def custom_eye_kernel(C:UOp) -> UOp:
  i = UOp.range(C.shape[0], 0)
  j = UOp.range(C.shape[1], 1)
  return C[i, j].store((i.eq(j)).cast(C.dtype.base)).end(i, j).sink(arg=KernelInfo(name=f"custom_eye_{C.numel()}"))

def custom_add_one_kernel(B:UOp, A:UOp) -> UOp:
  A,B = A.flatten(), B.flatten()
  assert B.numel() == A.numel()
  i = UOp.range(A.numel(), 0)
  return B[i].store(A[i] + 1).end(i).sink(arg=KernelInfo(name=f"add_one_{A.numel()}"))

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
  i, j, k = UOp.range(C.shape[0], 0), UOp.range(C.shape[1], 1), UOp.range(A.shape[1], 2, axis_type=AxisType.REDUCE)
  C = C[i, j].set(0.0)
  C = C[i, j].set(C.after(k)[i, j] + A[i, k] * B[k, j], end=k)
  prog = C.end(i, j)
  return prog.sink(arg=KernelInfo(name=f"custom_gemm_{C.shape[0]}_{C.shape[1]}_{A.shape[1]}", opts_to_apply=()))

def custom_sum(B:UOp, A:UOp) -> UOp:
  i = UOp.range(A.shape[0], 0, axis_type=AxisType.REDUCE)
  B = B[0].set(0.0)
  B = B[0].set(B.after(i)[0] + A[i], end=i)
  return B.sink(arg=KernelInfo(name=f"custom_sum_{A.shape[0]}", opts_to_apply=()))

def flip_contract_kernel(dest:UOp, src:UOp):
  i = UOp.range(dest.shape[0], 0)
  j = UOp.range(dest.shape[1], 1, AxisType.UPCAST)
  vec = src[i, j].contract(j)
  store = UOp.group(*[dest[i, k].store(vec.gep(3-k)) for k in range(4)])
  return store.end(i, j).sink(arg=KernelInfo(name=f"flip_contract_{dest.numel()}", opts_to_apply=()))

def slice_sum_kernel(dest:UOp, src:UOp):
  G = UOp.range(src.shape[0], 0)
  slice_src = src[G, :]
  reg = UOp.placeholder((1,), dest.dtype.base, 0, addrspace=AddrSpace.REG)
  reg = reg.after(G)[0].set(0)
  R = UOp.range(src.shape[1], 1, AxisType.REDUCE)
  reg = reg[0].set(reg.after(R)[0] + slice_src[R], end=R)
  ast = dest[G].set(reg[0], end=G)
  return ast.sink(arg=KernelInfo(name=f"slice_sum_{src.shape[0]}_{src.shape[1]}", opts_to_apply=()))

def simple_qkv_kernel(O:UOp, Q:UOp, K:UOp, V:UOp) -> UOp:
  # attention without softmax
  N, d = Q.shape[0], Q.shape[1]

  i = UOp.range(N, 0)  # output row
  d_out = UOp.range(d, 1)  # output column
  j = UOp.range(N, 2, axis_type=AxisType.REDUCE)

  k_inner = UOp.range(d, 3, axis_type=AxisType.REDUCE)
  qk_acc = UOp.placeholder((1,), Q.dtype.base, 0, addrspace=AddrSpace.REG)
  qk_acc = qk_acc.after(i, j)[0].set(0.0)
  qk_acc = qk_acc[0].set(qk_acc.after(k_inner)[0] + Q[i, k_inner] * K[j, k_inner], end=k_inner)
  qk_score = qk_acc[0] / (d ** 0.5)

  out_acc = UOp.placeholder((1,), Q.dtype.base, 1, addrspace=AddrSpace.REG)
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

  def test_simple_sharded(self):
    devs = ("CPU:0", "CPU:1")

    a = Tensor.ones(16, 16).contiguous().shard(devs, axis=0)
    b = Tensor.ones(16, 16).contiguous().shard(devs, axis=0)
    # ugly construction to get a sharded empty tensor
    c = Tensor(Tensor.empty(8, 16, device=devs).uop.multi(0), device=devs)
    c = Tensor.custom_kernel(c,a,b, fxn=custom_elementwise_add_kernel)[0]
    out = c.flatten().tolist()
    assert all(x == 2 for x in out), "all 2"

  def test_sharded_add_one(self):
    # PYTHON backend explicitly checks for OOB access for wrong multi shape regression
    devs = ("PYTHON:0", "PYTHON:1")
    a = Tensor.ones(4, 4).contiguous().shard(devs, axis=0)
    c = Tensor(Tensor.empty(2, 4, device=devs).uop.multi(0), device=devs)
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
    ref = Tensor.eye(1024).contiguous().realize()
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

  def test_sum(self):
    a = Tensor([1.0, 2, 3, 4, 5])
    tst = Tensor.empty(1)
    b = Tensor.custom_kernel(tst, a, fxn=custom_sum)[0]
    self.assertEqual(b.item(), 15)

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
    err = (tst - (a@b)).square().max()
    self.assertLess(err.item(), 1e-6)

  def test_gemm_multi(self):
    devs = ("CPU:0", "CPU:1")
    N = 16
    a = Tensor.randn(N, N).shard_(devs, axis=0)
    b = Tensor.randn(N, N).to(devs)
    c = Tensor(Tensor.empty(N//2, N, device=devs).uop.multi(0), device=devs)
    tst = Tensor.custom_kernel(c, a, b, fxn=custom_gemm)[0]
    err = (tst - (a@b)).square().max()
    self.assertLess(err.item(), 1e-6)

  def test_gemm_backward_custom(self): self.test_gemm_backward(True)
  # NOTE: grad_fxn doesn't work with pyrender
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

    err = (tst - ref).square().max()
    self.assertLess(err.item(), 1e-6)

    err = (grad_a - real_grad_a).square().max()
    self.assertLess(err.item(), 1e-6)

    err = (grad_b - real_grad_b).square().max()
    self.assertLess(err.item(), 1e-6)

  def test_simple_qkv(self):
    N, d = 8, 4
    Q = Tensor.randn(N, d)
    K = Tensor.randn(N, d)
    V = Tensor.randn(N, d)
    O = Tensor.empty(N, d)

    O_custom = Tensor.custom_kernel(O, Q, K, V, fxn=lambda o,q,k,v: simple_qkv_kernel(o,q,k,v))[0]
    O_ref = ((Q @ K.T) / (d ** 0.5)) @ V

    Tensor.realize(O_custom, O_ref)
    err = (O_custom - O_ref).square().max()
    self.assertLess(err.item(), 1e-6)

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
      self.assertTrue(len(prg.arg.globals) > 0, f"empty kernel compiled (no globals): name={prg.arg.name}")

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
    x = Tensor.arange(32).reshape(8, 4).realize()
    y = Tensor.empty_like(x)
    y = Tensor.custom_kernel(y, x, fxn=custom_add_one_kernel)[0]
    if use_custom:
      z = Tensor.empty_like(x)
      z = Tensor.custom_kernel(y, y.T.T, fxn=custom_add_one_kernel)[0]
    else: z = y.T.T+1
    GlobalCounters.reset()
    z.realize()
    self.assertEqual(GlobalCounters.kernel_count, 2)
    self.assertEqual(z.tolist(), x.add(2).tolist())

  @unittest.expectedFailure
  def test_custom_kernel_sched_copy(self): self.test_custom_kernel_sched(use_custom=True)

  @unittest.expectedFailure
  def test_sliced_buffer_function(self):
    x = Tensor.arange(32).reshape(8, 4).realize()
    from tinygrad import function
    @function(precompile=True)
    def run(x:Tensor) -> Tensor:
      y = Tensor.invalids(*x.shape, dtype=x.dtype)
      return Tensor.custom_kernel(y, x, fxn=custom_add_one_kernel)[0]
    GlobalCounters.reset()
    y = run(x[0]).realize()
    # it's copying the input and the output
    self.assertEqual(GlobalCounters.kernel_count, 1)
    self.assertEqual(y.tolist(), [1, 2, 3, 4])

  @Context(DEV="CPU")
  def test_simple_from_source(self):
    a = Tensor([0., 1., 2.]).realize()

    src = "void test_src(float* restrict a) { a[0] = 1.0; }"
    # TODO: it currently requires a compiler for Ops.BINARY
    from tinygrad.device import Device
    binary = Device[a.device].renderer.compiler.compile(src)
    def custom_src_kernel(A:UOp) -> UOp:
      sink = UOp.sink(A, arg=KernelInfo(name="test_src"))
      return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="CPU"), UOp(Ops.LINEAR, src=tuple(sink.toposort())),
                                   UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=binary)))

    a = Tensor.custom_kernel(a, fxn=custom_src_kernel)[0]
    self.assertEqual(a.tolist(), [1., 1., 2.])

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

  def test_uop_sum_keepdim(self):
    a = Tensor.arange(6).reshape(2, 3).float()
    result = Tensor(a.uop.sum(axis=1, keepdim=True))
    assert result.shape == (2, 1)

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
