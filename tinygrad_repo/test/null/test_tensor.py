# tensor tests that pass on NULL backend (no copyout needed)
import unittest
import numpy as np
import torch
from tinygrad import Tensor, Device, dtypes
from tinygrad.dtype import DTYPES_DICT, DType
from tinygrad.uop.ops import UOp, Ops
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.renderer.nir import NIRRenderer
from tinygrad.codegen import to_program

x_init = np.random.randn(1,3).astype(np.float32)
W_init = np.random.randn(3,3).astype(np.float32)
m_init = np.random.randn(1,3).astype(np.float32)

class TestInferenceMode(unittest.TestCase):
  def test_inference(self):
    x = Tensor(x_init)
    m = Tensor(m_init)
    W = Tensor(W_init)
    tmp = x.mul(m)
    mm = tmp.matmul(W)
    out = mm.relu()
    out = out.sum()
    #out.backward()
    assert x.grad is None
    assert m.grad is None
    assert tmp.grad is None
    assert mm.grad is None
    assert W.grad is None

  def test_no_grad_mode_context_manager(self):
    x = Tensor(x_init)
    m = Tensor(m_init)
    W = Tensor(W_init)
    def f(x, m, W):
      tmp = x.mul(m)
      mm = tmp.matmul(W)
      out = mm.relu()
      out = out.sum()
      #out.backward()
      assert x.grad is None
      assert m.grad is None
      assert tmp.grad is None
      assert mm.grad is None
      assert W.grad is None
    f(x, m, W)

class TestIdxUpcast(unittest.TestCase):
  def _find_op(self, ast: UOp, op: Ops):
    if ast.op is op: return ast
    for src in ast.src:
      if (ret:=self._find_op(src, op)) is not None: return ret
  def _schedule_render(self, a: Tensor):
    linear, _ = a.linear_with_vars()
    for si in linear.src:
      ast = si.src[0]
      if ast.op is Ops.SINK:
        renderer = Device[si.src[1].buffer.device].renderer
        prg = to_program(ast, renderer)
        return tuple(prg.src[1].src)

  def _assert(self, dtype: DType, a: Tensor):
    uops = self._schedule_render(a)
    # Assert the dtype of the INDEX value, This will need be updated if UOp spec changes
    store = next(uop for uop in uops if uop.op is Ops.STORE)
    assert store.op is Ops.STORE
    idx = self._find_op(store, Ops.INDEX)
    # PTX and NIR turn Ops.INDEX into pointer arithmetic earlier than cstyle, plus it's already cast to int64
    if not isinstance(Device[Device.DEFAULT].renderer, (PTXRenderer, NIRRenderer)):
      assert idx.op is Ops.INDEX
      idx_val = idx.src[1]
      self.assertFalse(idx_val.overflows(idx_val.dtype))

  # use expand to generate kernel that uses large idx
  def do_op_then_assert(self, dtype: DType, dim1, dim2, dim3):
    self._assert(dtype, Tensor.empty(dim1, dim2, 1).expand(-1, -1, dim3).contiguous())

  @unittest.skipUnless(dtypes.long in Device[Device.DEFAULT].renderer.supported_dtypes(), "int64 is supported")
  def test_overflow(self):
    # 2**11, 2**11, 2**11 -> 2**33 will overflow when indexed
    self.do_op_then_assert(dtypes.long, 2048, 2048, 2048)

  @unittest.skipUnless(dtypes.long in Device[Device.DEFAULT].renderer.supported_dtypes(), "int64 is supported")
  def test_overflow_sym(self):
    self.do_op_then_assert(dtypes.long, 2048, 2048, UOp.variable("dim3", 1, 2048).bind(32))

  def test_regular(self):
    self.do_op_then_assert(dtypes.int, 64, 64, 64)

  def test_regular_sym(self):
    self.do_op_then_assert(dtypes.int, 256, 256, UOp.variable("dim3", 1, 64).bind(32))

  @unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, (PTXRenderer, NIRRenderer)), "PTX and NIR always converts Ops.INDEX to int64")
  def test_symfold(self):
    # This would cause an overflow, but after sym fold it's within int32
    a = Tensor.arange(65535).clone()
    uops = self._schedule_render(a)
    assert all(uop.dtype is not dtypes.long for uop in uops)

  @unittest.skipIf(dtypes.long in Device[Device.DEFAULT].renderer.supported_dtypes(), "int64 is supported")
  def test_int64_unsupported_overflow_sym(self):
    with self.assertRaises((KeyError, RuntimeError)):
      self.do_op_then_assert(dtypes.long, 2048, 2048, UOp.variable("dim3", 1, 2048).bind(32))

  @unittest.skipIf(dtypes.long in Device[Device.DEFAULT].renderer.supported_dtypes(), "int64 is supported")
  @unittest.expectedFailure  # bug in gpu dims limiting
  def test_int64_unsupported_overflow(self):
    with self.assertRaises((KeyError, RuntimeError)):
      self.do_op_then_assert(dtypes.long, 2048, 2048, 2048)

  @unittest.skip("This is kept for reference, it requires large memory to run")
  def test_overflow_kernel_run(self):
    # This creates a total of 2**31+10 elements, requiring at least 2147 MB memory to run
    # Modified example from issue 3271
    a = Tensor.empty(2**11, 2**11, 1, dtype=dtypes.int8).permute((2, 0, 1)).expand((2**9+10, -1, -1)).contiguous()
    a.realize()

class TestTensorUnique(unittest.TestCase):
  def test_empty_bufs_unique(self):
    a = Tensor.empty(10, 10).contiguous()
    b = Tensor.empty(10, 10).contiguous()
    Tensor.realize(a,b)
    self.assertIsNot(a.uop.buffer, b.uop.buffer)

  def test_zeros_bufs_unique_sep(self):
    a = Tensor.zeros(10, 10).contiguous()
    Tensor.realize(a)
    b = Tensor.zeros(10, 10).contiguous()
    Tensor.realize(b)
    self.assertIsNot(a.uop.buffer, b.uop.buffer)

  def test_zeros_bufs_unique(self):
    a = Tensor.zeros(10, 10).contiguous()
    b = Tensor.zeros(10, 10).contiguous()
    Tensor.realize(a,b)
    self.assertIsNot(a.uop.buffer, b.uop.buffer)

  def test_times_2_not_unique(self):
    a = Tensor.zeros(10, 10).contiguous()
    b = a * 2
    c = a * 2
    Tensor.realize(b,c)
    self.assertIs(b.uop.buffer, c.uop.buffer)

class TestRand(unittest.TestCase):
  def test_rand_large_tensor(self):
    # large tensor rand (num > uint32.max) should not crash in frontend
    Tensor.manual_seed(0)
    Tensor.rand(2**17, 2**17).schedule_linear()
    Tensor.rand(2**17, 2**17).schedule_linear()
    Tensor.rand(2**17, 2**17).schedule_linear()

class TestTensorConstLike(unittest.TestCase):
  def test_const_like_shape(self):
    t = Tensor.ones(3, 4)
    c = t.const_like(0)
    self.assertEqual(c.shape, (3, 4))
    self.assertEqual(c.dtype, t.dtype)

  def test_const_like_multi_device(self):
    devs = ("NULL:0", "NULL:1")
    t = Tensor.ones(8, 4).shard(devs, axis=0)
    c = t.const_like(5)
    self.assertEqual(c.shape, (8, 4))
    self.assertIsNone(c.device)
    out = t+c
    self.assertEqual(out.device, t.device)
    self.assertEqual(out.uop.axis, 0)

  def test_full_like_device_on_multi_raises(self):
    t = Tensor.ones(8, 4).shard(("NULL:0", "NULL:1"), axis=0)
    with self.assertRaises(RuntimeError): t.full_like(5, device="NULL")

class TestTensorShape(unittest.TestCase):
  def test_float_shape_raises(self):
    for dim in (2.0, 2.5):
      with self.subTest(dim=dim), self.assertRaisesRegex(RuntimeError, "shape must be int"): Tensor.ones(dim)

class TestTensorDevice(unittest.TestCase):
  def test_create_from_single_device_tuple(self):
    (Tensor([1.0], device=(Device.DEFAULT,)) + Tensor([2.0])).realize()

class TestTensorPad(unittest.TestCase):
  # padding int tensor with float-only value (like -inf) must promote dtype to fit value
  def test_pad_int_with_neg_inf(self):
    t = Tensor.arange(9).reshape(1, 1, 3, 3)
    self.assertEqual(t.dtype, dtypes.int)
    r = t.pad((1, 2, 0, -1), value=-float('inf'))
    self.assertEqual(r.dtype, dtypes.weakfloat)
    self.assertEqual(r.shape, (1, 1, 2, 6))

class TestTensorDeviceMismatch(unittest.TestCase):
  def test_gather(self):
    x = Tensor.empty(3, 4, device="NULL")
    idx = Tensor.zeros(3, 4, dtype=dtypes.int32, device="NULL:1")
    with self.assertRaises(RuntimeError): x.gather(0, idx)
  def test_scatter_index(self):
    x = Tensor.zeros(3, 4, device="NULL")
    idx = Tensor.zeros(3, 4, dtype=dtypes.int32, device="NULL:1")
    src = Tensor.ones(3, 4, device="NULL")
    with self.assertRaises(RuntimeError): x.scatter(0, idx, src)
  def test_scatter_src(self):
    x = Tensor.zeros(3, 4, device="NULL")
    idx = Tensor.zeros(3, 4, dtype=dtypes.int32, device="NULL")
    src = Tensor.ones(3, 4, device="NULL:1")
    with self.assertRaises(RuntimeError): x.scatter(0, idx, src)
  def test_getitem_tensor_index(self):
    x = Tensor.empty(4, 5, device="NULL")
    idx = Tensor([0, 1], dtype=dtypes.int32, device="NULL:1")
    with self.assertRaises(RuntimeError): x[idx]
  def test_sparse_categorical_crossentropy(self):
    x = Tensor.zeros(2, 3, device="NULL")
    Y = Tensor([0, 1], dtype=dtypes.int32, device="NULL:1")
    with self.assertRaises(RuntimeError): x.sparse_categorical_crossentropy(Y)

class TestTinygrad(unittest.TestCase):
  def test_zerodim_initialization(self):
    self.assertEqual(Tensor(55).shape, ())
    self.assertEqual(Tensor(3.14).shape, ())

  def test_deviceless_const_construct_device_repr(self):
    t = Tensor(UOp.const(2.0).cast(dtypes.float))
    self.assertIsNone(t.uop.device)
    self.assertIsNone(t.device)
    self.assertIn("<UOp None", repr(t))

  def test_rand_rejects_unknown_kwargs(self):
    with self.assertRaises(TypeError): Tensor.rand(5, generator="foo")

  def test_zeros_like_has_same_dtype_and_shape(self):
    for datatype in [dtypes.float16, dtypes.float32, dtypes.int8, dtypes.int32, dtypes.int64, dtypes.uint8]:
      a = Tensor([1, 2, 3], dtype=datatype)
      b = Tensor.zeros_like(a)
      assert a.dtype == b.dtype, f"dtype mismatch {a.dtype=} != {b.dtype}"
      assert a.shape == b.shape, f"shape mismatch {a.shape} != {b.shape}"

    a = Tensor([1, 2, 3])
    b = Tensor.zeros_like(a, dtype=dtypes.int8)
    assert a.dtype == dtypes.default_int and b.dtype == dtypes.int8, "a.dtype should be int and b.dtype should be char"
    assert a.shape == b.shape, f"shape mismatch {a.shape} != {b.shape}"

  def test_ones_like_has_same_dtype_and_shape(self):
    for datatype in [dtypes.float16, dtypes.float32, dtypes.int8, dtypes.int32, dtypes.int64, dtypes.uint8]:
      a = Tensor([1, 2, 3], dtype=datatype)
      b = Tensor.ones_like(a)
      assert a.dtype == b.dtype, f"dtype mismatch {a.dtype=} != {b.dtype}"
      assert a.shape == b.shape, f"shape mismatch {a.shape} != {b.shape}"

    a = Tensor([1, 2, 3])
    b = Tensor.ones_like(a, dtype=dtypes.int8)
    assert a.dtype == dtypes.default_int and b.dtype == dtypes.int8, "a.dtype should be int and b.dtype should be char"
    assert a.shape == b.shape, f"shape mismatch {a.shape} != {b.shape}"

  def test_rand_like_device(self):
    a = Tensor.ones(3, 3, device="NULL")
    b = Tensor.rand_like(a)
    self.assertEqual(b.device, a.device)

  def test_ndim(self):
    assert Tensor(1).ndim == 0
    assert Tensor.randn(1).ndim == 1
    assert Tensor.randn(2,2,2).ndim == 3
    assert Tensor.randn(1,1,1,1,1,1).ndim == 6

  def test_argfix(self):
    for f in [Tensor.zeros, Tensor.ones, Tensor.rand, Tensor.randn, Tensor.empty]:
      self.assertEqual(f().shape, ())
      self.assertEqual(f(1).shape, (1,))
      self.assertEqual(f(10,20,40).shape, (10,20,40))
      self.assertEqual(f([]).shape, ())
      self.assertEqual(f([1]).shape, (1,))
      self.assertEqual(f([10,20,40]).shape, (10,20,40))
      self.assertEqual(f(()).shape, ())
      self.assertEqual(f((1,)).shape, (1,))
      self.assertEqual(f((10,20,40)).shape, (10,20,40))

      with self.assertRaises(ValueError): f((2, 2), 2, 2)
      with self.assertRaises(ValueError): f((2, 2), (2, 2))
      with self.assertRaises(ValueError): f((128, 128), 0.0, 0.01)

  def test_numel(self):
    assert Tensor.randn(10, 10).numel() == 100
    assert Tensor.randn(1,2,5).numel() == 10
    assert Tensor.randn(1,1,1,1,1,1).numel() == 1
    assert Tensor([]).numel() == 0
    assert Tensor.randn(1,0,2,5).numel() == 0
    assert Tensor(3).numel() == 1

  def test_len(self):
    assert len(torch.zeros(7)) == len(Tensor.zeros(7))
    assert len(torch.zeros(10,20)) == len(Tensor.zeros(10,20))
    assert len(torch.zeros(10,20)) == len(Tensor.zeros(10,20,30))
    assert len(torch.zeros(1).flatten()) == len(Tensor.zeros(1).flatten())
    with self.assertRaises(TypeError): len(Tensor(3))

  def test_size(self):
    t1, t2 = torch.zeros(10,20), Tensor.zeros(10,20)
    assert t1.size() == t2.size()
    assert t1.size(0) == t2.size(0)
    assert t1.size(1) == t2.size(1)
    assert t1.size(-1) == t2.size(-1)
    assert t1.size(-2) == t2.size(-2)
    with self.assertRaises(IndexError): t2.size(2)

  def test_element_size(self):
    for _, dtype in DTYPES_DICT.items():
      assert dtype.itemsize == Tensor.randn(3, dtype=dtype).element_size(), f"Tensor.element_size() not matching Tensor.dtype.itemsize for {dtype}"

  def test_deepwalk_ctx_check(self):
    layer = Tensor.uniform(1, 1)
    x = Tensor.randn(1, 1, 1)
    x.dot(layer).mean().backward()
    x = Tensor.randn(1, 1, 1)
    x.dot(layer).mean().backward()

  def test_tensor_ndarray_dtype(self):
    arr = np.array([1]) # where dtype is implicitly int64
    assert Tensor(arr).dtype == dtypes.int64
    assert Tensor(arr, dtype=dtypes.float32).dtype == dtypes.float32 # check if ndarray correctly casts to Tensor dtype
    assert Tensor(arr, dtype=dtypes.float64).dtype == dtypes.float64 # check that it works for something else

  def test_tensor_list_shapes(self):
    self.assertEqual(Tensor([[[]]]).shape, (1,1,0))
    self.assertEqual(Tensor([[],[]]).shape, (2,0))
    self.assertEqual(Tensor([[[[]],[[]]], [[[]],[[]]], [[[]],[[]]]]).shape, (3,2,1,0))

  def test_tensor_list_errors(self):
    # inhomogeneous shape
    with self.assertRaises(ValueError): Tensor([[],[[]]])
    with self.assertRaises(ValueError): Tensor([[1],[]])
    with self.assertRaises(ValueError): Tensor([[1],[1],1])
    with self.assertRaises(ValueError): Tensor([[[1,1,1],[1,1]]])
    with self.assertRaises(ValueError): Tensor([[1,1,1],[[1,1,1]]])

  def test_tensor_dtype_errors(self):
    with self.assertRaises(AttributeError): Tensor([3], dtype="typo")
    with self.assertRaises(AttributeError): Tensor([3], dtype=(dtypes.int,))

  def test_no_bool(self):
    with self.assertRaises(TypeError):
      if Tensor(3):
        print("hi")

    with self.assertRaises(TypeError):
      _a = Tensor([3]) in [Tensor([3]), Tensor([4]), Tensor([5])]

  def test_repr_with_grad(self):
    a = Tensor([1.0])
    b = Tensor([1])
    c = (a + b).sum().backward()
    print(a)
    print(c)

class TestZeroShapeTensor(unittest.TestCase):
  def test_max_shape(self):
    from tinygrad import UOp
    t = Tensor.empty(2, UOp.variable('v', 1, 32), 4)
    self.assertEqual(t.max_shape, (2, 32, 4))
    self.assertEqual(t.max_numel(), 2*32*4)
    self.assertEqual(Tensor.empty(2, 3).max_shape, (2, 3))

if __name__ == '__main__':
  unittest.main()
