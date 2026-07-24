import tempfile, unittest

from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp


class TestWeakPromotion(unittest.TestCase):
  def test_rand_requires_concrete(self):
    with self.assertRaises(ValueError): Tensor.rand(2, dtype=dtypes.weakfloat)
    with self.assertRaises(ValueError): Tensor.const(dtypes.weakfloat, 1.0).rand_like()
    with self.assertRaises(ValueError): Tensor.const(dtypes.weakfloat, 1.0).randn_like()

  def test_sum_stays_weak(self):
    for weak, value in ((dtypes.weakfloat, 1.0),):
      self.assertEqual(Tensor.const(weak, value).expand(3).sum().dtype, weak)
    self.assertEqual((Tensor.const(dtypes.weakfloat, 1.0).expand(3).sum() + Tensor([1], dtype=dtypes.float16)).dtype, dtypes.float16)

  def test_materialize_at_default_dtype(self):
    for weak, value, strong in ((dtypes.weakfloat, 0.5, dtypes.default_float),):
      t = Tensor.const(weak, value)
      self.assertEqual(t.dtype, weak)
      self.assertEqual(t.data().itemsize, strong.itemsize)
      self.assertEqual(t.numpy().dtype.itemsize, strong.itemsize)
      with self.assertRaises(RuntimeError): t.clone("CPU")

  def test_uop_scalar_const_unchanged(self):
    for dtype, value in ((dtypes.weakint, 1), (dtypes.int32, 1), (dtypes.float32, 0.5)):
      out = UOp.variable("x", 0.0 if dtype == dtypes.float32 else 0, 10.0 if dtype == dtypes.float32 else 10, dtype) + value
      self.assertEqual((out.dtype, out.src[1].dtype), (dtype, dtype))

  @unittest.expectedFailure  # TODO: a weak const defers to its consumer (JAX): these dtypes change once python scalars are weak consts
  def test_changed_rows(self):
    t_i8, t_f16, t_bf16 = Tensor([1], dtype=dtypes.int8), Tensor([1], dtype=dtypes.float16), Tensor([1], dtype=dtypes.bfloat16)
    t_bool, t_u16 = Tensor([True]), Tensor([1], dtype=dtypes.uint16)
    self.assertEqual((t_i8 + 0.5).dtype, dtypes.weakfloat)
    self.assertEqual(((t_i8 + 0.5) + t_f16).dtype, dtypes.float16)
    self.assertEqual(((t_i8 + 0.5) + t_bf16).dtype, dtypes.bfloat16)
    self.assertEqual(((t_bool + 1) + t_i8).dtype, dtypes.int8)
    self.assertEqual(((t_bool + 1) + t_u16).dtype, dtypes.uint16)
    self.assertEqual((Tensor(3) + t_i8).dtype, dtypes.int8)
    self.assertEqual(Tensor([2], dtype=dtypes.uint8).pad(((1, 1),), value=1).dtype, dtypes.uint8)
    # zeros/ones are full with a python fill value, so they are weak too (jnp.zeros pins float32; deliberate divergence)
    self.assertEqual((Tensor.zeros(3) + t_f16).dtype, dtypes.float16)

  def test_unchanged_rows(self):
    t_i8, t_f16, t_f32 = Tensor([1], dtype=dtypes.int8), Tensor([1], dtype=dtypes.float16), Tensor([1], dtype=dtypes.float32)
    self.assertEqual((t_i8 + 1).dtype, dtypes.int8)
    self.assertEqual((t_f16 + 0.5).dtype, dtypes.float16)
    self.assertEqual((t_f32 + t_f16).dtype, dtypes.float32)

  @unittest.expectedFailure  # TODO: dot of a weak const tensor defers to the other operand once python scalars are weak consts
  def test_dot_defers_weak(self):
    weak = Tensor([True, False]).where(Tensor(1), 2)
    self.assertEqual(weak.dot(Tensor([1, 1], dtype=dtypes.int8)).dtype, dtypes.int8)

  def test_integer_values(self):
    x = Tensor.full((1,), 1, dtype=dtypes.int64, device="CPU")
    self.assertEqual((x + 2**40).item(), 2**40 + 1)
    self.assertEqual((x << 3).item(), 8)
    self.assertTrue((x < 2**40).item())

  def test_float64_precision(self):
    value = 1.0 + 2**-40
    x64 = Tensor.full((1,), 1.0, dtype=dtypes.float64, device="CPU")
    self.assertEqual((x64 + value).item(), 2.0 + 2**-40)
    x32 = Tensor.full((1,), 0.0, dtype=dtypes.float32, device="CPU")
    self.assertEqual((x32 + value).item(), 1.0)

  @unittest.expectedFailure  # TODO: exp/cos/sigmoid of a weak const stay weak instead of casting to a concrete float
  def test_weak_transcendentals(self):
    t_f16 = Tensor([1], dtype=dtypes.float16)
    for out in (Tensor(2).exp(), Tensor(2).cos(), Tensor(2).sigmoid()):
      self.assertEqual((out.dtype, (out + t_f16).dtype), (dtypes.weakfloat, dtypes.float16))

  def test_null_lowering(self):
    for t in (Tensor.full((1,), 1, dtype=dtypes.int64, device="NULL") + 2**40,
              Tensor.full((1,), 1.0, dtype=dtypes.float64, device="NULL") + (1.0 + 2**-40)):
      t.realize()
      self.assertNotIn(t.uop.buffer.dtype, dtypes.weaks)


class TestWeakStorageBoundary(unittest.TestCase):
  # weak has no storage: a weak assignment source casts when it defers to the destination, everything else raises
  def test_weak_source(self):
    w05 = Tensor.const(dtypes.weakfloat, 0.5).reshape(1)
    dst = Tensor.zeros(2, dtype=dtypes.int8, device="CPU").contiguous().realize()
    with self.assertRaises(RuntimeError): dst.assign(w05.expand(2))                   # weakfloat into int does not defer
    with self.assertRaises(RuntimeError): dst[0:1] = w05
    fdst = Tensor.zeros(2, dtype=dtypes.float32, device="CPU").contiguous().realize()
    fdst[0:1] = w05                                                                    # weakfloat defers to float
    self.assertEqual(fdst.tolist(), [0.5, 0.0])
    with tempfile.TemporaryDirectory() as td:                                          # the DISK path checks the same
      ddst = Tensor.empty(2, dtype=dtypes.int32, device=f"DISK:{td}/t")
      with self.assertRaises(RuntimeError): ddst.assign(w05.expand(2))

  def test_weak_has_no_storage(self):
    import numpy as np
    with self.assertRaises(RuntimeError): Tensor(np.ones(2, dtype=np.float32), dtype=dtypes.weakfloat)
    with self.assertRaises(RuntimeError): Tensor(bytes(8), dtype=dtypes.weakfloat)

class TestWeakMaterializationEntries(unittest.TestCase):
  # everything that creates storage from a weak value raises
  def test_reads_commit_storage_raises(self):
    for weak, value, strong in ((dtypes.weakfloat, 0.5, dtypes.default_float),):
      def weak_val():
        return Tensor([True], device="CPU").where(Tensor.const(weak, value), Tensor.const(weak, value))
      self.assertEqual(weak_val().dtype, weak)
      self.assertEqual(weak_val().to("CPU").dtype, weak)
      self.assertEqual(weak_val().data().format, strong.fmt)
      self.assertEqual(weak_val().numpy().dtype.itemsize, strong.itemsize)
      self.assertEqual(weak_val().tolist(), [value])
      self.assertEqual(weak_val().cast(strong).realize().uop.buffer.dtype, strong)
      for entry in (lambda t: t.contiguous(), lambda t: t.realize(), lambda t: t.clone(),
                    lambda t: t.to("CPU:1").realize(), lambda t: t.as_param(0)):
        with self.assertRaises(RuntimeError): entry(weak_val())

  def test_empty_reads_commit(self):
    for weak, strong in ((dtypes.weakfloat, dtypes.default_float),):
      empty = Tensor.const(weak, 0).reshape(1).shrink(((0, 0),))
      self.assertEqual(empty.data().format, strong.fmt)
      self.assertEqual(empty.numpy().dtype.itemsize, strong.itemsize)
      self.assertEqual(empty.tolist(), [])


if __name__ == "__main__":
  unittest.main()
