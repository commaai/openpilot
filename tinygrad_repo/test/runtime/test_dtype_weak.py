import tempfile, unittest, math

from tinygrad import Tensor, dtypes, TinyJit, Device
from tinygrad.helpers import Context
from tinygrad.uop.ops import UOp, Ops
from tinygrad.engine.jit import JitError


class TestWeakPromotion(unittest.TestCase):
  def test_materialize_at_default_dtype(self):
    for weak, value, strong in ((dtypes.weakfloat, 0.5, dtypes.default_float),):
      t = Tensor.const(value, weak)
      self.assertEqual(t.dtype, weak)
      self.assertEqual(t.data().itemsize, strong.itemsize)
      self.assertEqual(t.numpy().dtype.itemsize, strong.itemsize)
      # materializing commits at the kind default; contiguous has no layout to fix so it stays weak
      self.assertEqual((c := t.clone(Device.DEFAULT)).dtype, strong)
      self.assertEqual(c.item(), value)
      self.assertEqual(t.contiguous().dtype, weak)

  def test_assign_into_weak_commits(self):
    t = Tensor.const(0.5)
    t.assign(Tensor(1.0, dtype=dtypes.default_float))
    self.assertEqual((t.dtype, t.item()), (dtypes.default_float, 1.0))

  def test_copysign_meets_operands(self):
    r = Tensor([2], dtype=dtypes.uint8).copysign(Tensor([1], dtype=dtypes.uint32))
    self.assertEqual((r.dtype, r.tolist()), (dtypes.uint32, [2]))

  def test_minimum_reflects_weak_operand(self):
    r = Tensor(1).minimum(Tensor([2], dtype=dtypes.uint8))
    self.assertEqual((r.dtype, r.tolist()), (dtypes.uint8, [1]))
    for dt in dtypes.uints:
      r = Tensor([dt.max], dtype=dt).minimum(1)
      self.assertEqual((r.dtype, r.tolist()), (dt, [1]))
      self.assertNotIn(Ops.CAST, [u.op for u in r._uop.toposort()])

  def test_promote_keeps_shape_args(self):
    # the shape arg is the same CONST as the value, only the value lifts
    self.assertEqual((Tensor(5).expand(5) + 1.5).tolist(), [6.5]*5)
    self.assertEqual((Tensor(2).reshape(1,1).expand(2,2).pad(((0,2),(0,0))) + 0.5).tolist(), [[2.5,2.5],[2.5,2.5],[0.5,0.5],[0.5,0.5]])
    x, _ = Tensor(5).reshape(1).pad((1,1))._broadcasted(0.5)
    self.assertEqual((x._uop.op, x._uop.base.dtype, x._uop.src[1].dtype), (Ops.PAD, dtypes.weakfloat, dtypes.weakint))

  def test_cast_weak_expression_value_uses_cast_floor(self):
    with Context(DEFAULT_FLOAT=dtypes.float16):
      denom = Tensor.ones(1, dtype=dtypes.int32).sum() * 70000 + 1e-5
      out = Tensor(1.0, dtype=dtypes.float32) / denom
      self.assertAlmostEqual(out.item(), 1 / (70000 + 1e-5), places=10)

  def test_stacked_weak_casts_convert_each_kind(self):
    # each weak cast is a kind conversion: weakint truncates before weakfloat re-lifts (neither is only a marker)
    x = Tensor([2.5, -3.7], dtype=dtypes.float32)
    stacked = x.cast(dtypes.weakint).cast(dtypes.weakfloat)
    self.assertIs(stacked.dtype, dtypes.weakfloat)
    self.assertEqual(stacked.tolist(), [2.0, -3.0])

  def test_weakint_cast_truncates_for_every_consumer(self):
    # a weakint cast of a float is a truncation whether a cast, a compare or an arithmetic op consumes it
    x = Tensor([2.5, -3.5], dtype=dtypes.float32)
    self.assertEqual(x.cast(dtypes.weakint).cast(dtypes.float32).tolist(), [2.0, -3.0])
    self.assertEqual((x.cast(dtypes.weakint) * x).tolist(), [5.0, 10.5])
    self.assertEqual(Tensor([0.5, -0.5], dtype=dtypes.float32).cast(dtypes.weakint).cast(dtypes.bool).tolist(), [False, False])

  def test_concrete_pair_promotes_weak(self):
    out = Tensor([-1], dtype=dtypes.int64) + Tensor([3], dtype=dtypes.uint64) + Tensor(0.5)
    self.assertEqual((out.dtype, out.tolist()), (dtypes.weakfloat, [2.5]))

  def test_integer_values(self):
    x = Tensor.full((1,), 1, dtype=dtypes.int64)
    self.assertEqual((x + 2**40).item(), 2**40 + 1)
    self.assertEqual((x << 3).item(), 8)
    self.assertTrue((x < 2**40).item())

  @unittest.skipUnless(dtypes.float64 in Device[Device.DEFAULT].renderer.supported_dtypes(), "requires float64 precision")
  def test_float64_precision(self):
    value = 1.0 + 2**-40
    x64 = Tensor.full((1,), 1.0, dtype=dtypes.float64)
    self.assertEqual((x64 + value).item(), 2.0 + 2**-40)
    x32 = Tensor.full((1,), 0.0, dtype=dtypes.float32)
    self.assertEqual((x32 + value).item(), 1.0)

class TestWeakBounds(unittest.TestCase):
  def test_bounds_survive_movement(self):
    moved = Tensor(5).reshape(1).expand(2).pad((1, 1)).detach().contiguous_backward()
    self.assertEqual((moved.uop.vmin, moved.uop.vmax, moved.uop.bufferize().vmax), (0, 5, 5))
    self.assertEqual(moved.numpy().dtype, Tensor(5).numpy().dtype)   # a moved weak int reads at the same dtype as the bare one

  def test_wide_src_keeps_its_width(self):
    # the node's result fits int32, its variable does not: the shift runs at long, only the result narrows
    v = UOp.variable("v", 0, 2**40).bind(2**35+7)
    for t in (Tensor(v) // 2**31, (Tensor(v) - 1) // 2**31, Tensor(v).reshape(1) // 2**31): self.assertEqual(t.item(), 16)

  def test_padded_weak_const_keeps_its_zeros(self):
    self.assertEqual(Tensor(1).expand(1).cat(Tensor(2).expand(2), Tensor(3).expand(3)).tolist(), [1, 2, 2, 3, 3, 3])
    self.assertEqual((Tensor(5).reshape(1).pad((1, 1)) == 5).tolist(), [False, True, False])
    self.assertEqual((Tensor(5).reshape(1,1).expand(1,2).pad(((0,2),(0,0))) + Tensor([[1],[2],[3]])).tolist(), [[6,6],[2,2],[3,3]])

class TestWeakStorageBoundary(unittest.TestCase):
  # weak has no storage: a weak assignment source casts when it defers to the destination, everything else raises
  def test_weak_source(self):
    w05 = Tensor.const(0.5).reshape(1)
    dst = Tensor.zeros(2, dtype=dtypes.int8).contiguous().realize()
    with self.assertRaises(RuntimeError): dst.assign(w05.expand(2))                   # weakfloat into int does not defer
    with self.assertRaises(RuntimeError): dst[0:1] = w05
    fdst = Tensor.zeros(2, dtype=dtypes.float32).contiguous().realize()
    fdst[0:1] = w05                                                                    # weakfloat defers to float
    self.assertEqual(fdst.tolist(), [0.5, 0.0])
    with tempfile.TemporaryDirectory() as td:                                          # the DISK path checks the same
      ddst = Tensor.empty(2, dtype=dtypes.int32, device=f"DISK:{td}/t")
      with self.assertRaises(RuntimeError): ddst.assign(w05.expand(2))

  def test_weak_commits_by_bounds(self):
    big = Tensor(2**40)
    edges = (big.clone(), big.sum(), big.reshape(1).max(), big.reshape(1).mean(), Tensor.stack(big, Tensor(1)).sum() - 1,
             Tensor([2**40]), big.full_like(2**40))
    for t in edges: self.assertEqual(t.item(), 2**40)
    self.assertEqual(Tensor([10, 20, 30])[[2**32+1]].tolist(), [0])  # a wide list index is out of range, not wrapped
    with Context(DEFAULT_INT=dtypes.int64): self.assertEqual(Tensor(2).clone().dtype, dtypes.int64)

  @unittest.skipUnless(dtypes.int64 in Device[Device.DEFAULT].renderer.supported_dtypes(), "long scalar arguments cannot be emulated")
  def test_bound_variable_commits_by_bounds(self):
    bound = Tensor(UOp.variable("b", 0, 2**40).bind(2**35+3)).clone(Device.DEFAULT)
    self.assertEqual((bound.dtype, bound.item()), (dtypes.int64, 2**35+3))

  def test_literal_beyond_any_int_raises(self):
    for make in (lambda: Tensor(2**64).item(), lambda: Tensor([2**64]), lambda: Tensor.full((2,), -2**63-1)):
      with self.assertRaises(OverflowError): make()

  def test_weak_sentinels_commit_first(self):
    # max_pool2d, scatter_reduce and cummax pad with the dtype's min/max, which a weak dtype does not have
    self.assertEqual(Tensor(-5).expand(1, 1, 2, 2).max_pool2d(2, padding=1).dtype, Tensor(-5).clone().dtype)
    self.assertEqual(Tensor(-5).expand(2).scatter_reduce(0, Tensor([0]), Tensor(-5).expand(1), "amax", include_self=False).tolist(), [-5, -5])
    self.assertEqual(Tensor(2**40).expand(3).cummax(0)[0].tolist(), [2**40]*3)

class TestWeakMaterializationEntries(unittest.TestCase):
  # everything that creates storage from a weak value raises
  def test_reads_commit_storage_raises(self):
    for weak, value, strong in ((dtypes.weakfloat, 0.5, dtypes.default_float),):
      def weak_val():
        return Tensor([True]).where(Tensor.const(value, weak), Tensor.const(value, weak))
      self.assertEqual(weak_val().dtype, weak)
      self.assertEqual(weak_val().to(Device.DEFAULT).dtype, weak)
      self.assertEqual(weak_val().data().format, strong.fmt)
      self.assertEqual(weak_val().numpy().dtype.itemsize, strong.itemsize)
      self.assertEqual(weak_val().tolist(), [value])
      typed = weak_val().cast(strong).realize().uop
      self.assertEqual((typed.dtype, typed.buffer.nbytes), (strong, strong.itemsize))
      self.assertEqual(weak_val().contiguous().dtype, weak)                 # no layout to fix, stays weak
      self.assertEqual(weak_val().realize().dtype, weak)                    # no width to store, stays weak
      self.assertEqual(weak_val().clone().dtype, strong)                    # storage commits at the default
      for entry in (lambda t: t.to(f"{Device.DEFAULT}:1").realize(), lambda t: t.as_param(0)):
        with self.assertRaises(RuntimeError): entry(weak_val())

  def test_weak_is_virtual(self):
    # NOTE: int64 lub uint64 is weakfloat, so this is device-ful weak from promotion, never from a cast to weak
    devful = Tensor([1], dtype=dtypes.int64) + Tensor([1], dtype=dtypes.uint64)
    for t in (Tensor.const(0.5), devful):
      self.assertTrue(t.uop.is_virtual)
      # realize is a no-op, so a weak input can never become the real buffer TinyJit needs
      with self.assertRaises(JitError): TinyJit(lambda x: (x+1).realize())(t)
    c = devful.alu(Ops.STAGE)
    self.assertIs(c.dtype, dtypes.weakfloat)

  def test_empty_reads_commit(self):
    for weak, strong in ((dtypes.weakfloat, dtypes.default_float),):
      empty = Tensor.const(0, weak).reshape(1).shrink(((0, 0),))
      self.assertEqual(empty.data().format, strong.fmt)
      self.assertEqual(empty.numpy().dtype.itemsize, strong.itemsize)
      self.assertEqual(empty.tolist(), [])


class TestSignedUint64Weakfloat(unittest.TestCase):
  # int64 and uint64 have no common integer supertype (JAX JEP), so the join defers to weakfloat instead of wrapping
  def test_no_wrap(self):
    r = Tensor([-1], dtype=dtypes.int64) + Tensor([1], dtype=dtypes.uint64)
    self.assertEqual((r.dtype, r.item()), (dtypes.weakfloat, 0.0))

  def test_weakfloat_lowers(self):
    i64, u64 = Tensor([-1], dtype=dtypes.int64), Tensor([3], dtype=dtypes.uint64)
    r = i64 + u64 + Tensor([2], dtype=dtypes.float16)  # a concrete consumer takes the join
    self.assertEqual((r.dtype, r.cast(dtypes.float32).item()), (dtypes.half, 4.0))
    self.assertEqual((i64 < u64).item(), True)  # comparison meets at float
    self.assertAlmostEqual((i64 + u64).sin().item(), math.sin(2), places=5)  # Unary lowers before transcendental


if __name__ == "__main__":
  unittest.main()
