import tempfile, unittest, math

from tinygrad import Tensor, dtypes, TinyJit
from tinygrad.helpers import Context
from tinygrad.dtype import least_upper_float
from tinygrad.uop.ops import UOp, Ops, dtype_from_uop, graph_rewrite
from tinygrad.uop.weak import pm_lower_index_dtype, pm_commit_weak
from tinygrad.uop.symbolic import symbolic_simple
from tinygrad.uop.spec import spec_shared, type_verify
from tinygrad.engine.jit import JitError


class TestWeakPromotion(unittest.TestCase):
  def test_rand_requires_concrete(self):
    with self.assertRaises(ValueError): Tensor.rand(2, dtype=dtypes.weakfloat)
    with self.assertRaises(ValueError): Tensor.const(1.0).rand_like()
    with self.assertRaises(ValueError): Tensor.const(1.0).randn_like()

  def test_reduce_strips_weakness(self):
    for weak, value, strong in ((dtypes.weakint, 1, dtypes.default_int), (dtypes.weakfloat, 1.0, dtypes.default_float)):
      t = Tensor.const(value, weak).expand(3)
      for out in (t.sum(), t.max(), t.prod(), t.cumsum(0), t.cummax(0)[0]): self.assertEqual(out.dtype, strong)
    self.assertEqual((Tensor.const(1.0).expand(3).sum() + Tensor([1], dtype=dtypes.float16)).dtype, dtypes.float32)

  def test_materialize_at_default_dtype(self):
    for weak, value, strong in ((dtypes.weakfloat, 0.5, dtypes.default_float),):
      t = Tensor.const(value, weak)
      self.assertEqual(t.dtype, weak)
      self.assertEqual(t.data().itemsize, strong.itemsize)
      self.assertEqual(t.numpy().dtype.itemsize, strong.itemsize)
      # materializing commits at the kind default; contiguous has no layout to fix so it stays weak
      self.assertEqual((c := t.clone("CPU")).dtype, strong)
      self.assertEqual(c.item(), value)
      self.assertEqual(t.contiguous().dtype, weak)

  def test_assign_into_weak_commits(self):
    t = Tensor.const(0.5)
    t.assign(Tensor(1.0, dtype=dtypes.default_float))
    self.assertEqual((t.dtype, t.item()), (dtypes.default_float, 1.0))

  def test_float_unary_on_weakint_stays_weak(self):
    self.assertIs(least_upper_float(dtypes.weakint), dtypes.weakfloat)

  def test_copysign_meets_operands(self):
    r = Tensor([2], dtype=dtypes.uint8, device="CPU").copysign(Tensor([1], dtype=dtypes.uint32, device="CPU"))
    self.assertEqual((r.dtype, r.tolist()), (dtypes.uint32, [2]))

  def test_minimum_reflects_weak_operand(self):
    r = Tensor(1).minimum(Tensor([2], dtype=dtypes.uint8, device="CPU"))
    self.assertEqual((r.dtype, r.tolist()), (dtypes.uint8, [1]))
    for dt in dtypes.uints:
      r = Tensor([dt.max], dtype=dt, device="CPU").minimum(1)
      self.assertEqual((r.dtype, r.tolist()), (dt, [1]))
      self.assertNotIn(Ops.CAST, [u.op for u in r._uop.toposort()])

  def test_broadcasted_keeps_const_weak(self):
    # a python scalar stays a bare weak CONST through _broadcasted, lifted only to the KIND of the lub
    x, y = Tensor([1], dtype=dtypes.int8)._broadcasted(3)
    self.assertEqual((y._uop.base.op, y.dtype, x.dtype), (Ops.CONST, dtypes.weakint, dtypes.int8))
    x, y = Tensor([1], dtype=dtypes.int8)._broadcasted(0.5)
    self.assertEqual((y._uop.base.op, y.dtype, x.dtype), (Ops.CONST, dtypes.weakfloat, dtypes.weakfloat))
    x, y = Tensor.const(1).reshape(1)._broadcasted(Tensor([1.0], dtype=dtypes.float32))
    self.assertEqual((x._uop.base.op, x._uop.base.val, x.dtype, x.shape, y.dtype),
                     (Ops.CONST, 1, dtypes.weakfloat, (1,), dtypes.float32))

  def test_weak_expression_anchors_at_strong_lub(self):
    # regression test for the HALF bert nan (#17408, reverted in #17409): lub(int32, weakfloat)==weakfloat makes
    # `loss_mask.sum() + 1e-5` a weakfloat EXPRESSION. Meeting a strong float in a binop must pin it at the lub
    denom = (Tensor.zeros(912, dtype=dtypes.int32) != Tensor.zeros(912, dtype=dtypes.float32)).sum() + 1e-5
    self.assertIs(denom.dtype, dtypes.weakfloat)  # the setup: the denominator expression itself is weak
    x, y = Tensor([2048.0], dtype=dtypes.float32)._broadcasted(denom)
    self.assertIs(y.dtype, dtypes.float32)
    recips = [u for u in (x / y)._uop.toposort() if u.op is Ops.RECIPROCAL]
    self.assertEqual([(u.dtype, u.src[0].dtype) for u in recips], [(dtypes.float32, dtypes.float32)])
    with Context(DEFAULT_FLOAT=dtypes.float16):
      committed = graph_rewrite((UOp.const(1).cast(dtypes.int32) + UOp.const(1.0)).cast(dtypes.float32), pm_lower_index_dtype, ctx={})
    self.assertEqual([u.dtype for u in committed.toposort() if u.op is Ops.ADD], [dtypes.float32])

  def test_cast_weak_expression_commits_at_cast_floor(self):
    # the floor never narrows: a cast BELOW the default does not pull the compute width down with it
    with Context(DEFAULT_FLOAT=dtypes.float32):
      narrowed = graph_rewrite((UOp.const(1.0) + UOp.const(2.0)).cast(dtypes.float16), pm_lower_index_dtype, ctx={})
    self.assertEqual((narrowed.dtype, narrowed.src[0].dtype), (dtypes.float16, dtypes.float32))

  def test_cast_weak_expression_value_uses_cast_floor(self):
    with Context(DEFAULT_FLOAT=dtypes.float16):
      denom = Tensor.ones(1, dtype=dtypes.int32, device="CPU").sum() * 70000 + 1e-5
      out = Tensor(1.0, dtype=dtypes.float32, device="CPU") / denom
      self.assertAlmostEqual(out.item(), 1 / (70000 + 1e-5), places=10)

  def test_uop_scalar_const_lifts_kind(self):
    for dtype, value, out_dtype, const_dtype in ((dtypes.weakint, 1, dtypes.weakint, dtypes.weakint),
                                                 (dtypes.int32, 1, dtypes.int32, dtypes.weakint),
                                                 (dtypes.int32, 0.5, dtypes.weakfloat, dtypes.weakfloat),
                                                 (dtypes.float32, 1, dtypes.float32, dtypes.weakfloat)):
      out = UOp.variable("x", 0.0 if dtype == dtypes.float32 else 0, 10.0 if dtype == dtypes.float32 else 10, dtype) + value
      self.assertEqual((out.dtype, out.src[1].op, out.src[1].dtype), (out_dtype, Ops.CONST, const_dtype))
    # the kind lift converts the VALUE too (the arg is the only dtype carrier once UOp.const loses its dtype arg),
    # and a bare weak const UOp is the same spelling as the python scalar: both lift to the same node
    x = UOp.variable("x", 0.0, 1.0, dtypes.float32)
    self.assertIsInstance((x + 2).src[1].val, float)
    self.assertIs(x + UOp.const(2), x + 2)

  def test_index_dtype_ignores_weakness(self):
    with Context(SPEC=2):
      idx = UOp.const(0).cast(dtypes.int32)
      weak = UOp.const(1.0).expand((1,))
      self.assertEqual(UOp(Ops.INDEX, dtypes.float32, (weak, idx)).dtype, dtypes.float32)
      with self.assertRaisesRegex(RuntimeError, "bad dtype"): UOp(Ops.INDEX, dtypes.int32, (weak, idx))

  def test_store_weak_value_uses_destination_dtype(self):
    with Context(DEFAULT_FLOAT=dtypes.float16):
      dst = UOp.param(0, dtypes.bfloat16, (1,)).index(UOp.const(0).cast(dtypes.int32))
      gate = UOp.const(True)
      out = graph_rewrite(dst.store(UOp.const(5.0), gate), pm_lower_index_dtype, ctx={})
    # a bare weak CONST commits directly: the pass runs without symbolic, so a CAST here would survive it
    self.assertEqual((out.src[1], out.src[2]), (UOp.const(5.0, dtypes.bfloat16), gate))

  def test_weak_srcs_commit_only_at_a_concrete_lub(self):
    weak_lub = UOp(Ops.ADD, src=(UOp.const(1), UOp.const(1.0)))
    self.assertIs(graph_rewrite(weak_lub, pm_lower_index_dtype, ctx={}), weak_lub)
    concrete = UOp.const(2.0).cast(dtypes.float16)
    where = graph_rewrite(UOp(Ops.WHERE, src=(UOp.const(True), concrete, UOp.const(1.0))), pm_lower_index_dtype, ctx={})
    self.assertEqual(tuple(x.dtype for x in where.src), (dtypes.bool, dtypes.float16, dtypes.float16))

  def test_weak_shift_lhs_commits_the_node(self):
    # a shift derives its lhs's dtype, so committing the lhs restates the root (WGSL's packed store writes `mask << shift_am`)
    shl = graph_rewrite(UOp.const(0xFFFF) << UOp.variable("x", 0, 16, dtypes.uint), symbolic_simple+pm_commit_weak)
    self.assertEqual((shl.dtype, shl.src[0]), (dtypes.uint, UOp.const(0xFFFF, dtypes.uint)))

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
    # zeros/ones are full with a python fill value, so they are weak too (jnp.zeros pins float32; deliberate divergence)
    self.assertEqual((Tensor.zeros(3) + t_f16).dtype, dtypes.float16)

  def test_unchanged_rows(self):
    t_i8, t_f16, t_f32 = Tensor([1], dtype=dtypes.int8), Tensor([1], dtype=dtypes.float16), Tensor([1], dtype=dtypes.float32)
    self.assertEqual((t_i8 + 1).dtype, dtypes.int8)
    self.assertEqual((t_f16 + 0.5).dtype, dtypes.float16)
    self.assertEqual((t_f32 + t_f16).dtype, dtypes.float32)
    self.assertEqual(Tensor([2], dtype=dtypes.uint8).pad(((1, 1),), value=1).dtype, dtypes.uint8)

  def test_concrete_pair_promotes_weak(self):
    out = Tensor([-1], dtype=dtypes.int64, device="CPU") + Tensor([3], dtype=dtypes.uint64, device="CPU") + Tensor(0.5)
    self.assertEqual((out.dtype, out.tolist()), (dtypes.weakfloat, [2.5]))

  def test_dot_defers_weak(self):
    weak = Tensor([True, False]).where(Tensor(1), 2)
    self.assertEqual(weak.dot(Tensor([1, 1], dtype=dtypes.int8)).dtype, dtypes.int8)

  def test_weak_int_binop(self):
    v = UOp.variable("i", 0, 10, dtypes.weakint)
    self.assertEqual((v << 1).dtype, dtypes.weakint)
    self.assertEqual(dtype_from_uop(Ops.SHL, (UOp.const(1, dtypes.int8), UOp.const(1, dtypes.uint32)), None), dtypes.int8)
    self.assertEqual(UOp.const(1).alu(Ops.SHL, UOp.const(1, dtypes.uint)).dtype, dtypes.weakint)
    self.assertEqual((v & 3).dtype, dtypes.weakint)
    with self.assertRaises(RuntimeError): Tensor.const(1.0) << Tensor.const(1.0)
    with self.assertRaises(RuntimeError): UOp.const(1, dtypes.int32).alu(Ops.SHL, UOp.const(1, dtypes.float64))
    for op in (Ops.SHL, Ops.SHR):
      with self.assertRaises(RuntimeError):
        UOp.const(1, dtypes.float32).alu(op, UOp.const(1, dtypes.int32))
    # float bitwise builds, the spec rejects it
    with Context(SPEC=1):
      f32, wf = UOp.const(1.0, dtypes.float32), UOp.const(1.0)
      for bad in (f32.alu(Ops.AND, f32), UOp(Ops.AND, dtypes.float32, (f32, f32)), UOp(Ops.AND, dtypes.int32, (wf, wf))):
        with self.assertRaises(RuntimeError): type_verify([bad], spec_shared)

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

  def test_weak_transcendentals(self):
    t_f16 = Tensor([1], dtype=dtypes.float16)
    for out in (Tensor(2).exp(), Tensor(2).cos(), Tensor(2).sigmoid()):
      self.assertEqual((out.dtype, (out + t_f16).dtype), (dtypes.weakfloat, dtypes.float16))

  def test_null_lowering(self):
    for t in (Tensor.full((1,), 1, dtype=dtypes.int64, device="NULL") + 2**40,
              Tensor.full((1,), 1.0, dtype=dtypes.float64, device="NULL") + (1.0 + 2**-40)):
      t.realize()
      self.assertNotIn(t.uop.buffer.dtype, dtypes.weaks)

  def test_computed_float_index_lowers(self):
    # a half-pixel nearest index resolves its float-scaled range before the gather
    idx = (Tensor.arange(8) + 0.5) / 4 - 0.5
    idx = (idx.clip(0, 1) - 0.5).ceil().int()
    out = Tensor([0, 1], device="NULL")[idx].contiguous().realize()
    self.assertNotIn(out.uop.buffer.dtype, dtypes.weaks)


class TestWeakStorageBoundary(unittest.TestCase):
  # weak has no storage: a weak assignment source casts when it defers to the destination, everything else raises
  def test_weak_source(self):
    w05 = Tensor.const(0.5).reshape(1)
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
        return Tensor([True], device="CPU").where(Tensor.const(value, weak), Tensor.const(value, weak))
      self.assertEqual(weak_val().dtype, weak)
      self.assertEqual(weak_val().to("CPU").dtype, weak)
      self.assertEqual(weak_val().data().format, strong.fmt)
      self.assertEqual(weak_val().numpy().dtype.itemsize, strong.itemsize)
      self.assertEqual(weak_val().tolist(), [value])
      self.assertEqual(weak_val().cast(strong).realize().uop.buffer.dtype, strong)
      self.assertEqual(weak_val().contiguous().dtype, weak)                 # no layout to fix, stays weak
      self.assertEqual(weak_val().realize().dtype, weak)                    # no width to store, stays weak
      self.assertEqual(weak_val().clone().dtype, strong)                    # storage commits at the default
      for entry in (lambda t: t.to("CPU:1").realize(), lambda t: t.as_param(0)):
        with self.assertRaises(RuntimeError): entry(weak_val())

  def test_weak_is_virtual(self):
    # NOTE: int64 lub uint64 is weakfloat, so this is device-ful weak from promotion, never from a cast to weak
    devful = Tensor([1], dtype=dtypes.int64, device="CPU") + Tensor([1], dtype=dtypes.uint64, device="CPU")
    for t in (Tensor.const(0.5), devful):
      self.assertTrue(t.uop.is_virtual)
      # realize is a no-op, so a weak input can never become the real buffer TinyJit needs
      with self.assertRaises(JitError): TinyJit(lambda x: (x+1).realize())(t)
    # callify must not silently commit a weak CONTIGUOUS to storage
    c = devful.alu(Ops.CONTIGUOUS)
    c.callify()
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
    r = Tensor([-1], dtype=dtypes.int64, device="CPU") + Tensor([1], dtype=dtypes.uint64, device="CPU")
    self.assertEqual((r.dtype, r.item()), (dtypes.weakfloat, 0.0))

  def test_weakfloat_lowers(self):
    i64, u64 = Tensor([-1], dtype=dtypes.int64, device="CPU"), Tensor([3], dtype=dtypes.uint64, device="CPU")
    r = i64 + u64 + Tensor([2], dtype=dtypes.float16, device="CPU")  # a concrete consumer takes the join
    self.assertEqual((r.dtype, r.cast(dtypes.float32).item()), (dtypes.half, 4.0))
    self.assertEqual((i64 < u64).item(), True)  # comparison meets at float
    self.assertAlmostEqual((i64 + u64).sin().item(), math.sin(2), places=5)  # Unary lowers before transcendental


if __name__ == "__main__":
  unittest.main()
