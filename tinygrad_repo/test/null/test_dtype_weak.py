import unittest
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context
from tinygrad.dtype import least_upper_float
from tinygrad.uop.ops import UOp, Ops, GroupOp, dtype_from_uop, graph_rewrite
from tinygrad.uop.weak import pm_commit_weak
from tinygrad.uop.symbolic import symbolic_simple
from tinygrad.uop.spec import spec_shared, type_verify
from test.helpers import full_rewrite

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

  def test_float_unary_on_weakint_stays_weak(self):
    self.assertIs(least_upper_float(dtypes.weakint), dtypes.weakfloat)

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
      committed = graph_rewrite((UOp.const(1).cast(dtypes.int32) + UOp.const(1.0)).cast(dtypes.float32), pm_commit_weak)
    self.assertEqual([u.dtype for u in committed.toposort() if u.op is Ops.ADD], [dtypes.float32])

  def test_div_sub_operand_kept_weak(self):
    a = Tensor.empty(4, dtype=dtypes.float32)
    for t in (a / 1, a - 0):
      self.assertEqual(t.uop.src[1].dtype, dtypes.weakfloat)

  def test_cast_weak_expression_commits_at_cast_floor(self):
    # the floor never narrows: a cast BELOW the default does not pull the compute width down with it
    with Context(DEFAULT_FLOAT=dtypes.float32):
      narrowed = graph_rewrite((UOp.const(1.0) + UOp.const(2.0)).cast(dtypes.float16), pm_commit_weak)
    self.assertEqual((narrowed.dtype, narrowed.src[0].dtype), (dtypes.float16, dtypes.float32))

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

  def test_store_weak_value_uses_destination_dtype(self):
    with Context(DEFAULT_FLOAT=dtypes.float16):
      dst = UOp.param(0, dtypes.bfloat16, 1).index(UOp.const(0).cast(dtypes.int32))
      gate = UOp.const(True)
      out = graph_rewrite(dst.store(UOp.const(5.0), gate), pm_commit_weak)
    # a bare weak CONST commits directly: the pass runs without symbolic, so a CAST here would survive it
    self.assertEqual((out.src[1], out.src[2]), (UOp.const(5.0, dtypes.bfloat16), gate))

  def test_weak_srcs_commit_only_at_a_concrete_lub(self):
    weak_lub = UOp(Ops.ADD, src=(UOp.const(1), UOp.const(1.0)))
    self.assertIs(graph_rewrite(weak_lub, pm_commit_weak), weak_lub)
    concrete = UOp.const(2.0).cast(dtypes.float16)
    # the weak arm stays bare: its sibling states the width, so the WHERE already derives float16 for it
    where = graph_rewrite(UOp(Ops.WHERE, src=(UOp.const(True), concrete, UOp.const(1.0))), pm_commit_weak)
    self.assertEqual((where.dtype, tuple(x.dtype for x in where.src)), (dtypes.float16, (dtypes.bool, dtypes.float16, dtypes.weakfloat)))

  def test_derivable_const_rounds_at_the_derived_width(self):
    # re-rounds a derivable const in place (still bare) so value-keyed folds (x*1 -> x, x*-1 -> NEG) still fire
    x = UOp.param(0, dtypes.float32, 1).index(UOp.const(0).cast(dtypes.int32)).load()
    mul = graph_rewrite(x * UOp.const(-0.9999999893980771), symbolic_simple+pm_commit_weak)
    self.assertIs(mul.src[1], UOp.const(-1.0))
    self.assertIs(graph_rewrite(x * UOp.const(1.0000000106), symbolic_simple+pm_commit_weak), x)

  def test_committed_const_conversion_folds(self):
    folded = graph_rewrite(UOp.const(16256, dtypes.ushort).cast(dtypes.uint), symbolic_simple)
    self.assertIs(folded, UOp.const(16256, dtypes.uint))
    # an emulated dtype const is a value too: one committed const, the renderer emits it directly
    emulated = UOp.const(1.0, dtypes.float).cast(dtypes.bfloat16)
    self.assertIs(graph_rewrite(emulated, symbolic_simple), UOp.const(1.0, dtypes.bfloat16))

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

  def test_dot_defers_weak(self):
    weak = Tensor([True, False]).where(Tensor(1), 2)
    self.assertEqual(weak.dot(Tensor([1, 1], dtype=dtypes.int8)).dtype, dtypes.int8)

  def test_weak_int_binop(self):
    v = UOp.variable("i", 0, 10, dtypes.weakint)
    self.assertEqual((v << 1).dtype, dtypes.weakint)
    self.assertEqual(dtype_from_uop(Ops.SHL, (UOp.const(1, dtypes.int8), UOp.const(1, dtypes.uint32)), None), dtypes.int8)
    self.assertEqual(UOp.const(1).alu(Ops.SHL, UOp.const(1, dtypes.uint)).dtype, dtypes.weakint)
    self.assertEqual((v & 3).dtype, dtypes.weakint)
    with self.assertRaises(RuntimeError): (Tensor.const(1.0) << Tensor.const(1.0)).dtype
    with self.assertRaises(RuntimeError): UOp.const(1, dtypes.int32).alu(Ops.SHL, UOp.const(1, dtypes.float64)).dtype
    for op in (Ops.SHL, Ops.SHR):
      with self.assertRaises(RuntimeError):
        UOp.const(1, dtypes.float32).alu(op, UOp.const(1, dtypes.int32)).dtype
    # float bitwise builds, the spec rejects it
    with Context(SPEC=1):
      f32, wf = UOp.const(1.0, dtypes.float32), UOp.const(1.0)
      for bad in (f32.alu(Ops.AND, f32), UOp(Ops.AND, (f32, f32)), UOp(Ops.AND, (wf, wf))):
        with self.assertRaises(RuntimeError): type_verify([bad], spec_shared)

  def test_weak_transcendentals(self):
    t_f16 = Tensor([1], dtype=dtypes.float16)
    for out in (Tensor(2).exp(), Tensor(2).cos(), Tensor(2).sigmoid()):
      self.assertEqual((out.dtype, (out + t_f16).dtype), (dtypes.weakfloat, dtypes.float16))

  def test_null_lowering(self):
    for t in (Tensor.full((1,), 1, dtype=dtypes.int64, device="NULL") + 2**40,
              Tensor.full((1,), 1.0, dtype=dtypes.float64, device="NULL") + (1.0 + 2**-40)):
      t.realize()
      self.assertNotIn(t.uop.dtype, dtypes.weaks)
      self.assertEqual(t.uop.buffer.nbytes, t.numel() * t.dtype.itemsize)

  def test_computed_float_index_lowers(self):
    # a half-pixel nearest index resolves its float-scaled range before the gather
    idx = (Tensor.arange(8) + 0.5) / 4 - 0.5
    idx = (idx.clip(0, 1) - 0.5).ceil().int()
    out = Tensor([0, 1], device="NULL")[idx].contiguous().realize()
    self.assertNotIn(out.uop.dtype, dtypes.weaks)
    self.assertEqual(out.uop.buffer.nbytes, out.numel() * out.dtype.itemsize)

class TestWeakStorageBoundary(unittest.TestCase):
  # weak has no storage: a weak assignment source casts when it defers to the destination, everything else raises
  def test_weak_has_no_storage(self):
    import numpy as np
    with self.assertRaises(RuntimeError): Tensor(np.ones(2, dtype=np.float32), dtype=dtypes.weakfloat)
    with self.assertRaises(RuntimeError): Tensor(bytes(8), dtype=dtypes.weakfloat)

class TestNoRedundantWide(unittest.TestCase):
  def wide_alu(self, t:Tensor) -> int:
    return sum(sum(1 for u in full_rewrite(call.src[0]).toposort() if u.op in GroupOp.ALU and u.dtype in {dtypes.long, dtypes.ulong})
               for call in t.schedule_linear().src if call.src[0].op is Ops.SINK)

  def test_unbounded_long_stays_long(self):
    self.assertGreater(self.wide_alu(Tensor.empty(16, dtype=dtypes.long)*3 + 1), 0)

  def test_fancy_index_has_no_wide_alu(self):
    j, o = Tensor([0, 1, 2]).reshape(3, 1), Tensor([0, 1]).reshape(1, 2)
    self.assertEqual(self.wide_alu(Tensor.empty(8, 9, 10, 11, 12)[1, j, 2, o, 2]), 0)

if __name__ == '__main__':
  unittest.main()
