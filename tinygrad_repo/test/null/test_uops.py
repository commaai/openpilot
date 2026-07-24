# uops tests that pass on NULL backend (no copyout needed)
import unittest
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.helpers import Timing, Context, cdiv
from tinygrad.dtype import dtypes, ConstFloat, Invalid  # noqa: F401
from tinygrad.device import Device
from tinygrad.uop.ops import Ops, ParamArg, UOp, UPat, dtype_from_uop, exec_alu  # noqa: F401  # ParamArg used by eval(str(uop)) roundtrip tests
from tinygrad.uop.spec import spec_program, spec_shared, type_verify
from tinygrad.uop.symbolic import sym
from test.helpers import eval_uop, to_uops_list

class TestDTypeFromUOp(unittest.TestCase):
  def test_broadcastable_promotion(self):
    self.assertEqual(dtype_from_uop(Ops.ADD, (UOp.const(dtypes.float32, 1.0), UOp.const(dtypes.float16, 1.0)), None), dtypes.float32)
    self.assertEqual(dtype_from_uop(Ops.MUL, (UOp.const(dtypes.int8, 1), UOp.const(dtypes.int32, 1)), None), dtypes.int32)
    with self.assertRaises(KeyError): dtype_from_uop(Ops.ADD, (UOp.const(dtypes.weakint, 1), UOp.const(dtypes.int8, 1)), None)

  def test_same_dtype_fast_path(self):
    src = (UOp.const(dtypes.weakint, 1), UOp.const(dtypes.weakint, 2))
    self.assertEqual(dtype_from_uop(Ops.ADD, src, None), dtypes.weakint)

  def test_where_promotion(self):
    cond = UOp.const(dtypes.bool, True)
    self.assertEqual(dtype_from_uop(Ops.WHERE, (cond, UOp.const(dtypes.float32, 1.0), UOp.const(dtypes.float16, 1.0)), None), dtypes.float32)
    idx = UOp.range(4, 0)
    self.assertEqual(idx.valid(idx < 4).dtype, dtypes.weakint)

  def test_const_dtype_from_value(self):
    self.assertEqual(dtype_from_uop(Ops.CONST, (), True), dtypes.bool)
    self.assertEqual(dtype_from_uop(Ops.CONST, (), ConstFloat(3.0)), dtypes.weakfloat)
    self.assertEqual(dtype_from_uop(Ops.CONST, (), Invalid), dtypes.bool)
    self.assertRaises(TypeError, dtype_from_uop, Ops.CONST, (), (1, 2))

  @Context(SPEC=2)
  def test_const_default_dtype_is_derived(self):
    self.assertEqual(UOp(Ops.CONST, arg=ConstFloat(3.0)).dtype, dtypes.weakfloat)
    self.assertEqual(UOp(Ops.CONST, arg=True).dtype, dtypes.bool)
    self.assertEqual(UOp(Ops.CONST, arg=Invalid).dtype, dtypes.bool)
    # an explicit (strong) const dtype is legal until the field is removed
    self.assertEqual(UOp.const(dtypes.int32, 3).dtype, dtypes.int32)

  def test_weak_dtype_rejected_by_program_spec(self):
    for weak, concrete, value in ((dtypes.weakint, dtypes.int32, 1), (dtypes.weakfloat, dtypes.float32, 1.0)):
      with self.assertRaises(RuntimeError): type_verify(UOp.const(weak, value).sink(), spec_program)
      type_verify(UOp.const(concrete, value).sink(), spec_program)

class TestSafeCast(unittest.TestCase):
  def test_cast_folds(self):
    a = UOp.variable("a", 1, 10, dtype=dtypes.int32)
    self.assertEqual(a.cast(dtypes.int64).cast(dtypes.int32).simplify(), a)
    self.assertEqual(a.cast(dtypes.double).cast(dtypes.int32).simplify(), a)
    a = UOp.variable("a", 1, 10, dtype=dtypes.uint8)
    self.assertEqual(a.cast(dtypes.int64).cast(dtypes.uint8).simplify(), a)
    self.assertEqual(a.cast(dtypes.uint32).cast(dtypes.uint8).simplify(), a)

  def test_remove_intermediate_cast(self):
    a = UOp.variable("a", 0., 100., dtype=dtypes.half)
    self.assertEqual(a.cast(dtypes.double).cast(dtypes.float).simplify(), a.cast(dtypes.float))
    a = UOp.variable("a", 1, 10, dtype=dtypes.int32)
    # TODO: double preserves certain int dtypes
    self.assertEqual(a.cast(dtypes.double).cast(dtypes.float).simplify(), a.cast(dtypes.float))
    self.assertEqual(a.cast(dtypes.int64).cast(dtypes.int16).simplify(), a.cast(dtypes.int16))
    a = UOp.variable("a", 1, 10, dtype=dtypes.uint8)
    self.assertEqual(a.cast(dtypes.int64).cast(dtypes.int32).simplify(), a.cast(dtypes.int32))

  def test_safe_cast_using_bounds(self):
    a = UOp.variable("a", 1, 10, dtype=dtypes.uint64)
    self.assertEqual(a.cast(dtypes.int16).cast(dtypes.int).simplify(), a.cast(dtypes.int))
    a = UOp.variable("a", -10, 10, dtype=dtypes.int32)
    self.assertEqual(a.cast(dtypes.int8).cast(dtypes.int64).simplify(), a.cast(dtypes.int64))
    self.assertEqual(a.cast(dtypes.int8).cast(dtypes.float).simplify(), a.cast(dtypes.float))

class TestExecALU(unittest.TestCase):
  def test_sqrt(self):
    self.assertEqual(exec_alu(Ops.SQRT, dtypes.float, (0.0,)), 0.0)

  def test_invalid_poison(self):
    # Invalid poisons any binary op regardless of result dtype: a comparison must not fold to a boolean
    self.assertIs(exec_alu(Ops.CMPLT, dtypes.bool, (Invalid, 1)), Invalid)
    self.assertIs(exec_alu(Ops.CMPNE, dtypes.bool, (Invalid, 1)), Invalid)
    self.assertIs(exec_alu(Ops.ADD, dtypes.weakint, (Invalid, 1)), Invalid)

  def test_div(self):
    self.assertEqual(exec_alu(Ops.CDIV, dtypes.int8, (8, 2)), 4)
    self.assertEqual(exec_alu(Ops.CDIV, dtypes.int8, (7, 3)), 2)
    self.assertEqual(exec_alu(Ops.CDIV, dtypes.int8, (7, -3)), -2)
    self.assertEqual(exec_alu(Ops.CDIV, dtypes.int8, (-50, 6)), -8)

  def test_floordiv(self):
    self.assertEqual(exec_alu(Ops.FLOORDIV, dtypes.int8, (8, 2)), 4)
    self.assertEqual(exec_alu(Ops.FLOORDIV, dtypes.int8, (7, 3)), 2)
    self.assertEqual(exec_alu(Ops.FLOORDIV, dtypes.int8, (7, -3)), -3)
    self.assertEqual(exec_alu(Ops.FLOORDIV, dtypes.int8, (-7, 3)), -3)
    self.assertEqual(exec_alu(Ops.FLOORDIV, dtypes.int8, (-50, 6)), -9)

  def test_floormod(self):
    self.assertEqual(exec_alu(Ops.FLOORMOD, dtypes.int8, (8, 2)), 0)
    self.assertEqual(exec_alu(Ops.FLOORMOD, dtypes.int8, (7, 3)), 1)
    self.assertEqual(exec_alu(Ops.FLOORMOD, dtypes.int8, (7, -3)), -2)
    self.assertEqual(exec_alu(Ops.FLOORMOD, dtypes.int8, (-7, 3)), 2)
    self.assertEqual(exec_alu(Ops.FLOORMOD, dtypes.int8, (-50, 6)), 4)

    np.testing.assert_allclose(exec_alu(Ops.MUL, dtypes.float32, (7.0, exec_alu(Ops.RECIPROCAL, dtypes.float32, (3.0,)))), 2+(1.0/3.0))
    np.testing.assert_allclose(exec_alu(Ops.MUL, dtypes.float32, (7.0, exec_alu(Ops.RECIPROCAL, dtypes.float32, (-3.0,)))), -2-(1.0/3.0))

  def test_recip(self):
    np.testing.assert_allclose(exec_alu(Ops.RECIPROCAL, dtypes.float32, (8,)), 1/8)
    np.testing.assert_allclose(exec_alu(Ops.RECIPROCAL, dtypes.float32, (7,)), 1/7)
    np.testing.assert_allclose(exec_alu(Ops.RECIPROCAL, dtypes.float32, (-3,)), 1/-3)
    np.testing.assert_allclose(exec_alu(Ops.RECIPROCAL, dtypes.float32, (-50,)), 1/-50)

    np.testing.assert_allclose(exec_alu(Ops.RECIPROCAL, dtypes.float32, ((32+521+3),)), 1/(32+521+3))
    np.testing.assert_allclose(exec_alu(Ops.RECIPROCAL, dtypes.float32, ((34**2),)), 1/(34**2))
    np.testing.assert_allclose(exec_alu(Ops.RECIPROCAL, dtypes.float32, (10,)), 1/10)

  def test_bool_cmplt(self):
    self.assertEqual(exec_alu(Ops.CMPLT, dtypes.bool, (False, False)), False)
    self.assertEqual(exec_alu(Ops.CMPLT, dtypes.bool, (False, True)), True)
    self.assertEqual(exec_alu(Ops.CMPLT, dtypes.bool, (True, False)), False)
    self.assertEqual(exec_alu(Ops.CMPLT, dtypes.bool, (True, True)), False)

  def test_bool_cmpne(self):
    self.assertEqual(exec_alu(Ops.CMPNE, dtypes.bool, (False, False)), False)
    self.assertEqual(exec_alu(Ops.CMPNE, dtypes.bool, (False, True)), True)
    self.assertEqual(exec_alu(Ops.CMPNE, dtypes.bool, (True, False)), True)
    self.assertEqual(exec_alu(Ops.CMPNE, dtypes.bool, (True, True)), False)

  def test_bool_where(self):
    self.assertEqual(exec_alu(Ops.WHERE, dtypes.bool, (False, False, False)), False)
    self.assertEqual(exec_alu(Ops.WHERE, dtypes.int, (False, 2, 4)), 4)
    np.testing.assert_allclose(exec_alu(Ops.WHERE, dtypes.float, (False, 2.2, 4.5)), 4.5)

  def test_overflow(self):
    self.assertEqual(exec_alu(Ops.ADD, dtypes.uint8, (250, 250)), 244)
    self.assertEqual(exec_alu(Ops.ADD, dtypes.uint8, (256, 0)), 0)
    self.assertEqual(exec_alu(Ops.ADD, dtypes.uint8, (0, -1)), 255)
    self.assertEqual(exec_alu(Ops.ADD, dtypes.uint8, (0, -1000)), 24)

    self.assertEqual(exec_alu(Ops.ADD, dtypes.int8, (127, 0)), 127)
    self.assertEqual(exec_alu(Ops.ADD, dtypes.int8, (-128, 0)), -128)
    self.assertEqual(exec_alu(Ops.ADD, dtypes.int8, (-100, -100)), 56)
    self.assertEqual(exec_alu(Ops.ADD, dtypes.int8, (-1000, -0)), 24)
    self.assertEqual(exec_alu(Ops.ADD, dtypes.int8, (-130, -0)), 126)

    self.assertEqual(exec_alu(Ops.ADD, dtypes.int8, (1, 1)), 2)
    self.assertEqual(exec_alu(Ops.ADD, dtypes.int8, (-128, 0)), -128)

    # test no truncate
    self.assertEqual(exec_alu(Ops.ADD, dtypes.uint8, (250, 250), truncate_output=False), 500)

class TestGatedStoreRewrite(unittest.TestCase):
  def test_tiny_gate_store(self):
    gmem = UOp.param(0, dtypes.float, (8,))
    gidx0 = UOp.special(4, 'gidx0')
    gate = gidx0<UOp.const(dtypes.weakint, 1)
    idx = UOp(Ops.INDEX, src=(gmem, (gidx0 * UOp.const(dtypes.weakint, 2)).valid(gate)))
    val = UOp.const(dtypes.float, 42.0)
    store = UOp(Ops.STORE, src=(idx, val))
    uops = to_uops_list([store])
    if_uop = next(u for u in uops if u.op is Ops.IF)
    endif = next(u for u in uops if u.op is Ops.ENDIF)
    assert endif.src[0] is if_uop
    gated_uops = tuple(uops[uops.index(if_uop)+1:uops.index(endif)])
    self.assertEqual(len(gated_uops), 1)
    self.assertIs(gated_uops[-1].op, Ops.STORE)
    self.assertEqual(len(gated_uops[-1].src), 2)

  def test_gate_some_stores(self):
    gmem0 = UOp.param(0, dtypes.float, (8,))
    gmem1 = UOp.param(1, dtypes.float, (8,))
    gidx0 = UOp.special(4, 'gidx0')
    idx = gidx0 * UOp.const(dtypes.weakint, 2)
    idx0 = UOp(Ops.INDEX, src=(gmem0, idx.valid(gidx0<UOp.const(dtypes.weakint, 1))))
    idx1 = UOp(Ops.INDEX, src=(gmem1, idx))
    val = UOp.const(dtypes.float, 42.0)
    stores = [UOp.store(idx0, val), UOp.store(idx1, val)]
    uops = to_uops_list(stores)
    if_uop = next(u for u in uops if u.op is Ops.IF)
    endif = next(u for u in uops if u.op is Ops.ENDIF)
    assert endif.src[0] is if_uop
    gated_uops = tuple(uops[uops.index(if_uop)+1:uops.index(endif)])
    self.assertEqual(len(gated_uops), 1)
    self.assertIs(gated_uops[-1].op, Ops.STORE)
    self.assertEqual(len(gated_uops[-1].src), 2)

  # scaled down version of TestLinearizerDumb.test_unmerged_ifs
  @unittest.skip("we don't merge ifs anymore")
  def test_merge_ifs_alt(self):
    gmem0 = UOp.param(0, dtypes.float, (8,))
    gmem1 = UOp.param(1, dtypes.float, (8,))
    gidx0 = UOp.special(4, 'gidx0')
    idx = gidx0*UOp.const(dtypes.weakint, 2)
    gate = gidx0<UOp.const(dtypes.weakint, 1)
    idx0 = UOp(Ops.INDEX, src=(gmem0, idx.valid(gate)))
    idx1 = UOp(Ops.INDEX, src=(gmem1, idx.valid(gate)))
    val = UOp.const(dtypes.float, 42.0)
    stores = [UOp.store(idx0, val), UOp.store(idx1, val)]
    uops = to_uops_list(stores)
    ifs = [u for u in uops if u.op is Ops.IF]
    endifs = [u for u in uops if u.op is Ops.ENDIF]
    self.assertEqual(len(ifs), 1)
    self.assertEqual(len(endifs), 1)
    gated_uops = tuple(uops[uops.index(ifs[0])+1:uops.index(endifs[0])])
    self.assertEqual(len(gated_uops), 2)
    for x in gated_uops: self.assertIs(x.op, Ops.STORE)
    for x in gated_uops: self.assertEqual(len(x.src), 2)

@unittest.skipIf(Device.DEFAULT == "METAL", "compiler bug")
@unittest.skipUnless(Ops.SHR in Device[Device.DEFAULT].renderer.code_for_op, "fast_idiv requires SHR")
class TestFastIdiv(unittest.TestCase):
  def test_division_power_of_two(self):
    for dt in (dtypes.int32, dtypes.uint32):
      g = UOp.param(0, dt, (3,))
      c = UOp.const(dt, 2)
      l = g.index(c)
      a = UOp(Ops.CDIV, dt, (l, c))
      uops = to_uops_list([a], ren=Device[Device.DEFAULT].renderer)
      Device[Device.DEFAULT].renderer.render(uops)
      ops = [x.op for x in uops]
      self.assertIn(Ops.SHR, ops, f"For dtype={dt} divison by power of two did not simplify to shift")
      self.assertNotIn(Ops.CDIV, ops, f"For dtype={dt} divison by power of two did not simplify to shift")

  def test_floormod_power_of_two(self):
    # FLOORMOD by a power of two lowers to AND (correct floor mod for any sign in two's complement)
    for dt in (dtypes.int32, dtypes.uint32):
      g = UOp.param(0, dt, (9,))
      c = UOp.const(dt, 8)
      a = UOp(Ops.FLOORMOD, dt, (g.index(c), c))
      uops = to_uops_list([a], ren=Device[Device.DEFAULT].renderer)
      ops = [x.op for x in uops]
      self.assertIn(Ops.AND, ops, f"For dtype={dt} FLOORMOD by pow2 did not simplify to AND")
      self.assertNotIn(Ops.CMOD, ops, f"For dtype={dt} FLOORMOD by pow2 left a MOD")
      self.assertNotIn(Ops.FLOORMOD, ops, f"For dtype={dt} FLOORMOD survived past late rewrite")

  def test_floordiv_power_of_two_uint(self):
    # uint FLOORDIV by a power of two lowers to a shift, leaving no IDIV/FLOORDIV in the kernel
    for dt in (dtypes.uint32, dtypes.uint64):
      g = UOp.param(0, dt, (3,))
      c = UOp.const(dt, 2)
      a = UOp(Ops.FLOORDIV, dt, (g.index(c), c))
      uops = to_uops_list([a], ren=Device[Device.DEFAULT].renderer)
      ops = [x.op for x in uops]
      self.assertIn(Ops.SHR, ops, f"For dtype={dt} FLOORDIV by power of two did not simplify to shift")
      self.assertNotIn(Ops.CDIV, ops, f"For dtype={dt} FLOORDIV by power of two did not simplify to shift")
      self.assertNotIn(Ops.FLOORDIV, ops, f"For dtype={dt} FLOORDIV survived past late rewrite")

  @Context(DISABLE_FAST_IDIV=0)
  @unittest.skipIf(Device.DEFAULT == "WEBGPU", "WEBGPU doesn't support long")
  def test_fast_idiv_and_mod(self):
    g = UOp.param(0, dtypes.uint32, (4,))
    c = UOp.const(dtypes.uint, 3)
    l = g.index(c)
    a = UOp(Ops.CDIV, src=(l, c))
    uops = to_uops_list([a], ren=Device[Device.DEFAULT].renderer)
    Device[Device.DEFAULT].renderer.render(uops)
    ops = [x.op for x in uops]
    self.assertIn(Ops.SHR, ops)
    self.assertNotIn(Ops.CDIV, ops)

    b = UOp(Ops.CMOD, src=(l, c))
    uops = to_uops_list([b], ren=Device[Device.DEFAULT].renderer)
    Device[Device.DEFAULT].renderer.render(uops)
    ops = [x.op for x in uops]
    self.assertIn(Ops.SHR, ops)
    self.assertNotIn(Ops.CMOD, ops)

  @Context(DISABLE_FAST_IDIV=0)
  def test_fast_idiv_bounded_numerator_zero(self):
    x = UOp.variable("x", 0, 1, dtype=dtypes.int32)
    for val in range(2):
      self.assertEqual(eval_uop(x.alu(Ops.CDIV, x.const_like(3)), vals=(val,)), cdiv(val, 3))

  @Context(DISABLE_FAST_IDIV=0)
  def test_fast_idiv_remove_powers_of_two(self):
    ridx = UOp.range(2**20, 0)
    uops = to_uops_list([ridx//(7*64)], ren=Device[Device.DEFAULT].renderer)
    ops = [x.op for x in uops]
    # this requires shifting out the powers of two before doing fast_idiv
    # (((ridx0>>6)*18725)>>17) instead of (int)((((long)(ridx0)*1198373)>>29))
    self.assertNotIn(Ops.CAST, ops)

  @unittest.expectedFailure
  def test_fast_idiv_overflow(self):
    # This will be possible with a slightly different method for fast_idiv
    g = UOp.param(0, dtypes.uint32, (8,))
    c = UOp.const(dtypes.uint, 7)
    l = UOp(Ops.LOAD, src=(g.index(c),))
    a = UOp(Ops.CDIV, src=(l, c))
    uops = to_uops_list([a], ren=Device[Device.DEFAULT].renderer)
    Device[Device.DEFAULT].renderer.render(uops)
    ops = [x.op for x in uops]
    self.assertIn(Ops.SHR, ops)
    self.assertNotIn(Ops.CDIV, ops)

  def test_disable_fast_idiv(self):
    g = UOp.param(0, dtypes.uint32, (4,))
    c = UOp.const(dtypes.uint, 3)
    l = g.index(c)
    a = UOp(Ops.CDIV, src=(l, c))
    with Context(DISABLE_FAST_IDIV=1):
      uops = to_uops_list([a], ren=Device[Device.DEFAULT].renderer)
    ops = [x.op for x in uops]
    self.assertNotIn(Ops.SHR, ops)
    self.assertIn(Ops.CDIV, ops)

class TestUOpMethod(unittest.TestCase):
  @unittest.skip("uops lt no longer ordered")
  def test_compare_alu_same_src_different_arg(self):
    a = UOp.const(dtypes.float, 2.0)
    b = UOp.const(dtypes.float, 3.0)

    add = UOp(Ops.ADD, src=(a, b))
    mul = UOp(Ops.MUL, src=(a, b))
    assert (add < mul) or (mul < add), "add and mul with same src should have an order"

  def test_uop_variables(self):
    a = UOp.variable("a", 1, 10)
    uop_var = Tensor(a.bind(1))
    st_var = Tensor.empty((2, 10))[:, :a.bind(1)]
    _, var_vals = (uop_var+st_var).linear_with_vars()
    self.assertEqual(len(var_vals), 1)
    self.assertEqual(list(var_vals)[0], a.expr)

  def test_const_factor(self):
    gidx0 = UOp(Ops.SPECIAL, src=(UOp.const(dtypes.int, 8),), arg='gidx0')
    self.assertEqual(UOp.const(dtypes.int, 17).const_factor(), 17)
    self.assertEqual(gidx0.const_factor(), 1)
    self.assertEqual((gidx0*3).const_factor(), 3)
    self.assertEqual((gidx0*3+6).const_factor(), 3)
    self.assertEqual((gidx0*3+1).const_factor(), 1)

  def test_cmp_self_folding_multidim(self):
    for shape in ((), (3,), (2, 3), (2, 3, 4)):
      x = Tensor.empty(*shape, dtype=dtypes.int).uop
      self.assertIs((x < x).simplify(), x.const_like(False, dtypes.bool))
      self.assertIs((x != x).simplify(), x.const_like(False, dtypes.bool))

  def test_replace(self):
    x = UOp.param(0, dtypes.int, (1,))
    self.assertEqual(x.replace(arg=UOp.param(1, dtypes.int, (1,)).arg).arg.slot, 1)
    with self.assertRaises(AssertionError): x.replace(field="a")

  def test_const_zero_neg_zero_different(self):
    # -0.0 and 0.0 must be different UOps (for IEEE754 correctness, e.g. 1/-0.0 = -inf)
    pos_zero = UOp.const(dtypes.float, 0.0)
    neg_zero = UOp.const(dtypes.float, -0.0)
    self.assertIsNot(pos_zero, neg_zero)
    self.assertNotEqual(hash(pos_zero.arg), hash(neg_zero.arg))

  def test_const_nan_same(self):
    # nan constants should be deduplicated
    nan1 = UOp.const(dtypes.float, float('nan'))
    nan2 = UOp.const(dtypes.float, float('nan'))
    self.assertIs(nan1, nan2)

class TestUOpStr(unittest.TestCase):
  def test_uop_str(self):
    a = UOp.const(dtypes.float, 2.0) + UOp.const(dtypes.float, 3.0)
    for _ in range(20): a = a + a
    assert len(str(a)) < 10_000, "exponential string growth"
    assert str(eval(str(a))) == str(a)

  def test_vectorized_str(self):
    vec = UOp(Ops.STACK, src=tuple(UOp.const(dtypes.int, x) for x in range(4)))
    assert str(eval(str(vec))) == str(vec)

  def test_reduceop_arg(self):
    sum_uop = Tensor.empty(32, 32).sum().uop
    assert str(eval(str(sum_uop))) == str(sum_uop)

class TestUPatHelpers(unittest.TestCase):
  def test_location(self):
    self.assertEqual(sym.patterns[-1][0].location[0].replace("\\", "/").split("/")[-1], "symbolic.py")
    self.assertEqual(spec_shared.patterns[0][0].location[0].replace("\\", "/").split("/")[-1], "spec.py")
    test_upat = UPat(Ops.CONST, dtypes.bool)
    self.assertEqual(test_upat.location[0].replace("\\", "/").split("/")[-1], __file__.replace("\\", "/").split("/")[-1])
    test_upat_named = test_upat.named("test_name")
    self.assertEqual(test_upat.location[0], test_upat_named.location[0])
    self.assertNotEqual(test_upat.location[1], test_upat_named.location[1])

class TestUopsObject(unittest.TestCase):
  def test_timing(self):
    with Timing("create 10k uops:"): ret = [UOp(Ops.CONST, dtypes.int, arg=10000000+i) for i in range(10000)]
    assert len(ret) == 10000

  def test_nested(self):
    a = UOp.new_buffer(Device.DEFAULT, 1, dtypes.char)
    for _ in range(10_000): a = a+a
    self.assertEqual(a.device, Device.DEFAULT)

class TestUOpRender(unittest.TestCase):
  def test_render_vectorize_empty(self):
    u = UOp(Ops.STACK, dtype=dtypes.void, src=())
    self.assertEqual(u.render(simplify=False), "{}")
  def test_render_vectorize_empty_simplified(self):
    u = UOp(Ops.STACK, dtype=dtypes.void, src=())
    self.assertEqual(u.render(), "{}")
  def test_render_vectorize_same(self):
    u = UOp(Ops.STACK, dtype=dtypes.int, src=(UOp.const(dtypes.int, 0), UOp.const(dtypes.int, 0), UOp.const(dtypes.int, 0)))
    self.assertEqual(u.render(simplify=False), "{0,0,0}")
  def test_render_vectorize_different(self):
    u = UOp(Ops.STACK, dtype=dtypes.int, src=(UOp.const(dtypes.int, 0), UOp.const(dtypes.int, 1), UOp.const(dtypes.int, 2)))
    self.assertEqual(u.render(simplify=False), "{0,1,2}")
  def test_render_vectorize_same_simplified(self):
    u = UOp(Ops.STACK, dtype=dtypes.int, src=(UOp.const(dtypes.int, 0), UOp.const(dtypes.int, 0), UOp.const(dtypes.int, 0)))
    self.assertEqual(u.render(), "{0,0,0}")
  def test_render_vectorize_different_simplified(self):
    u = UOp(Ops.STACK, dtype=dtypes.int, src=(UOp.const(dtypes.int, 0), UOp.const(dtypes.int, 1), UOp.const(dtypes.int, 2)))
    self.assertEqual(u.render(), "{0,1,2}")

class TestContiguousViewOffset(unittest.TestCase):
  def _check(self, u, expected): self.assertEqual(u.contiguous_view_offset(), expected)

  def test_simple(self): self._check(UOp.empty(10), 0)
  def test_shrink(self): self._check(UOp.empty(10)[1:8], 1)
  def test_2d(self): self._check(UOp.empty(2,5)[1, 2:4], 7)
  def test_shrink_to_one(self): self._check(UOp.empty(10)[1], 1)
  def test_expand_is_none(self): self._check(UOp.empty(1).expand(2), None)
  def test_shrink_invalid(self): self._check(UOp.empty(4).pad((2,2))[0], None)
  def test_strided(self): self._check(UOp.empty(4)[::2], None)

if __name__ == '__main__':
  unittest.main()
