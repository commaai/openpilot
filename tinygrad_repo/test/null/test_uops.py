# uops tests that pass on NULL backend (no copyout needed)
import unittest
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.helpers import Timing, Context
from tinygrad.dtype import dtypes, ConstFloat  # noqa: F401
from tinygrad.device import Device
from tinygrad.uop.ops import Ops, UOp, UPat, KernelInfo, exec_alu
from tinygrad.uop.spec import shared_spec
from tinygrad.uop.symbolic import sym
from test.helpers import get_uops

def to_uops_list(u:list[UOp], ren=None) -> list[UOp]:
  sink = UOp.group(*u)
  for r in sink.ranges: sink = sink.end(r)
  ret = get_uops(sink.sink(arg=KernelInfo(opts_to_apply=())), ren)
  assert ret[-1].op is Ops.SINK
  return ret

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

  def test_div(self):
    self.assertEqual(exec_alu(Ops.IDIV, dtypes.int8, (8, 2)), 4)
    self.assertEqual(exec_alu(Ops.IDIV, dtypes.int8, (7, 3)), 2)
    self.assertEqual(exec_alu(Ops.IDIV, dtypes.int8, (7, -3)), -2)
    self.assertEqual(exec_alu(Ops.IDIV, dtypes.int8, (-50, 6)), -8)

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

class TestConstantFolding(unittest.TestCase):
  def test_cast_const(self):
    t = Tensor(1, dtype=dtypes.float).cast(dtypes.int)
    si = t.schedule()
    assert len(si) == 0

class TestGatedStoreRewrite(unittest.TestCase):
  def test_tiny_gate_store(self):
    gmem = UOp(Ops.PARAM, dtypes.float.ptr(), (), 0)
    gidx0 = UOp(Ops.SPECIAL, dtypes.int, (UOp.const(dtypes.int, 4),), 'gidx0')
    gate = gidx0<UOp.const(dtypes.int, 1)
    idx = UOp(Ops.INDEX, dtypes.float.ptr(), (gmem, (gidx0 * UOp.const(dtypes.int, 2)).valid(gate)))
    val = UOp.const(dtypes.float, 42.0)
    store = UOp(Ops.STORE, dtypes.void, (idx, val))
    uops = to_uops_list([store])
    if_uop = next(u for u in uops if u.op is Ops.IF)
    endif = next(u for u in uops if u.op is Ops.ENDIF)
    assert endif.src[0] is if_uop
    gated_uops = tuple(uops[uops.index(if_uop)+1:uops.index(endif)])
    self.assertEqual(len(gated_uops), 1)
    self.assertIs(gated_uops[-1].op, Ops.STORE)

  def test_gate_some_stores(self):
    gmem0 = UOp(Ops.PARAM, dtypes.float.ptr(), (), 0)
    gmem1 = UOp(Ops.PARAM, dtypes.float.ptr(), (), 1)
    gidx0 = UOp(Ops.SPECIAL, dtypes.int, (UOp.const(dtypes.int, 4),), 'gidx0')
    idx = gidx0 * UOp.const(dtypes.int, 2)
    idx0 = UOp(Ops.INDEX, dtypes.float.ptr(), (gmem0, idx.valid(gidx0<UOp.const(dtypes.int, 1))))
    idx1 = UOp(Ops.INDEX, dtypes.float.ptr(), (gmem1, idx))
    val = UOp.const(dtypes.float, 42.0)
    stores = [UOp.store(idx0, val), UOp.store(idx1, val)]
    uops = to_uops_list(stores)
    if_uop = next(u for u in uops if u.op is Ops.IF)
    endif = next(u for u in uops if u.op is Ops.ENDIF)
    assert endif.src[0] is if_uop
    gated_uops = tuple(uops[uops.index(if_uop)+1:uops.index(endif)])
    self.assertEqual(len(gated_uops), 1)
    self.assertIs(gated_uops[-1].op, Ops.STORE)

  # scaled down version of TestLinearizerDumb.test_unmerged_ifs
  @unittest.skip("we don't merge ifs anymore")
  def test_merge_ifs_alt(self):
    gmem0 = UOp(Ops.PARAM, dtypes.float.ptr(), (), 0)
    gmem1 = UOp(Ops.PARAM, dtypes.float.ptr(), (), 1)
    gidx0 = UOp(Ops.SPECIAL, dtypes.int, (UOp.const(dtypes.int, 4),), 'gidx0')
    idx = gidx0*UOp.const(dtypes.int, 2)
    gate = gidx0<UOp.const(dtypes.int, 1)
    idx0 = UOp(Ops.INDEX, dtypes.float.ptr(), (gmem0, idx, gate))
    idx1 = UOp(Ops.INDEX, dtypes.float.ptr(), (gmem1, idx, gate))
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

@unittest.skipIf(Device.DEFAULT == "METAL", "compiler bug")
@unittest.skipUnless(Ops.SHR in Device[Device.DEFAULT].renderer.code_for_op, "fast_idiv requires SHR")
class TestFastIdiv(unittest.TestCase):
  def test_division_power_of_two(self):
    for dt in (dtypes.int32, dtypes.uint32):
      g = UOp(Ops.PARAM, dt.ptr(), (), 0)
      c = UOp.const(dt, 2)
      l = g.index(c)
      a = UOp(Ops.IDIV, dt, (l, c))
      uops = to_uops_list([a], ren=Device[Device.DEFAULT].renderer)
      Device[Device.DEFAULT].renderer.render(uops)
      ops = [x.op for x in uops]
      self.assertIn(Ops.SHR, ops, f"For dtype={dt} divison by power of two did not simplify to shift")
      self.assertNotIn(Ops.IDIV, ops, f"For dtype={dt} divison by power of two did not simplify to shift")

  @unittest.skipIf(Device.DEFAULT == "WEBGPU", "WEBGPU doesn't support long")
  def test_fast_idiv_and_mod(self):
    g = UOp(Ops.PARAM, dtypes.uint32.ptr(), (), 0)
    c = UOp.const(dtypes.uint, 3)
    l = g.index(c)
    a = UOp(Ops.IDIV, dtypes.uint, (l, c))
    uops = to_uops_list([a], ren=Device[Device.DEFAULT].renderer)
    Device[Device.DEFAULT].renderer.render(uops)
    ops = [x.op for x in uops]
    self.assertIn(Ops.SHR, ops)
    self.assertNotIn(Ops.IDIV, ops)

    b = UOp(Ops.MOD, dtypes.uint, (l, c))
    uops = to_uops_list([b], ren=Device[Device.DEFAULT].renderer)
    Device[Device.DEFAULT].renderer.render(uops)
    ops = [x.op for x in uops]
    self.assertIn(Ops.SHR, ops)
    self.assertNotIn(Ops.MOD, ops)

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
    g = UOp(Ops.PARAM, dtypes.uint32.ptr(), (), 0)
    c = UOp.const(dtypes.uint, 7)
    l = UOp(Ops.LOAD, dtypes.uint, (g.index(c),))
    a = UOp(Ops.IDIV, dtypes.uint, (l, c))
    uops = to_uops_list([a], ren=Device[Device.DEFAULT].renderer)
    Device[Device.DEFAULT].renderer.render(uops)
    ops = [x.op for x in uops]
    self.assertIn(Ops.SHR, ops)
    self.assertNotIn(Ops.IDIV, ops)

  def test_disable_fast_idiv(self):
    g = UOp(Ops.PARAM, dtypes.uint32.ptr(), (), 0)
    c = UOp.const(dtypes.uint, 3)
    l = g.index(c)
    a = UOp(Ops.IDIV, dtypes.uint, (l, c))
    with Context(DISABLE_FAST_IDIV=1):
      uops = to_uops_list([a], ren=Device[Device.DEFAULT].renderer)
    ops = [x.op for x in uops]
    self.assertNotIn(Ops.SHR, ops)
    self.assertIn(Ops.IDIV, ops)

class TestUOpMethod(unittest.TestCase):
  @unittest.skip("uops lt no longer ordered")
  def test_compare_alu_same_src_different_arg(self):
    a = UOp.const(dtypes.float, 2.0)
    b = UOp.const(dtypes.float, 3.0)

    add = UOp(Ops.ADD, dtypes.float, (a, b))
    mul = UOp(Ops.MUL, dtypes.float, (a, b))
    assert (add < mul) or (mul < add), "add and mul with same src should have an order"

  def test_uop_variables(self):
    a = UOp.variable("a", 1, 10)
    uop_var = Tensor(a.bind(1))
    st_var = Tensor.empty((2, 10))[:, :a.bind(1)]
    _, var_vals = (uop_var+st_var).schedule_with_vars()
    self.assertEqual(len(var_vals), 1)
    self.assertEqual(list(var_vals)[0], a.expr)

  def test_const_factor(self):
    gidx0 = UOp(Ops.SPECIAL, dtypes.int, (UOp.const(dtypes.int, 8),), 'gidx0')
    self.assertEqual(UOp.const(dtypes.int, 17).const_factor(), 17)
    self.assertEqual(gidx0.const_factor(), 1)
    self.assertEqual((gidx0*3).const_factor(), 3)
    self.assertEqual((gidx0*3+6).const_factor(), 3)
    self.assertEqual((gidx0*3+1).const_factor(), 1)

  def test_replace(self):
    x = UOp(Ops.PARAM, dtypes.int.ptr(), (), 0)
    self.assertIs(x.replace(arg=None).arg, None)
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
    vec = UOp(Ops.VECTORIZE, dtypes.int.vec(4), tuple(UOp.const(dtypes.int, x) for x in range(4)))
    assert str(eval(str(vec))) == str(vec)

  def test_device_arg(self):
    device = UOp(Ops.DEVICE, arg="CL")
    assert str(eval(str(device))) == str(device)

  def test_reduceop_arg(self):
    sum_uop = Tensor.empty(32, 32).sum().uop
    assert str(eval(str(sum_uop))) == str(sum_uop)

class TestUPatHelpers(unittest.TestCase):
  def test_location(self):
    self.assertEqual(sym.patterns[-1][0].location[0].replace("\\", "/").split("/")[-1], "symbolic.py")
    self.assertEqual(shared_spec.patterns[0][0].location[0].replace("\\", "/").split("/")[-1], "spec.py")
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
    u = UOp(Ops.VECTORIZE, dtype=dtypes.int.vec(0), src=())
    self.assertEqual(u.render(simplify=False), "{}")
  def test_render_vectorize_empty_simplified(self):
    u = UOp(Ops.VECTORIZE, dtype=dtypes.int.vec(0), src=())
    self.assertEqual(u.render(), "{}")
  def test_render_vectorize_same(self):
    u = UOp(Ops.VECTORIZE, dtype=dtypes.int.vec(3), src=(UOp.const(dtypes.int, 0), UOp.const(dtypes.int, 0), UOp.const(dtypes.int, 0)))
    self.assertEqual(u.render(simplify=False), "{0, ...}")
  def test_render_vectorize_different(self):
    u = UOp(Ops.VECTORIZE, dtype=dtypes.int.vec(3), src=(UOp.const(dtypes.int, 0), UOp.const(dtypes.int, 1), UOp.const(dtypes.int, 2)))
    self.assertEqual(u.render(simplify=False), "{0,1,2}")
  def test_render_vectorize_same_simplified(self):
    u = UOp(Ops.VECTORIZE, dtype=dtypes.int.vec(3), src=(UOp.const(dtypes.int, 0), UOp.const(dtypes.int, 0), UOp.const(dtypes.int, 0)))
    self.assertEqual(u.render(), "0")
  def test_render_vectorize_different_simplified(self):
    u = UOp(Ops.VECTORIZE, dtype=dtypes.int.vec(3), src=(UOp.const(dtypes.int, 0), UOp.const(dtypes.int, 1), UOp.const(dtypes.int, 2)))
    self.assertEqual(u.render(), "(0, 1, 2)")

if __name__ == '__main__':
  unittest.main()
