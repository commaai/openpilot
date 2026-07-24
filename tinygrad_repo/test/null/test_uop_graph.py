import unittest, pytest
from tinygrad import dtypes, Variable
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import DEBUG, Context
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher, graph_rewrite, GroupOp, AxisType, broadcast_axes
from tinygrad.uop.symbolic import sym
from test.helpers import to_uops_list

simple_pm = PatternMatcher([
  (UPat.cvar('x', dtypes.int), lambda x: UOp.const(dtypes.float, 1.0) + UOp.const(dtypes.float, 2.0)),
  (UPat.cvar('x') + UPat.cvar('y'), lambda x,y: UOp.const(dtypes.float, x.arg+y.arg)),
  (UPat.cvar('x') * UPat.cvar('y') * UPat.cvar('z'), lambda x,y,z: UOp.const(dtypes.float, x.arg*y.arg*z.arg)),
  ((UPat.var('x') + UPat.cvar('c1')) + UPat.cvar('c2'), lambda x,c1,c2: x + (c1.arg+c2.arg)),
])

def const_values(u:UOp):
  if u.op is Ops.CONST: return (u.arg,) if not isinstance(u.arg, tuple) else u.arg
  if u.op is Ops.STACK: return tuple(x.arg for x in u.src)
  raise AssertionError(f"expected const-like UOp, got {u.op}")

class TestGraphRewriteConst(unittest.TestCase):
  def test_gep_const(self):
    v1 = UOp.const(dtypes.int, (0,1,2))
    v2 = v1.index(1)
    ret = graph_rewrite(v2, sym)
    self.assertEqual(ret.dtype, dtypes.int)
    self.assertEqual(ret.arg, 1)

  def test_add_const(self):
    v1 = UOp.const(dtypes.int, (0,1,2))
    v2 = UOp.const(dtypes.int, (5,6,7))
    ret = graph_rewrite(v1+v2, sym)
    self.assertEqual(ret.op, Ops.STACK)
    self.assertEqual(const_values(ret), (5,7,9))

  def test_add_const_lose_v(self):
    v1 = UOp.const(dtypes.int, (0,1,2))
    v2 = UOp.const(dtypes.int, (2,1,0))
    ret = graph_rewrite(v1+v2, sym)
    self.assertEqual(ret.op, Ops.STACK)
    self.assertEqual(const_values(ret), (2,2,2))

def xfail_broken_const_wraparound(fn):
  fn = pytest.mark.xfail(reason="const folding does not properly implement modular arithmetic")(fn)
  return unittest.expectedFailure(fn)
class TestModularWraparound(unittest.TestCase):
  def _test(self, uop:UOp, expected:int):
    results = to_uops_list([uop])
    self.assertEqual(len(results), 2)  # +1 for SINK
    self.assertEqual(results[0].op, Ops.CONST)
    self.assertEqual(results[0].dtype, uop.dtype)
    self.assertEqual(results[0].arg, expected)

  @xfail_broken_const_wraparound
  def test_cast(self):
    t = self._test
    t(UOp.const(dtypes.uint, 0xABCD17D6).cast(dtypes.uint8), 0xD6)
    t(UOp.const(dtypes.uint, 0xABCD17D6).cast(dtypes.uint8).cast(dtypes.uint), 0xD6)

  @xfail_broken_const_wraparound
  def test_mul(self):
    t = self._test
    t(UOp.const(dtypes.uint, 0xABCD17D6) * 0xAABBCCDD, 1147018174)
    t(UOp.const(dtypes.int, 0xABCD17D6) * 10, -1241321892)

  @xfail_broken_const_wraparound
  def test_div(self):
    t = self._test
    t(UOp.const(dtypes.uint, 0xABCD17D6) * 0xAABBCCDD // 11, 104274379)
    t(UOp.const(dtypes.int, 0xABCD17D6) * 10 // 11, -112847444)

  @xfail_broken_const_wraparound
  def test_neg(self):
    t = self._test
    t(-UOp.const(dtypes.uint8, 1), 0xFF)
    t(-UOp.const(dtypes.uint16, 1), 0xFFFF)
    t(-UOp.const(dtypes.uint32, 1), 0xFFFFFFFF)
    t(-UOp.const(dtypes.uint64, 1), 0xFFFFFFFFFFFFFFFF)

  @xfail_broken_const_wraparound
  def test_neg_min_int(self):
    t = self._test
    t(-UOp.const(dtypes.int8, -2**7), -2**7)
    t(-UOp.const(dtypes.int16, -2**15), -2**15)
    t(-UOp.const(dtypes.int32, -2**31), -2**31)
    t(-UOp.const(dtypes.int64, -2**63), -2**63)

  @xfail_broken_const_wraparound
  def test_payne_hanek_reduction_bug(self):
    t = self._test
    a = (UOp.const(dtypes.uint, 43748177600).cast(dtypes.uint) | 36).cast(dtypes.ulong)
    b = 2536655455 * a + 4294967296 * UOp.const(dtypes.ulong, 25366554550)
    c = (b + 2261737165) // 4611686018427387904
    t(c, 0)

class TestGraphRewrite(unittest.TestCase):
  def test_dedup(self):
    v1 = UOp.variable("v", 0, 1, dtypes.float)
    v2 = UOp.variable("v", 0, 1, dtypes.float)
    nout = graph_rewrite(v1+v2, PatternMatcher([]))
    self.assertIs(nout.src[0], nout.src[1])

  # NOTE: this shows why we can't have a UOp in arg
  @unittest.expectedFailure
  def test_no_dedup_args(self):
    a1 = UOp.variable("a1", UOp.const(dtypes.int, 0), UOp.const(dtypes.int, 11), dtypes.int)
    a2 = UOp.variable("a2", UOp.const(dtypes.int, 0), UOp.const(dtypes.int, 11), dtypes.int)
    sink = a1.sink(a2)
    variables = [x for x in graph_rewrite(sink, PatternMatcher([])).toposort() if x.op is Ops.PARAM and x.addrspace is AddrSpace.ALU]
    self.assertEqual(len(variables), 1)

  def test_simple(self):
    c1 = UOp.const(dtypes.float, 1.0)
    c2 = UOp.const(dtypes.float, 2.0)
    nout = graph_rewrite(c1+c2, simple_pm)
    self.assertEqual(nout.op, Ops.CONST)
    self.assertEqual(nout.arg, 3.0)

  def test_depth_2_late(self):
    c1 = UOp.const(dtypes.float, 1.0)
    c2 = UOp.const(dtypes.float, 2.0)
    c3 = UOp.const(dtypes.float, 3.0)
    nout = graph_rewrite(c1*c2*(c3+c3), simple_pm)
    self.assertEqual(nout.op, Ops.CONST)
    self.assertEqual(nout.arg, 12.0)

  def test_double(self):
    c1 = UOp.const(dtypes.float, 1.0)
    c2 = UOp.const(dtypes.float, 2.0)
    c3 = UOp.const(dtypes.float, 3.0)
    nout = graph_rewrite(c1+c2+c3, simple_pm)
    self.assertEqual(nout.op, Ops.CONST)
    self.assertEqual(nout.arg, 6.0)

  def test_triple(self):
    c1 = UOp.const(dtypes.float, 1.0)
    c2 = UOp.const(dtypes.float, 2.0)
    c3 = UOp.const(dtypes.float, 3.0)
    c4 = UOp.const(dtypes.float, 4.0)
    nout = graph_rewrite(c1+c2+c3+c4, simple_pm)
    self.assertEqual(nout.op, Ops.CONST)
    self.assertEqual(nout.arg, 10.0)

  def test_diamond(self):
    c1 = UOp.const(dtypes.float, 1.0)
    c2 = UOp.const(dtypes.float, 2.0)
    c3 = UOp.const(dtypes.float, 3.0)
    nout = graph_rewrite((c1+c2)+(c1+c3), simple_pm)
    self.assertEqual(nout.op, Ops.CONST)
    self.assertEqual(nout.arg, 7.0)

  def test_magic_4(self):
    c1 = UOp.const(dtypes.int, 4.0)
    nout = graph_rewrite(c1, simple_pm)
    self.assertEqual(nout.op, Ops.CONST)
    self.assertEqual(nout.arg, 3.0)

  def test_depth_2_fold(self):
    v = UOp.variable("v", 0, 1, dtypes.float)
    c1 = UOp.const(dtypes.float, 1.0)
    c2 = UOp.const(dtypes.float, 2.0)
    nout = graph_rewrite(v+c1+c2, simple_pm)
    self.assertEqual(nout.op, Ops.ADD)
    self.assertEqual(nout.src[0].op, Ops.PARAM)
    self.assertEqual(nout.src[1].op, Ops.CONST)
    self.assertEqual(nout.src[1].arg, 3.0)

  def test_commutative_work(self):
    a = UOp.variable('a', 0, 1)
    b = UOp.variable('b', 0, 1)
    self.assertIs((a+b).simplify(), (b+a).simplify())

  def test_consts_go_last_right_away(self):
    a = UOp.variable('a', 0, 1)
    tst = (2+a).simplify()
    self.assertIs(tst.src[0], a)
    self.assertIs(tst.src[1], a.const_like(2))

  def test_consts_go_last(self):
    a = UOp.variable('a', 0, 1)
    b = UOp.variable('b', 0, 1)
    c = UOp.variable('c', 0, 1)
    d = UOp.variable('d', 0, 1)
    outs = [2+a, 2+a+d+3+b+c+4, a.const_like(2)+a, (4+d)+c+(2+a)+b]
    for out in outs:
      sink = graph_rewrite(out, sym)
      print(sink.render())
      self.assertEqual(sink.op, Ops.ADD)
      self.assertEqual(sink.src[1].op, Ops.CONST)
      self.assertEqual(len([x for x in sink.toposort() if x.op is Ops.CONST]), 1)

class TestUOpGraph(unittest.TestCase):
  def test_add_constant_fold(self):
    c1 = UOp.const(dtypes.float, 1.0)
    c2 = UOp.const(dtypes.float, 2.0)
    out = c1+c2
    uops = to_uops_list([out])
    self.assertEqual(len(uops), 2)  # +1 for SINK
    out = uops[-2]
    self.assertEqual(out.op, Ops.CONST)
    self.assertEqual(out.arg, 3.0)

  def test_where_same_fold(self):
    v = UOp.variable('tmp', 0, 1)
    c0 = UOp.const(dtypes.weakint, 0)
    vc = v != c0
    c1 = UOp.const(dtypes.float, 1.0)
    out = vc.where(c1, c1)
    uops = to_uops_list([out])
    self.assertEqual(len(uops), 2)  # +1 for SINK
    out = uops[-2]
    self.assertEqual(out.op, Ops.CONST)
    self.assertEqual(out.arg, 1.0)

  def test_where_const_fold(self):
    bf = UOp.const(dtypes.bool, False)
    c1 = UOp.const(dtypes.float, 1.0)
    c2 = UOp.const(dtypes.float, 2.0)
    out = bf.where(c1, c2)
    uops = to_uops_list([out])
    self.assertEqual(len(uops), 2)  # +1 for SINK
    out = uops[-2]
    self.assertEqual(out.op, Ops.CONST)
    self.assertEqual(out.arg, 2.0)

  def test_const_cast(self):
    bf = UOp.const(dtypes.bool, False)
    out = bf.cast(dtypes.int)
    uops = to_uops_list([out])
    self.assertEqual(len(uops), 2)  # +1 for SINK
    out = uops[-2]
    self.assertEqual(out.op, Ops.CONST)
    self.assertEqual(out.arg, 0)

  def test_const_bitcast(self):
    bf = UOp.const(dtypes.float, 1.0)
    out = bf.bitcast(dtypes.uint32)
    uops = to_uops_list([out])
    self.assertEqual(len(uops), 2)  # +1 for SINK
    out = uops[-2]
    self.assertEqual(out.op, Ops.CONST)
    self.assertEqual(out.arg, 0x3F800000)

  @unittest.expectedFailure
  def test_const_shape_change_bitcast(self):
    bf = UOp.const(dtypes.uint8, 0x3F)
    out = bf.bitcast(dtypes.half)
    uops = to_uops_list([out])
    self.assertEqual(len(uops), 2)  # +1 for SINK

  @unittest.skip("this test isn't valid uops")
  def test_noop_vectorize_fold(self):
    d0 = UOp.param(0, dtypes.float, (1,))
    idx = UOp.const(dtypes.int, 0)
    ld = d0.load(idx, dtype=dtypes.float)
    vec = UOp(Ops.STACK, dtypes.float, (ld,))
    x = vec.index(0)
    alu = UOp(Ops.SQRT, src=(x, ))
    out = UOp(Ops.STORE, src=(d0, idx, alu))
    uops = to_uops_list([out])
    self.assertEqual(len([x for x in uops if x.op is Ops.STACK]), 0)

  @unittest.skip("this test isn't valid uops")
  def test_gep_vec_fold(self):
    d0 = UOp.param(0, dtypes.float, (1,))
    d1 = UOp.param(1, dtypes.float, (1,))
    d2 = UOp.param(2, dtypes.float, (1,))
    idx = UOp.const(dtypes.int, 0)
    def _test_vec(geps, count=4):
      vec = UOp(Ops.STACK, dtypes.float, geps)
      out = d0.index(idx).store(vec)
      uops = to_uops_list([out])
      if DEBUG >= 4:
        from tinygrad import Device
        print(Device[Device.DEFAULT].renderer.render(uops))
      return uops[-2].src[-1]  # -2 to skip SINK

    # possible
    val = d1.index(idx).load(dtype=dtypes.float)
    xyzw = tuple(val.index(i) for i in range(4))
    self.assertIs(_test_vec(xyzw).op, Ops.LOAD)

    # unaligned
    val = d1.index(idx).load(dtype=dtypes.float)
    wzyx = tuple(val.index(i) for i in reversed(range(4)))
    self.assertIs(_test_vec(wzyx).op, Ops.STACK)

    # different_size
    val = d1.index(idx).load(dtype=dtypes.float)
    xy = tuple(val.index(i) for i in range(2))
    self.assertIs(_test_vec(xy+xy).op, Ops.STACK)
    val = d1.index(idx).load(dtype=dtypes.float)
    xy = tuple(val.index(i) for i in range(2))
    self.assertIs(_test_vec(xy, count=2).op, Ops.STACK)

    # different vals
    val1 = d1.index(idx).load(dtype=dtypes.float)
    val2 = d2.index(idx).load(dtype=dtypes.float)
    xy1 = tuple(val1.index(i) for i in range(2))
    xy2 = tuple(val2.index(i) for i in range(2))
    self.assertIs(_test_vec(xy1+xy2).op, Ops.STACK)

  def test_gep_vec_const_fold(self):
    for vec_size in [2, 4, 8]:
      consts = [UOp.const(dtypes.float, float(i)) for i in range(vec_size)]
      vec = UOp(Ops.STACK, src=tuple(consts))
      with Context(SPEC=0):
        uops = to_uops_list([vec.index(i) for i in range(vec_size)])
        for uop, const in zip(uops, consts):
          self.assertEqual(uop, const)

  def test_cast_alu_fold(self):
    d0 = UOp.param(0, dtypes.bool, (1,))
    d1 = UOp.param(1, dtypes.int, (1,))
    idx = UOp.const(dtypes.int, 0)
    ld = d1.index(idx)
    alu = (ld<1).cast(dtypes.bool)
    out = d0.index(idx).store(alu)
    uops = to_uops_list([out])
    self.assertEqual(len([x for x in uops if x.op is Ops.CAST]), 0)

  def test_double_cast_fold(self):
    d0 = UOp.param(0, dtypes.float, (1,))
    d1 = UOp.param(1, dtypes.int, (1,))
    idx = UOp.const(dtypes.int, 0)
    ld = d1.index(idx)
    alu = ld.cast(dtypes.float).cast(dtypes.float)
    out = d0.index(idx).store(alu)
    uops = to_uops_list([out])
    self.assertEqual(len([x for x in uops if x.op is Ops.CAST]), 1)

  def test_depth_2_const_fold(self):
    v = UOp.variable("tmp", 0, 1, dtypes.int)
    c2 = UOp.const(dtypes.int, 2)
    c4 = UOp.const(dtypes.int, 4)
    vc = v+c2
    out = vc+c4
    uops = to_uops_list([out])
    self.assertEqual(len(uops), 5)  # +1 for SINK, +1 for the PARAM shape STACK
    out = uops[-2]  # -2 to skip SINK
    self.assertEqual(out.op, Ops.ADD)
    self.assertEqual(out.src[1].op, Ops.CONST)
    self.assertEqual(out.src[1].arg, 6)

  def test_bitcast_to_same_dtype_fold(self):
    for dt in dtypes.ints + dtypes.floats + (dtypes.bool,):
      d0 = UOp.param(0, dt, (1,))
      v = d0.index(UOp.const(dtypes.int, 0))
      uops = to_uops_list([v.bitcast(dt)])
      self.assertEqual(len([x for x in uops if x.op is Ops.BITCAST and x.dtype is dt]), 0, f"dtype = {dt}")

  def test_sub_with_cast_folds(self):
    a = Variable("a", 0, 5)
    uops = to_uops_list([a.cast(dtypes.int)+(-a).cast(dtypes.int)])
    assert uops[0] == UOp.const(dtypes.int, 0)
    assert uops[-1].op == Ops.SINK

  def test_where_on_gated_load_fold(self):
    ridx0 = UOp.range(100, 0)
    d0 = UOp.param(0, dtypes.long, (100,))
    ld = d0.index(ridx0.valid(ridx0<50))
    w = (ridx0<50).where(ld, 5)
    out = UOp.param(1, dtypes.long, (100,))
    uops = to_uops_list([out.index(ridx0).store(w)])
    for u in uops:
      assert u.op is not Ops.WHERE
      if u.op is Ops.LOAD and u.src[0].src[0].op is Ops.PARAM: assert u.src[1].arg==5

  def test_where_on_gated_load_folds_swapped_branches(self):
    ridx0 = UOp.range(100, 0)
    d0 = UOp.param(0, dtypes.long, (100,))
    ld = d0.index(ridx0.valid((ridx0<50).logical_not()))
    w = (ridx0<50).where(5, ld)
    uops = to_uops_list([w])
    for u in uops:
      assert u.op is not Ops.WHERE
      if u.op is Ops.LOAD: assert u.src[1].arg==5

  def test_where_on_gated_load_with_cast(self):
    ridx0 = UOp.range(100, 0)
    d0 = UOp.param(0, dtypes.int, (100,))
    gate_idx = ridx0.valid((ridx0<50))
    ld = d0.index(gate_idx).cast(dtypes.float)
    w = (ridx0<50).where(ld, 5.0)
    out = UOp.param(1, dtypes.float, (100,))
    uops = to_uops_list([out.index(ridx0).store(w)])
    for u in uops:
      assert u.op is not Ops.WHERE
      if u.op is Ops.LOAD and u.src[0].src[0].op is Ops.PARAM: assert u.src[1].arg == 5

  def test_where_on_casted_gated_load_extra_cond(self):
    ridx0 = UOp.range(100, 0)
    d0 = UOp.param(0, dtypes.float, (100,))
    ld = d0.index(ridx0.valid(ridx0<50))
    w = ((ridx0<50) & (ridx0>30)).where(ld, UOp.const(dtypes.float, 0)).cast(dtypes.half)
    out = UOp.param(1, dtypes.half, (100,))
    uops = to_uops_list([out.index(ridx0).store(w)])
    for u in uops:
      assert u.op is not Ops.WHERE

  def test_where_on_casted_gated_load_extra_cond_swapped(self):
    ridx0 = UOp.range(100, 0)
    d0 = UOp.param(0, dtypes.float, (100,))
    ld = d0.index(ridx0.valid(ridx0<50))
    w = ((ridx0<50) & (ridx0>30)).where(UOp.const(dtypes.float, 0), ld).cast(dtypes.half)
    out = UOp.param(1, dtypes.half, (100,))
    uops = to_uops_list([out.index(ridx0).store(w)])
    for u in uops:
      assert u.op is not Ops.WHERE

  def test_where_in_store_becomes_gate(self):
    ridx0 = UOp.range(100, 0)
    d0 = UOp.param(0, dtypes.long, (100,))
    idx = d0.index(ridx0)
    ld = idx.load()
    val = (ridx0<50).where(5, ld)
    st = idx.store(val).end(ridx0)
    uops = to_uops_list([st])
    for u in uops:
      assert u.op is not Ops.WHERE
      if u.op is Ops.STORE: assert u.src[1].arg==5

  def test_load_idx_becomes_int(self):
    # mnist indexing with split reduceop
    # Make sure we are not doign math on the loaded index, which would promote it to long
    c0 = UOp.param(0, dtypes.uchar, (128000,))
    c1 = UOp.range(UOp.const(dtypes.weakint, 512), 1, AxisType.LOOP)
    c2 = UOp.range(UOp.const(dtypes.weakint, 250), 2, AxisType.LOOP)
    c3 = UOp.param(1, dtypes.int, (512,))
    c4 = c3.index(c1)
    c5 = UOp.range(UOp.const(dtypes.weakint, 240), 0, AxisType.REDUCE)
    c6 = ((c2*UOp.const(dtypes.weakint, 240))+c5)
    c7 = UOp.param(2, dtypes.uchar, (60000,))
    c8 = c7.index(c6)
    c9 = ((c4<0).where((c4+60000), c4)!=c6.cast(dtypes.int)).where(0, c8.cast(dtypes.uint).cast(dtypes.uchar)).reduce(c5, arg=Ops.ADD)
    c10 = c0.index(((c1*UOp.const(dtypes.weakint, 250))+c2)).store(c9).end(c1, c2)
    uops = to_uops_list([c10])
    for u in uops:
      self.assertNotEqual(u.dtype, dtypes.long)

  def test_load_idx_no_math_on_loaded(self):
    # test the (x+y)<c pattern where x has loads - we shouldn't do math on loaded indices
    c0 = UOp.param(0, dtypes.uchar, (128000,))
    c1 = UOp.range(UOp.const(dtypes.weakint, 512), 1, AxisType.LOOP)
    c2 = UOp.range(UOp.const(dtypes.weakint, 250), 2, AxisType.LOOP)
    c3 = UOp.param(1, dtypes.int, (512,))
    c4 = c3.index(c1)  # c4 is a load
    c5 = UOp.range(UOp.const(dtypes.weakint, 240), 0, AxisType.REDUCE)
    c6 = ((c2*UOp.const(dtypes.weakint, 240))+c5)
    c7 = UOp.param(2, dtypes.uchar, (60000,))
    c8 = c7.index(c6)
    # (loaded + range) < const pattern - loaded value shouldn't be promoted to long
    loaded_idx = c4.cast(dtypes.weakint)
    comparison = (loaded_idx + c5) < UOp.const(dtypes.weakint, 60000)
    c9 = comparison.where(c8.cast(dtypes.uint).cast(dtypes.uchar), 0).reduce(c5, arg=Ops.ADD)
    c10 = c0.index(((c1*UOp.const(dtypes.weakint, 250))+c2)).store(c9).end(c1, c2)
    uops = to_uops_list([c10])
    for u in uops:
      self.assertNotEqual(u.dtype, dtypes.long)

  def test_fold_gated_load(self):
    glbl0 = UOp.param(0, dtypes.int, (1,))
    glbl1 = UOp.param(1, dtypes.int, (1,))
    glbl2 = UOp.param(2, dtypes.int, (1,))
    idx = UOp.const(dtypes.int, 0)
    ld0 = glbl1.index(UOp.invalid())
    ld1 = glbl2.index(idx.valid(UOp.const(dtypes.bool, True)))
    uops = to_uops_list([glbl0.index(idx).store(ld1+ld0)])
    # the gate and invalid value are deleted from ld1
    self.assertEqual(len([u for u in uops if u.op is Ops.LOAD]), 1)

  def test_fold_gated_load_local(self):
    glbl0 = UOp.param(0, dtypes.int, (16,))
    smem = UOp.placeholder((18,), dtypes.int, slot=0, addrspace=AddrSpace.LOCAL)
    lidx = UOp.special(16, "lidx0")
    st = smem.index(lidx).store(glbl0.index(lidx).load())
    barrier = st.barrier()
    ld0 = smem.after(barrier).index(UOp.invalid())
    ld1 = smem.after(barrier).index((lidx+2).valid(UOp.const(dtypes.bool, True)))
    uops = to_uops_list([glbl0.index(lidx).store(ld1+ld0)])

    # the gate and invalid value are deleted from ld1
    self.assertEqual(len([u for u in uops if u.op is Ops.LOAD]), 2)

  def test_fold_gated_store(self):
    glbl = UOp.param(0, dtypes.int, (1,))
    idx0 = UOp.const(dtypes.int, 0)
    val = UOp.const(dtypes.int, 42)
    st0 = glbl.index(UOp.invalid()).store(val)
    st1 = glbl.index(idx0.valid(UOp.const(dtypes.bool, True))).store(val)
    uops = to_uops_list([st0, st1])
    # only the second store happens
    self.assertEqual(len([u for u in uops if u.op is Ops.STORE]), 1)

  @unittest.skip("this is a uop type error")
  def test_asserts_bad_gate(self):
    glbl0 = UOp.param(0, dtypes.int, (1,))
    idx = UOp.const(dtypes.int, 0)
    bad_gate = UOp.const(dtypes.int, 1)
    with self.assertRaises(AssertionError): to_uops_list([UOp(Ops.STORE, src=(glbl0, idx, UOp.const(dtypes.int, 42), bad_gate))])

  def test_after_end(self):
    r = UOp.range(10, 0)

    c = r + 1
    self.assertIn(r, c.ranges)

    e = UOp.const(dtypes.int, 1).end(r)
    self.assertNotIn(r, e.ranges)

    a = c.after(e)
    self.assertNotIn(r, a.ranges)

class TestReduceCollapse(unittest.TestCase):
  def test_multi_range_reduce_add(self):
    """Test that (x + y).reduce(r1, r2) distributes over multiple ranges"""
    from tinygrad.codegen.simplify import pm_reduce_collapse
    # Create two ranges
    r1 = UOp.range(3, 0)
    r2 = UOp.range(4, 1)
    # Create x + y where x and y depend on different ranges
    x = r1.cast(dtypes.float)
    y = r2.cast(dtypes.float)
    # (x + y).reduce(r1, r2) should be rewritten
    red = (x + y).reduce(r1, r2, arg=Ops.ADD)
    self.assertEqual(len(red.src), 3)  # value + 2 ranges
    result = graph_rewrite(red, pm_reduce_collapse, name='test')
    # Should become add of two separate reduces
    self.assertEqual(result.op, Ops.ADD)

class TestMovementOps(unittest.TestCase):
  def test_pm_mops_partial_reshape_index_removes_reshape(self):
    from tinygrad.schedule.rangeify import pm_mops
    src = UOp.param(0, dtypes.float, shape=(32, 4))
    r0, r1 = UOp.range(4, 0), UOp.range(8, 1)
    result = graph_rewrite(src.reshape((4, 8, 4)).index(r0, r1), pm_mops, name="test")
    self.assertEqual(result.op, Ops.INDEX)
    self.assertIs(result.src[0], src)
    self.assertEqual(result.shape, (4,))
    self.assertNotIn(Ops.RESHAPE, [u.op for u in result.toposort()])

  def test_pm_mops_partial_reshape_index_suffix_mismatch_does_nothing(self):
    from tinygrad.schedule.rangeify import pm_mops
    src = UOp.param(0, dtypes.float, shape=(2, 6))
    result = graph_rewrite(src.reshape((2, 3, 2)).index(UOp.range(2, 0)), pm_mops, name="test")
    self.assertEqual(result.op, Ops.INDEX)
    self.assertEqual(result.src[0].op, Ops.RESHAPE)

class TestConstBufferize(unittest.TestCase):
  def test_const_bufferize_with_ranges(self):
    """Test that CONST.BUFFERIZE with ranges is folded correctly.

    BUFFERIZE can have ranges as additional sources beyond the value.
    The pattern at rangeify.py uses allow_any_len=True because
    CONST doesn't depend on ranges (constant is same value everywhere).
    """
    from tinygrad.schedule.rangeify import pm_const_buffer_folding, BufferizeOpts
    c = UOp.const(dtypes.float, 42.0)
    r1 = UOp.range(3, 0)
    bufferize_with_range = UOp(Ops.STAGE, src=(c, r1), arg=BufferizeOpts(device="CPU"))
    self.assertEqual(len(bufferize_with_range.src), 2)  # const + 1 range

    result = graph_rewrite(bufferize_with_range, pm_const_buffer_folding, name='test')
    # BUFFERIZE should be removed, result is const broadcast to shape
    self.assertNotEqual(result.op, Ops.STAGE)
    const_vals = [u.arg for u in result.toposort() if u.op is Ops.CONST and u.dtype == dtypes.float]
    self.assertIn(42.0, const_vals)

  def test_const_bufferize_with_multiple_ranges(self):
    """Test CONST.BUFFERIZE with multiple ranges is also folded."""
    from tinygrad.schedule.rangeify import pm_const_buffer_folding, BufferizeOpts
    c = UOp.const(dtypes.float, 3.14)
    r1 = UOp.range(3, 0)
    r2 = UOp.range(4, 1)
    bufferize_with_ranges = UOp(Ops.STAGE, src=(c, r1, r2), arg=BufferizeOpts(device="CPU"))
    self.assertEqual(len(bufferize_with_ranges.src), 3)  # const + 2 ranges

    result = graph_rewrite(bufferize_with_ranges, pm_const_buffer_folding, name='test')
    # BUFFERIZE should be removed
    self.assertNotEqual(result.op, Ops.STAGE)
    const_vals = [u.arg for u in result.toposort() if u.op is Ops.CONST and u.dtype == dtypes.float]
    self.assertIn(3.14, const_vals)

class TestUOpTags(unittest.TestCase):
  def test_inc_by_one(self):
    g = UOp.const(dtypes.int, 1) + UOp.const(dtypes.int, 1)
    assert g.ssimplify() == 2
    pm_plus_1 = PatternMatcher([(UPat(Ops.CONST, name="x"), lambda x: x.replace(arg=x.arg+1, tag=1) if x.tag is None else None)])
    pm_strip_tags = PatternMatcher([(UPat(GroupOp.All, name="x"), lambda x: x.replace(tag=None) if x.tag is not None else None)])
    g = graph_rewrite(g, pm_plus_1)
    assert g.ssimplify() == 4
    g = graph_rewrite(g, pm_plus_1)
    assert g.ssimplify() == 4
    g = graph_rewrite(g, pm_strip_tags)
    assert g.ssimplify() == 4
    g = graph_rewrite(g, pm_plus_1)
    assert g.ssimplify() == 6

class TestUOpGetItem(unittest.TestCase):
  def _placeholder(self, shape, dtype=dtypes.half):
    return UOp.placeholder(shape, dtype, slot=0, addrspace=AddrSpace.LOCAL)

  # full slices (no shrink)
  def test_full_slice(self):
    p = self._placeholder((64, 64))
    self.assertEqual(p[:, :].shape, (64, 64))
  def test_full_slice_explicit(self):
    p = self._placeholder((64, 64))
    self.assertEqual(p[0:64, 0:64].shape, (64, 64))

  # partial slices (shrink)
  def test_shrink_cols(self):
    p = self._placeholder((64, 80))
    self.assertEqual(p[:, :64].shape, (64, 64))
  def test_shrink_rows(self):
    p = self._placeholder((80, 64))
    self.assertEqual(p[:64, :].shape, (64, 64))
  def test_shrink_both(self):
    p = self._placeholder((80, 80))
    self.assertEqual(p[:64, :64].shape, (64, 64))
  def test_shrink_start(self):
    p = self._placeholder((64, 64))
    self.assertEqual(p[8:, :].shape, (56, 64))
  def test_shrink_start_and_end(self):
    p = self._placeholder((64, 64))
    self.assertEqual(p[8:56, 4:60].shape, (48, 56))

  # mixed slice and index
  def test_index_and_slice(self):
    p = self._placeholder((64, 80))
    r = UOp.range(64, 100)
    result = p[r, :64]
    self.assertEqual(result.shape, (64,))
  def test_slice_and_index(self):
    p = self._placeholder((80, 64))
    r = UOp.range(64, 100)
    result = p[:64, r]
    self.assertEqual(result.shape, (64,))
  def test_shrink_then_index(self):
    p = self._placeholder((64, 80))
    s = p[:, :64]
    r = UOp.range(64, 100)
    result = s[r]
    self.assertEqual(result.shape, (64,))

  # integer index (no slice)
  def test_int_index(self):
    p = self._placeholder((64, 64))
    result = p[0]
    self.assertEqual(result.shape, (64,))

  # ellipsis
  def test_ellipsis_all_slices(self):
    p = self._placeholder((64, 80))
    self.assertEqual(p[..., :64].shape, (64, 64))
  def test_ellipsis_with_int(self):
    p = self._placeholder((64, 80))
    r = UOp.range(64, 100)
    result = p[..., r]
    self.assertEqual(result.op, Ops.INDEX)
  def test_ellipsis_only(self):
    p = self._placeholder((64, 64))
    self.assertEqual(p[...].shape, (64, 64))

  # all slices should not create a bare INDEX
  def test_all_slices_no_index(self):
    p = self._placeholder((64, 80))
    result = p[:, :64]
    self.assertNotEqual(result.op, Ops.INDEX)
  def test_all_full_slices_no_index(self):
    p = self._placeholder((64, 64))
    result = p[:, :]
    self.assertNotEqual(result.op, Ops.INDEX)

class TestUOpBroadcast(unittest.TestCase):
  def test_broadcast_row(self):
    a = UOp.const(dtypes.float, 1, shape=(4, 8))
    b = UOp.const(dtypes.float, 2, shape=(4, 1))
    c = a + b
    self.assertEqual(c.shape, (4, 8))
    self.assertEqual(c.op, Ops.ADD)

  def test_broadcast_col(self):
    a = UOp.const(dtypes.float, 1, shape=(4, 8))
    b = UOp.const(dtypes.float, 2, shape=(1, 8))
    c = a + b
    self.assertEqual(c.shape, (4, 8))
    self.assertEqual(c.op, Ops.ADD)

  def test_broadcast_lower_dim(self):
    a = UOp.const(dtypes.float, 1, shape=(4, 8))
    b = UOp.const(dtypes.float, 2, shape=(8,))
    c = a * b
    self.assertEqual(c.shape, (4, 8))
    self.assertEqual(c.op, Ops.MUL)

  def test_broadcast_scalar(self):
    a = UOp.const(dtypes.float, 1, shape=(4, 8))
    c = a * 2
    self.assertEqual(c.shape, (4, 8))
    self.assertEqual(c.op, Ops.MUL)

  def test_broadcast_symbolic_same_shape(self):
    t = Variable("t", 1, 10)
    a = UOp.const(dtypes.float, 1, shape=(1, 1, t))
    b = UOp.const(dtypes.float, 2, shape=(1, 1, t))
    c = a + b
    self.assertEqual(c.op, Ops.ADD)

  def test_broadcast_axes(self):
    t = Variable("t", 1, 10)
    self.assertEqual(broadcast_axes((4, 8), (4, 8)), ())
    self.assertEqual(broadcast_axes((8,), (4, 8)), (0,))
    self.assertEqual(broadcast_axes((), (4, 8)), (0, 1))
    self.assertEqual(broadcast_axes((3, 1), (4, 3, 8)), (0, 2))
    self.assertEqual(broadcast_axes((1, 8), (1, 8)), ())
    self.assertEqual(broadcast_axes((t, 8), (t, 8)), ())
    self.assertEqual(broadcast_axes((1, 8), (t, 8)), (0,))
    with self.assertRaises(RuntimeError): broadcast_axes((4, 8), (8,))

if __name__ == '__main__':
  unittest.main(verbosity=2)
