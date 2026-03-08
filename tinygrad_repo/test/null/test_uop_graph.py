import unittest, pytest
from tinygrad import dtypes, Variable
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import DEBUG, Context
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher, track_rewrites, graph_rewrite, GroupOp, AxisType
from tinygrad.uop.symbolic import sym
from tinygrad.codegen.late.expander import expander
from test.test_uops import to_uops_list

simple_pm = PatternMatcher([
  (UPat.cvar('x', dtypes.int), lambda x: UOp.const(dtypes.float, 1.0) + UOp.const(dtypes.float, 2.0)),
  (UPat.cvar('x') + UPat.cvar('y'), lambda x,y: UOp.const(dtypes.float, x.arg+y.arg)),
  (UPat.cvar('x') * UPat.cvar('y') * UPat.cvar('z'), lambda x,y,z: UOp.const(dtypes.float, x.arg*y.arg*z.arg)),
  ((UPat.var('x') + UPat.cvar('c1')) + UPat.cvar('c2'), lambda x,c1,c2: x + (c1.arg+c2.arg)),
])

class TestGraphRewriteConst(unittest.TestCase):
  def test_gep_const(self):
    v1 = UOp.const(dtypes.int.vec(3), (0,1,2))
    v2 = v1.gep(1)
    ret = graph_rewrite(v2, sym)
    self.assertEqual(ret.dtype, dtypes.int)
    self.assertEqual(ret.arg, 1)

  def test_gep_const_single(self):
    v1 = UOp.const(dtypes.int.vec(3), 4)
    v2 = v1.gep(1)
    ret = graph_rewrite(v2, sym)
    self.assertEqual(ret.dtype, dtypes.int)
    self.assertEqual(ret.arg, 4)

  def test_add_const(self):
    v1 = UOp.const(dtypes.int.vec(3), (0,1,2))
    v2 = UOp.const(dtypes.int.vec(3), (5,6,7))
    ret = graph_rewrite(v1+v2, sym)
    self.assertEqual(ret.op, Ops.VCONST)
    self.assertEqual(ret.dtype, dtypes.int.vec(3))
    self.assertEqual(ret.arg, (5,7,9))

  def test_add_const_lose_v(self):
    v1 = UOp.const(dtypes.int.vec(3), (0,1,2))
    v2 = UOp.const(dtypes.int.vec(3), (2,1,0))
    ret = graph_rewrite(v1+v2, sym)
    self.assertEqual(ret.op, Ops.CONST)
    self.assertEqual(ret.dtype, dtypes.int.vec(3))
    self.assertEqual(ret.arg, 2)

xfail_broken_const_wraparound = pytest.mark.xfail(reason="const folding does not properly implement modular arithmetic")
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
    v1 = UOp(Ops.DEFINE_VAR, dtypes.float)
    v2 = UOp(Ops.DEFINE_VAR, dtypes.float)
    nout = graph_rewrite(v1+v2, PatternMatcher([]))
    self.assertIs(nout.src[0], nout.src[1])

  # NOTE: this shows why we can't have a UOp in arg
  @unittest.expectedFailure
  def test_no_dedup_args(self):
    a1 = UOp(Ops.DEFINE_VAR, dtypes.int, (), ("a1", UOp.const(dtypes.int, 0), UOp.const(dtypes.int, 11)))
    a2 = UOp(Ops.DEFINE_VAR, dtypes.int, (), ("a2", UOp.const(dtypes.int, 0), UOp.const(dtypes.int, 11)))
    sink = a1.sink(a2)
    define_vars = [x for x in graph_rewrite(sink, PatternMatcher([])).toposort() if x.op is Ops.DEFINE_VAR]
    self.assertEqual(len(define_vars), 1)

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
    v = UOp(Ops.DEFINE_VAR, dtypes.float)
    c1 = UOp.const(dtypes.float, 1.0)
    c2 = UOp.const(dtypes.float, 2.0)
    nout = graph_rewrite(v+c1+c2, simple_pm)
    self.assertEqual(nout.op, Ops.ADD)
    self.assertEqual(nout.src[0].op, Ops.DEFINE_VAR)
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
    outs = [2+a, 2+a+d+3+b+c+4, UOp(Ops.ADD, a.dtype, src=(a.const_like(2), a)), (4+d)+c+(2+a)+b]
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
    out = UOp(Ops.ADD, dtypes.float, (c1, c2))
    uops = to_uops_list([out])
    self.assertEqual(len(uops), 2)  # +1 for SINK
    out = uops[-2]
    self.assertEqual(out.op, Ops.CONST)
    self.assertEqual(out.arg, 3.0)

  def test_where_same_fold(self):
    v = UOp.variable('tmp', 0, 1)
    c0 = UOp.const(dtypes.index, 0)
    vc = UOp(Ops.CMPNE, dtypes.bool, (v, c0))
    c1 = UOp.const(dtypes.float, 1.0)
    out = UOp(Ops.WHERE, dtypes.float, (vc, c1, c1))
    uops = to_uops_list([out])
    self.assertEqual(len(uops), 2)  # +1 for SINK
    out = uops[-2]
    self.assertEqual(out.op, Ops.CONST)
    self.assertEqual(out.arg, 1.0)

  def test_where_const_fold(self):
    bf = UOp.const(dtypes.bool, False)
    c1 = UOp.const(dtypes.float, 1.0)
    c2 = UOp.const(dtypes.float, 2.0)
    out = UOp(Ops.WHERE, dtypes.float, (bf, c1, c2))
    uops = to_uops_list([out])
    self.assertEqual(len(uops), 2)  # +1 for SINK
    out = uops[-2]
    self.assertEqual(out.op, Ops.CONST)
    self.assertEqual(out.arg, 2.0)

  def test_const_cast(self):
    bf = UOp.const(dtypes.bool, False)
    out = UOp(Ops.CAST, dtypes.int, (bf,))
    uops = to_uops_list([out])
    self.assertEqual(len(uops), 2)  # +1 for SINK
    out = uops[-2]
    self.assertEqual(out.op, Ops.CONST)
    self.assertEqual(out.arg, 0)

  def test_const_bitcast(self):
    bf = UOp.const(dtypes.float, 1.0)
    out = UOp(Ops.BITCAST, dtypes.uint32, (bf,))
    uops = to_uops_list([out])
    self.assertEqual(len(uops), 2)  # +1 for SINK
    out = uops[-2]
    self.assertEqual(out.op, Ops.CONST)
    self.assertEqual(out.arg, 0x3F800000)

  @unittest.expectedFailure
  def test_const_shape_change_bitcast(self):
    bf = UOp.const(dtypes.uint8, 0x3F)
    out = UOp(Ops.BITCAST, dtypes.half, (bf,))
    uops = to_uops_list([out])
    self.assertEqual(len(uops), 2)  # +1 for SINK

  @unittest.skip("this test isn't valid uops")
  def test_noop_vectorize_fold(self):
    d0 = UOp(Ops.PARAM, dtypes.float.ptr(), arg=0)
    idx = UOp.const(dtypes.int, 0)
    ld = UOp(Ops.LOAD, dtypes.float.vec(2), (d0, idx))
    vec = UOp(Ops.VECTORIZE, dtypes.float.vec(2), (ld,))
    x = UOp(Ops.GEP, dtypes.float, (vec, ), arg=0)
    alu = UOp(Ops.SQRT, dtypes.float, (x, ))
    out = UOp(Ops.STORE, dtypes.void, (d0, idx, alu))
    uops = to_uops_list([out])
    self.assertEqual(len([x for x in uops if x.op is Ops.VECTORIZE]), 0)

  @unittest.skip("this test isn't valid uops")
  def test_gep_vec_fold(self):
    d0 = UOp(Ops.PARAM, dtypes.float.ptr(), (), 0)
    d1 = UOp(Ops.PARAM, dtypes.float.ptr(), (), 1)
    d2 = UOp(Ops.PARAM, dtypes.float.ptr(), (), 2)
    idx = UOp.const(dtypes.int, 0)
    def _test_vec(geps, count=4):
      vec = UOp(Ops.VECTORIZE, dtypes.float.vec(count), geps)
      out = UOp(Ops.STORE, dtypes.void, (d0.index(idx), vec))
      uops = to_uops_list([out])
      if DEBUG >= 4:
        from tinygrad import Device
        print(Device[Device.DEFAULT].renderer.render(uops))
      return uops[-2].src[-1]  # -2 to skip SINK

    # possible
    val = UOp(Ops.LOAD, dtypes.float.vec(4), (d1.index(idx),))
    xyzw = tuple(UOp(Ops.GEP, dtypes.float, (val,), (i,)) for i in range(4))
    self.assertIs(_test_vec(xyzw).op, Ops.LOAD)

    # unaligned
    val = UOp(Ops.LOAD, dtypes.float.vec(4), (d1.index(idx),))
    wzyx = tuple(UOp(Ops.GEP, dtypes.float, (val,), (i,)) for i in reversed(range(4)))
    self.assertIs(_test_vec(wzyx).op, Ops.VECTORIZE)

    # different_size
    val = UOp(Ops.LOAD, dtypes.float.vec(2), (d1.index(idx),))
    xy = tuple(UOp(Ops.GEP, dtypes.float, (val, ), (i,)) for i in range(2))
    self.assertIs(_test_vec(xy+xy).op, Ops.VECTORIZE)
    val = UOp(Ops.LOAD, dtypes.float.vec(4), (d1.index(idx),))
    xy = tuple(UOp(Ops.GEP, dtypes.float, (val, ), (i,)) for i in range(2))
    self.assertIs(_test_vec(xy, count=2).op, Ops.VECTORIZE)

    # different vals
    val1 = UOp(Ops.LOAD, dtypes.float.vec(2), (d1.index(idx),))
    val2 = UOp(Ops.LOAD, dtypes.float.vec(2), (d2.index(idx),))
    xy1 = tuple(UOp(Ops.GEP, dtypes.float, (val1, ), (i,)) for i in range(2))
    xy2 = tuple(UOp(Ops.GEP, dtypes.float, (val2, ), (i,)) for i in range(2))
    self.assertIs(_test_vec(xy1+xy2).op, Ops.VECTORIZE)

  def test_gep_vec_const_fold(self):
    for vec_size in [2, 4, 8]:
      consts = [UOp.const(dtypes.float, float(i)) for i in range(vec_size)]
      vec = UOp(Ops.VECTORIZE, dtypes.float.vec(vec_size), tuple(consts))
      with Context(SPEC=0):
        uops = to_uops_list([UOp(Ops.GEP, dtypes.float, (vec,), (i,)) for i in range(vec_size)])
        for uop, const in zip(uops, consts):
          self.assertEqual(uop, const)

  @unittest.skip("no longer testable standalone")
  def test_wmma_vectorize_fold(self):
    for i in [2, 4, 8]:
      vec = UOp(Ops.VECTORIZE, dtypes.half.vec(i), tuple(UOp.const(dtypes.half, 0.0) for _ in range(i)))
      var = UOp(Ops.DEFINE_VAR, dtypes.half.vec(i))
      acc = UOp.variable('acc', 0, 1, dtypes.half.vec(i))
      wmma = UOp(Ops.WMMA, dtypes.half.vec(i), (vec, var, acc))
      uops = to_uops_list([wmma])
      self.assertEqual(uops[0], acc)
      self.assertEqual(len(uops), 2)  # +1 for SINK

    for i in [2, 4, 8]:
      var = UOp(Ops.DEFINE_VAR, dtypes.half.vec(i))
      vec = UOp(Ops.VECTORIZE, dtypes.half.vec(i), tuple(UOp.const(dtypes.half, 0.0) for _ in range(i)))
      acc = UOp.variable('acc', 0, 1, dtypes.half.vec(i))
      wmma = UOp(Ops.WMMA, dtypes.half.vec(i), (var, vec, acc))
      uops = to_uops_list([wmma])
      self.assertEqual(uops[0], acc)
      self.assertEqual(len(uops), 2)  # +1 for SINK

  @unittest.skip("wmma is wrong here, it needs an arg")
  def test_wmma_vectorize_no_fold(self):
    for i in [4, 8]:
      vec = UOp(Ops.VECTORIZE, dtypes.half.vec(i),
                tuple(UOp.const(dtypes.half, 0.0) for _ in range(i//2)) +
                tuple(UOp(Ops.DEFINE_VAR, dtypes.half, arg=(f'tmp{j}', UOp.const(dtypes.half, 0), UOp.const(dtypes.half, 1))) for j in range(i//2)))
      var = UOp(Ops.DEFINE_VAR, dtypes.half.vec(i), arg=(f'tmp{i}', UOp.const(dtypes.half, 0), UOp.const(dtypes.half, 1)))
      acc = UOp(Ops.DEFINE_VAR, dtypes.half.vec(i), arg=('acc', UOp.const(dtypes.half, 0), UOp.const(dtypes.half, 1)))
      wmma = UOp(Ops.WMMA, dtypes.half.vec(i), (vec, var, acc))
      uops = to_uops_list([wmma])
      self.assertEqual(uops[-2], wmma)  # -2 to skip SINK

    for i in [4, 8]:
      var = UOp(Ops.DEFINE_VAR, dtypes.half.vec(i), arg=(f'tmp{i}', UOp.const(dtypes.half, 0), UOp.const(dtypes.half, 1)))
      vec = UOp(Ops.VECTORIZE, dtypes.half.vec(i),
                tuple(UOp.const(dtypes.half, 0.0) for _ in range(i//2)) +
                tuple(UOp(Ops.DEFINE_VAR, dtypes.half, arg=(f'tmp{j}', UOp.const(dtypes.half, 0), UOp.const(dtypes.half, 1))) for j in range(i//2)))
      acc = UOp(Ops.DEFINE_VAR, dtypes.half.vec(i), arg=('acc', UOp.const(dtypes.half, 0), UOp.const(dtypes.half, 1)))
      wmma = UOp(Ops.WMMA, dtypes.half.vec(i), (var, vec, acc))
      uops = to_uops_list([wmma])
      self.assertEqual(uops[-2], wmma)  # -2 to skip SINK

    for i in [2, 4, 8]:
      vec = UOp(Ops.VECTORIZE, dtypes.half.vec(i),
                tuple(UOp.const(dtypes.half, 1.0 if j == 0 else 0.0) for j in range(i)))
      var = UOp(Ops.DEFINE_VAR, dtypes.half.vec(i), arg=(f'tmp{i}', UOp.const(dtypes.half, 0), UOp.const(dtypes.half, 1)))
      acc = UOp(Ops.DEFINE_VAR, dtypes.half.vec(i), arg=('acc', UOp.const(dtypes.half, 0), UOp.const(dtypes.half, 1)))
      wmma = UOp(Ops.WMMA, dtypes.half.vec(i), (vec, var, acc))
      uops = to_uops_list([wmma])
      self.assertEqual(uops[-2], wmma)  # -2 to skip SINK

    for i in [2, 4, 8]:
      var = UOp(Ops.DEFINE_VAR, dtypes.half.vec(i), arg=(f'tmp{i}', UOp.const(dtypes.half, 0), UOp.const(dtypes.half, 1)))
      vec = UOp(Ops.VECTORIZE, dtypes.half.vec(i),
                tuple(UOp.const(dtypes.half, 1.0 if j == 0 else 0.0) for j in range(i)))
      acc = UOp(Ops.DEFINE_VAR, dtypes.half.vec(i), arg=('acc', UOp.const(dtypes.half, 0), UOp.const(dtypes.half, 1)))
      wmma = UOp(Ops.WMMA, dtypes.half.vec(i), (var, vec, acc))
      uops = to_uops_list([wmma])
      self.assertEqual(uops[-2], wmma)  # -2 to skip SINK

  def test_cast_alu_fold(self):
    d0 = UOp(Ops.PARAM, dtypes.bool.ptr(), arg=0)
    d1 = UOp(Ops.PARAM, dtypes.int.ptr(), arg=1)
    idx = UOp.const(dtypes.int, 0)
    ld = d1.index(idx)
    alu = (ld<1).cast(dtypes.bool)
    out = UOp(Ops.STORE, dtypes.void, (d0.index(idx), alu))
    uops = to_uops_list([out])
    self.assertEqual(len([x for x in uops if x.op is Ops.CAST]), 0)

  def test_double_cast_fold(self):
    d0 = UOp(Ops.PARAM, dtypes.float.ptr(), arg=0)
    d1 = UOp(Ops.PARAM, dtypes.int.ptr(), arg=1)
    idx = UOp.const(dtypes.int, 0)
    ld = d1.index(idx)
    alu = ld.cast(dtypes.float).cast(dtypes.float)
    out = UOp(Ops.STORE, dtypes.void, (d0.index(idx), alu))
    uops = to_uops_list([out])
    self.assertEqual(len([x for x in uops if x.op is Ops.CAST]), 1)

  def test_depth_2_const_fold(self):
    v = UOp.variable("tmp", 0, 1, dtypes.int)
    c2 = UOp.const(dtypes.int, 2)
    c4 = UOp.const(dtypes.int, 4)
    vc = UOp(Ops.ADD, dtypes.int, (v, c2))
    out = UOp(Ops.ADD, dtypes.int, (vc, c4))
    uops = to_uops_list([out])
    self.assertEqual(len(uops), 4)  # +1 for SINK
    out = uops[-2]  # -2 to skip SINK
    self.assertEqual(out.op, Ops.ADD)
    self.assertEqual(out.src[1].op, Ops.CONST)
    self.assertEqual(out.src[1].arg, 6)

  def test_bitcast_to_same_dtype_fold(self):
    for dt in dtypes.ints + dtypes.floats + (dtypes.bool,):
      d0 = UOp(Ops.PARAM, dt.ptr(), arg=0)
      v = d0.index(UOp.const(dtypes.int, 0))
      uops = to_uops_list([v.bitcast(dt)])
      self.assertEqual(len([x for x in uops if x.op is Ops.BITCAST]), 0, f"dtype = {dt}")

  def test_sub_with_cast_folds(self):
    a = Variable("a", 0, 5)
    uops = to_uops_list([a.cast(dtypes.int)+(-a).cast(dtypes.int)])
    assert uops[0] == UOp.const(dtypes.int, 0)
    assert uops[-1].op == Ops.SINK

  def test_where_on_gated_load_fold(self):
    ridx0 = UOp.range(100, 0)
    d0 = UOp(Ops.PARAM, dtypes.long.ptr(), (), 0)
    ld = d0.index(ridx0.valid(ridx0<50))
    w = (ridx0<50).where(ld, 5)
    uops = to_uops_list([w])
    for u in uops:
      assert u.op is not Ops.WHERE
      if u.op is Ops.LOAD: assert u.src[1].arg==5

  def test_where_on_gated_load_folds_swapped_branches(self):
    ridx0 = UOp.range(100, 0)
    d0 = UOp(Ops.PARAM, dtypes.long.ptr(), (), 0)
    ld = d0.index(ridx0.valid((ridx0<50).logical_not()))
    w = (ridx0<50).where(5, ld)
    uops = to_uops_list([w])
    for u in uops:
      assert u.op is not Ops.WHERE
      if u.op is Ops.LOAD: assert u.src[1].arg==5

  def test_where_on_gated_load_with_cast(self):
    ridx0 = UOp.range(100, 0)
    d0 = UOp(Ops.PARAM, dtypes.int.ptr(), (), 0)
    gate_idx = ridx0.valid((ridx0<50))
    ld = d0.index(gate_idx).cast(dtypes.float)
    w = (ridx0<50).where(ld, 5.0)
    uops = to_uops_list([w])
    for u in uops:
      assert u.op is not Ops.WHERE
      if u.op is Ops.LOAD: assert u.src[1].arg == 5

  def test_where_in_store_becomes_gate(self):
    ridx0 = UOp.range(100, 0)
    d0 = UOp(Ops.PARAM, dtypes.long.ptr(), (), 0)
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
    c0 = UOp(Ops.PARAM, dtypes.uchar.ptr(128000), arg=0, src=())
    c1 = UOp.range(UOp.const(dtypes.index, 512), 1, AxisType.LOOP)
    c2 = UOp.range(UOp.const(dtypes.index, 250), 2, AxisType.LOOP)
    c3 = UOp(Ops.PARAM, dtypes.int.ptr(512), arg=1, src=())
    c4 = c3.index(c1)
    c5 = UOp.range(UOp.const(dtypes.index, 240), 0, AxisType.REDUCE)
    c6 = ((c2*UOp.const(dtypes.index, 240))+c5)
    c7 = UOp(Ops.PARAM, dtypes.uchar.ptr(60000), arg=2, src=())
    c8 = c7.index(c6)
    c9 = ((c4<0).where((c4+60000), c4)!=c6.cast(dtypes.int)).where(0, c8.cast(dtypes.uint).cast(dtypes.uchar)).reduce(c5, arg=Ops.ADD)
    c10 = c0.index(((c1*UOp.const(dtypes.index, 250))+c2)).store(c9).end(c1, c2)
    uops = to_uops_list([c10])
    for u in uops:
      self.assertNotEqual(u.dtype, dtypes.long)

  def test_load_idx_no_math_on_loaded(self):
    # test the (x+y)<c pattern where x has loads - we shouldn't do math on loaded indices
    c0 = UOp(Ops.PARAM, dtypes.uchar.ptr(128000), arg=0, src=())
    c1 = UOp.range(UOp.const(dtypes.index, 512), 1, AxisType.LOOP)
    c2 = UOp.range(UOp.const(dtypes.index, 250), 2, AxisType.LOOP)
    c3 = UOp(Ops.PARAM, dtypes.int.ptr(512), arg=1, src=())
    c4 = c3.index(c1)  # c4 is a load
    c5 = UOp.range(UOp.const(dtypes.index, 240), 0, AxisType.REDUCE)
    c6 = ((c2*UOp.const(dtypes.index, 240))+c5)
    c7 = UOp(Ops.PARAM, dtypes.uchar.ptr(60000), arg=2, src=())
    c8 = c7.index(c6)
    # (loaded + range) < const pattern - loaded value shouldn't be promoted to long
    loaded_idx = c4.cast(dtypes.index)
    comparison = (loaded_idx + c5) < UOp.const(dtypes.index, 60000)
    c9 = comparison.where(c8.cast(dtypes.uint).cast(dtypes.uchar), 0).reduce(c5, arg=Ops.ADD)
    c10 = c0.index(((c1*UOp.const(dtypes.index, 250))+c2)).store(c9).end(c1, c2)
    uops = to_uops_list([c10])
    for u in uops:
      self.assertNotEqual(u.dtype, dtypes.long)

  def test_fold_gated_load(self):
    glbl0 = UOp(Ops.PARAM, dtypes.int.ptr(), (), 0)
    glbl1 = UOp(Ops.PARAM, dtypes.int.ptr(), (), 1)
    glbl2 = UOp(Ops.PARAM, dtypes.int.ptr(), (), 2)
    idx = UOp.const(dtypes.int, 0)
    ld0 = glbl1.index(UOp.invalid())
    ld1 = glbl2.index(idx.valid(UOp.const(dtypes.bool, True)))
    uops = to_uops_list([UOp(Ops.STORE, dtypes.void, (glbl0.index(idx), ld1+ld0))])
    ld0 = uops[-2].src[-1]  # -2 to skip SINK
    # the gate and invalid value are deleted from ld1
    self.assertEqual(ld0, UOp.load(glbl2.index(idx, ptr=True), dtype=dtypes.int))

  def test_fold_gated_load_local(self):
    glbl0 = UOp(Ops.PARAM, dtypes.int.ptr(), (), 0)
    smem = UOp(Ops.DEFINE_LOCAL, dtypes.int.ptr(size=18, addrspace=AddrSpace.LOCAL), (), "temp")
    lidx = UOp(Ops.SPECIAL, dtypes.int, (UOp.const(dtypes.int, 16),), "lidx0")
    st = UOp(Ops.STORE, dtypes.void, (smem.index(lidx, ptr=True), glbl0.index(lidx, ptr=True).load()))
    barrier = UOp(Ops.BARRIER, dtypes.void, (st, ))
    ld0 = smem.after(barrier).index(UOp.invalid())
    ld1 = smem.after(barrier).index((lidx+2).valid(UOp.const(dtypes.bool, True)))
    uops = to_uops_list([UOp(Ops.STORE, dtypes.void, (glbl0.index(lidx), ld1+ld0))])

    ld0 = uops[-2].src[-1]  # -2 to skip SINK
    # the gate and invalid value are deleted from ld1
    self.assertEqual(ld0.src[0], smem.after(barrier).index(lidx+2, ptr=True))

  def test_fold_gated_store(self):
    glbl = UOp(Ops.PARAM, dtypes.int.ptr(), (), 0)
    idx0 = UOp.const(dtypes.int, 0)
    idx1 = UOp.const(dtypes.int, 0)
    val = UOp.const(dtypes.int, 42)
    st0 = glbl.index(UOp.invalid(), ptr=True).store(val)
    st1 = glbl.index(idx0.valid(UOp.const(dtypes.bool, True)), ptr=True).store(val)
    uops = to_uops_list([st0, st1])
    # only the second store happens
    self.assertEqual(len(uops), 6)  # +1 for SINK
    self.assertEqual(uops[-2], glbl.index(idx1, ptr=True).store(val))  # -2 to skip SINK

  @unittest.skip("this is a uop type error")
  def test_asserts_bad_gate(self):
    glbl0 = UOp(Ops.PARAM, dtypes.int.ptr(), (), 0)
    idx = UOp.const(dtypes.int, 0)
    bad_gate = UOp.const(dtypes.int, 1)
    with self.assertRaises(AssertionError): to_uops_list([UOp(Ops.STORE, dtypes.void, (glbl0, idx, UOp.const(dtypes.int, 42), bad_gate))])

  def test_after_end(self):
    r = UOp.range(10, 0)

    c = r + 1
    self.assertIn(r, c.ranges)

    e = UOp.const(dtypes.int, 1).end(r)
    self.assertNotIn(r, e.ranges)

    a = c.after(e)
    self.assertNotIn(r, a.ranges)

@track_rewrites()
def expander_rewrite(sink): return graph_rewrite(sink, sym + expander)

class TestExpander(unittest.TestCase):
  def test_expand_add_broadcast(self):
    e1 = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(4), tuple(x for x in range(4))),), ((1,4),))
    sink = expander_rewrite(e1+3)
    assert sink.op is Ops.UNROLL and len(sink.src[0].arg) == 4
    self.assertTupleEqual(sink.src[0].arg, (3,4,5,6))

  def test_contract_simple(self):
    e1 = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(4), tuple(x for x in range(4))),), ((1,4),))
    con = UOp(Ops.CONTRACT, dtypes.int.vec(4), (e1,), ((1,4),))
    sink = expander_rewrite(con)
    self.assertEqual(sink.op, Ops.VCONST)
    self.assertTupleEqual(sink.arg, (0,1,2,3))

  def test_contract_axis_1(self):
    e1 = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(16), tuple(x for x in range(16))),), ((1,4),(2,4)))
    con = UOp(Ops.CONTRACT, dtypes.int.vec(4), (e1,), ((1,4),))
    sink = expander_rewrite(con)
    assert sink.op is Ops.UNROLL and len(sink.src[0].arg) == 16 and sink.arg == ((2,4),)
    assert sink.src[0].op is Ops.VCONST
    self.assertTupleEqual(sink.src[0].arg[0:4], (0,4,8,12))
    self.assertTupleEqual(sink.src[0].arg[12:], (3,7,11,15))

  def test_contract_axis_2(self):
    e1 = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(16), tuple(x for x in range(16))),), ((1,4),(2,4)))
    con = UOp(Ops.CONTRACT, dtypes.int.vec(4), (e1,), ((2,4),))
    sink = expander_rewrite(con)
    assert sink.op is Ops.UNROLL and len(sink.src[0].arg) == 16 and sink.arg == ((1,4),)
    assert sink.src[0].op is Ops.VCONST
    self.assertTupleEqual(sink.src[0].arg[0:4], (0,1,2,3))
    self.assertTupleEqual(sink.src[0].arg[12:], (12,13,14,15))

  def test_contract_axis_2_big(self):
    e1 = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(16), tuple(x for x in range(16))),), ((1,2),(2,2),(3,2),(4,2)))
    con = UOp(Ops.CONTRACT, dtypes.int.vec(2), (e1,), ((2,2),))
    sink = expander_rewrite(con)
    assert sink.op is Ops.UNROLL and sink.arg == ((1, 2), (3, 2), (4, 2))
    self.assertTupleEqual(sink.src[0].arg[0:2], (0,4))
    self.assertTupleEqual(sink.src[0].arg[12:14], (10,14))

  def test_contract_multi_axis(self):
    e1 = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(16), tuple(x for x in range(16))),), ((1,2),(2,2),(3,2),(4,2)))
    sink = expander_rewrite(UOp(Ops.CONTRACT, dtypes.int.vec(4), (e1,), ((3, 2), (2, 2))))
    assert sink.op is Ops.UNROLL and sink.arg == ((1, 2), (4, 2))
    self.assertTupleEqual(sink.src[0].arg[0:4], (0, 4, 2, 6))
    sink = expander_rewrite(UOp(Ops.CONTRACT, dtypes.int.vec(4), (e1,), ((2, 2), (3, 2))))
    assert sink.op is Ops.UNROLL and sink.arg == ((1, 2), (4, 2))
    self.assertTupleEqual(sink.src[0].arg[0:4], (0, 2, 4, 6))

  def test_contract_mid(self):
    e1 = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(8), tuple(x for x in range(8))),), ((1,2),(2,2),(3,2)))
    con = UOp(Ops.CONTRACT, dtypes.int.vec(2), (e1,), ((2,2),))
    sink = expander_rewrite(con)
    assert sink.op is Ops.UNROLL and sink.arg == ((1,2),(3,2))
    assert sink.src[0].op is Ops.VCONST and len(sink.src[0].arg) == 8
    self.assertTupleEqual(sink.src[0].arg, (0,2,1,3,4,6,5,7))

  def test_contract_no_expand(self):
    e1 = UOp.variable("i", 0, 10, dtype=dtypes.int)
    con = UOp(Ops.CONTRACT, dtypes.int.vec(2), (e1,), ((2,2),))
    sink = expander_rewrite(con)
    assert sink.op is Ops.VECTORIZE and len(sink.src) == 2
    assert sink.src[0] == sink.src[1]

  def test_contract_half_expand(self):
    e1 = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(4), tuple(x for x in range(4))),), ((1,4),))
    con = UOp(Ops.CONTRACT, dtypes.int.vec(8), (e1,), ((1,4), (2,2)))
    sink = expander_rewrite(con)
    assert sink.op is Ops.VCONST and len(sink.arg) == 8
    assert sink.arg[0] == sink.arg[1]
    assert sink.arg[0] != sink.arg[2]
    assert sink.arg[6] == sink.arg[7]

  def test_expand_same_axis(self):
    e1 = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(4), tuple(x for x in range(4))),), ((1,4),))
    e2 = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(4), tuple(4*x for x in range(4))),), ((1,4),))
    sink = expander_rewrite(e1+e2)
    self.assertEqual(sink.op, Ops.UNROLL)
    self.assertEqual(sink.src[0].op, Ops.VCONST)
    self.assertTupleEqual(sink.src[0].arg, (0,5,10,15))

  def test_expand_different_axis(self, flip=False):
    e1 = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(4), tuple(4*x for x in range(4))),), ((1,4),))
    e2 = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(4), tuple(x for x in range(4))),), ((2,4),))
    sink = expander_rewrite((e2+e1) if flip else (e1+e2))
    assert sink.op is Ops.UNROLL and len(sink.src[0].arg) == 16
    assert sink.arg == ((1, 4), (2, 4))
    self.assertTupleEqual(sink.src[0].arg, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))

  def test_expand_different_axis_flip(self): self.test_expand_different_axis(True)

  @unittest.skip("no longer supported")
  def test_reduce_known_axis(self):
    e1 = UOp(Ops.UNROLL, dtypes.int, tuple(UOp.const(dtypes.int, x) for x in range(4)), ((1,4),))
    sink = UOp(Ops.REDUCE, dtypes.int, (3*e1,e1), Ops.ADD)
    sink = expander_rewrite(sink)
    assert sink.op is Ops.CONST
    self.assertEqual(sink.arg, 3*(0+1+2+3))

  @unittest.skip("no longer supported")
  def test_reduce_const(self):
    e1 = UOp(Ops.UNROLL, dtypes.int, tuple(UOp.const(dtypes.int, x) for x in range(4)), ((1,4),))
    sink = UOp(Ops.REDUCE, dtypes.int, (UOp.const(dtypes.int, 3), e1), Ops.ADD)
    sink = expander_rewrite(sink)
    assert sink.op is Ops.CONST
    self.assertEqual(sink.arg, 3*4)

  @unittest.skip("no longer supported")
  def test_double_expand(self):
    e1 = UOp(Ops.UNROLL, dtypes.int, tuple(UOp.const(dtypes.int, x) for x in range(4)), ((2,4),))
    e2 = UOp(Ops.UNROLL, dtypes.int, tuple(UOp.const(dtypes.int, 4+x) for x in range(4)), ((2,4),))
    e = UOp(Ops.UNROLL, dtypes.int, (e1, e2), ((1,2),))
    sink = expander_rewrite(e)
    assert sink.op is Ops.UNROLL and len(sink.src) == 8
    assert sink.arg == ((1, 2), (2, 4))
    self.assertListEqual([x.arg for x in sink.src], [0,1,2,3,4,5,6,7])

  @unittest.skip("no longer supported")
  def test_double_expand_reverse(self):
    e1 = UOp(Ops.UNROLL, dtypes.int, tuple(UOp.const(dtypes.int, x) for x in range(4)), ((1,4),))
    e2 = UOp(Ops.UNROLL, dtypes.int, tuple(UOp.const(dtypes.int, 4+x) for x in range(4)), ((1,4),))
    e = UOp(Ops.UNROLL, dtypes.int, (e1, e2), ((2,2),))
    sink = expander_rewrite(e)
    assert sink.op is Ops.UNROLL and len(sink.src) == 8
    assert sink.arg == ((1, 4), (2, 2))
    self.assertListEqual([x.arg for x in sink.src], [0, 4, 1, 5, 2, 6, 3, 7])

  @unittest.skip("no longer supported")
  def test_double_expand_middle(self):
    e1 = UOp(Ops.UNROLL, dtypes.int, tuple(UOp.const(dtypes.int, x) for x in range(4)), ((1,2),(3,2)))
    e2 = UOp(Ops.UNROLL, dtypes.int, tuple(UOp.const(dtypes.int, 4+x) for x in range(4)), ((1,2),(3,2)))
    e = UOp(Ops.UNROLL, dtypes.int, (e1, e2), ((2,2),))
    sink = expander_rewrite(e)
    assert sink.op is Ops.UNROLL and len(sink.src) == 8
    assert sink.arg == ((1, 2), (2, 2), (3, 2))
    self.assertListEqual([x.arg for x in sink.src], [0, 1, 4, 5, 2, 3, 6, 7])

  # does this need to work?
  @unittest.expectedFailure
  @unittest.skip
  def test_reduce_different_axis(self):
    e1 = UOp(Ops.UNROLL, dtypes.int, tuple(UOp.const(dtypes.int, x) for x in range(4)), ((1,4),))
    e2 = UOp(Ops.UNROLL, dtypes.int, tuple(UOp.const(dtypes.int, x) for x in range(4)), ((2,4),))
    sink = UOp(Ops.REDUCE, dtypes.int, (e1,e2), Ops.ADD)
    sink = expander_rewrite(sink)
    print(sink)

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

class TestLoadStoreFolding(unittest.TestCase):
  def test_gated_load_gep_preserves_alt(self):
    """Test that LOAD(GEP, alt) preserves alt value after rewrite"""
    from tinygrad.codegen.late.devectorizer import load_store_folding
    buf = UOp(Ops.PARAM, dtypes.float.vec(4).ptr(), (), 0)
    idx = UOp.const(dtypes.int, 0)
    gate = UOp.const(dtypes.bool, True)
    gated_index = buf.index(idx, gate)
    gep = gated_index.gep(0)
    alt = UOp.const(dtypes.float, 42.0)
    gated_load = gep.load(alt)
    self.assertEqual(len(gated_load.src), 2)  # GEP + alt
    result = graph_rewrite(gated_load, load_store_folding, name='test')
    # After rewrite, should still have alt value preserved
    self.assertEqual(result.op, Ops.GEP)
    inner_load = result.src[0]
    self.assertEqual(inner_load.op, Ops.LOAD)
    self.assertEqual(len(inner_load.src), 2)  # INDEX + alt

  def test_gated_load_ptrcat_preserves_alt(self):
    """Test that LOAD(PTRCAT, alt) preserves alt value after rewrite"""
    from tinygrad.codegen.late.devectorizer import load_store_folding
    buf1 = UOp(Ops.PARAM, dtypes.float.ptr(), (), 0)
    buf2 = UOp(Ops.PARAM, dtypes.float.ptr(), (), 1)
    idx = UOp.const(dtypes.int, 0)
    idx1 = buf1.index(idx)
    idx2 = buf2.index(idx)
    ptrcat = UOp(Ops.PTRCAT, dtypes.float.ptr().vec(2), (idx1, idx2))
    alt = UOp.const(dtypes.float.vec(2), 42.0)
    gated_load = ptrcat.load(alt)
    self.assertEqual(len(gated_load.src), 2)  # PTRCAT + alt
    result = graph_rewrite(gated_load, load_store_folding, name='test')
    # After rewrite, should be CAT of LOADs, each preserving alt
    self.assertEqual(result.op, Ops.CAT)
    for inner_load in result.src:
      self.assertEqual(inner_load.op, Ops.LOAD)
      self.assertEqual(len(inner_load.src), 2)  # INDEX + alt
    self.assertEqual(inner_load.src[1].arg, 42.0)  # alt value preserved

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
    bufferize_with_range = UOp(Ops.BUFFERIZE, dtypes.float, (c, r1), arg=BufferizeOpts(device="CPU"))
    self.assertEqual(len(bufferize_with_range.src), 2)  # const + 1 range

    result = graph_rewrite(bufferize_with_range, pm_const_buffer_folding, name='test')
    # BUFFERIZE should be removed, result is const broadcast to shape
    self.assertNotEqual(result.op, Ops.BUFFERIZE)
    const_vals = [u.arg for u in result.toposort() if u.op is Ops.CONST and u.dtype == dtypes.float]
    self.assertIn(42.0, const_vals)

  def test_const_bufferize_with_multiple_ranges(self):
    """Test CONST.BUFFERIZE with multiple ranges is also folded."""
    from tinygrad.schedule.rangeify import pm_const_buffer_folding, BufferizeOpts
    c = UOp.const(dtypes.float, 3.14)
    r1 = UOp.range(3, 0)
    r2 = UOp.range(4, 1)
    bufferize_with_ranges = UOp(Ops.BUFFERIZE, dtypes.float, (c, r1, r2), arg=BufferizeOpts(device="CPU"))
    self.assertEqual(len(bufferize_with_ranges.src), 3)  # const + 2 ranges

    result = graph_rewrite(bufferize_with_ranges, pm_const_buffer_folding, name='test')
    # BUFFERIZE should be removed
    self.assertNotEqual(result.op, Ops.BUFFERIZE)
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

if __name__ == '__main__':
  unittest.main(verbosity=2)
