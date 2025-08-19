from typing import List
import unittest, pytest
from tinygrad import dtypes, Variable
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import DEBUG, Context
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher, track_rewrites, graph_rewrite, GroupOp
from tinygrad.uop.symbolic import sym
from tinygrad.codegen import full_rewrite, full_rewrite_to_sink
from tinygrad.codegen.expander import expander

simple_pm = PatternMatcher([
  (UPat.cvar('x', dtypes.int), lambda x: UOp.const(dtypes.float, 1.0) + UOp.const(dtypes.float, 2.0)),
  (UPat.cvar('x') + UPat.cvar('y'), lambda x,y: UOp.const(dtypes.float, x.arg+y.arg)),
  (UPat.cvar('x') * UPat.cvar('y') * UPat.cvar('z'), lambda x,y,z: UOp.const(dtypes.float, x.arg*y.arg*z.arg)),
  ((UPat.var('x') + UPat.cvar('c1')) + UPat.cvar('c2'), lambda x,c1,c2: x + (c1.arg+c2.arg)),
])

def to_uops_list(u:List[UOp]) -> List[UOp]:
  # we strip the SINK here for legacy reasons
  ret = full_rewrite(UOp.sink(*u))
  assert ret[-1].op is Ops.SINK
  return ret[:-1]

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
    self.assertEqual(len(results), 1)
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
    c1 = UOp(Ops.CONST, dtypes.float, arg=1.0)
    c2 = UOp(Ops.CONST, dtypes.float, arg=2.0)
    out = UOp(Ops.ADD, dtypes.float, (c1, c2))
    uops = to_uops_list([out])
    self.assertEqual(len(uops), 1)
    out = uops[-1]
    self.assertEqual(out.op, Ops.CONST)
    self.assertEqual(out.arg, 3.0)

  def test_where_same_fold(self):
    v = UOp.variable('tmp', 0, 1)
    c0 = UOp(Ops.CONST, dtypes.int, arg=0)
    vc = UOp(Ops.CMPNE, dtypes.bool, (v, c0))
    c1 = UOp(Ops.CONST, dtypes.float, arg=1.0)
    out = UOp(Ops.WHERE, dtypes.float, (vc, c1, c1))
    uops = to_uops_list([out])
    self.assertEqual(len(uops), 1)
    out = uops[-1]
    self.assertEqual(out.op, Ops.CONST)
    self.assertEqual(out.arg, 1.0)

  def test_where_const_fold(self):
    bf = UOp(Ops.CONST, dtypes.bool, arg=False)
    c1 = UOp(Ops.CONST, dtypes.float, arg=1.0)
    c2 = UOp(Ops.CONST, dtypes.float, arg=2.0)
    out = UOp(Ops.WHERE, dtypes.float, (bf, c1, c2))
    uops = to_uops_list([out])
    self.assertEqual(len(uops), 1)
    out = uops[-1]
    self.assertEqual(out.op, Ops.CONST)
    self.assertEqual(out.arg, 2.0)

  def test_const_cast(self):
    bf = UOp(Ops.CONST, dtypes.bool, arg=False)
    out = UOp(Ops.CAST, dtypes.int, (bf,))
    uops = to_uops_list([out])
    self.assertEqual(len(uops), 1)
    out = uops[-1]
    self.assertEqual(out.op, Ops.CONST)
    self.assertEqual(out.arg, 0)

  def test_const_bitcast(self):
    bf = UOp(Ops.CONST, dtypes.float, arg=1.0)
    out = UOp(Ops.BITCAST, dtypes.uint32, (bf,))
    uops = to_uops_list([out])
    self.assertEqual(len(uops), 1)
    out = uops[-1]
    self.assertEqual(out.op, Ops.CONST)
    self.assertEqual(out.arg, 0x3F800000)

  @unittest.expectedFailure
  def test_const_shape_change_bitcast(self):
    bf = UOp(Ops.CONST, dtypes.uint8, arg=0x3F)
    out = UOp(Ops.BITCAST, dtypes.half, (bf,))
    uops = to_uops_list([out])
    self.assertEqual(len(uops), 1)

  @unittest.skip("this test isn't valid uops")
  def test_noop_vectorize_fold(self):
    d0 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0)
    idx = UOp.const(dtypes.int, 0)
    ld = UOp(Ops.LOAD, dtypes.float.vec(2), (d0, idx))
    vec = UOp(Ops.VECTORIZE, dtypes.float.vec(2), (ld,))
    x = UOp(Ops.GEP, dtypes.float, (vec, ), arg=0)
    alu = UOp(Ops.SQRT, dtypes.float, (x, ))
    out = UOp(Ops.STORE, dtypes.void, (d0, idx, alu))
    uops = to_uops_list([out])
    self.assertEqual(len([x for x in uops if x.op is Ops.VECTORIZE]), 0)

  def test_gep_vec_fold(self):
    d0 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), (), 0)
    d1 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), (), 1)
    d2 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), (), 2)
    idx = UOp.const(dtypes.int, 0)
    def _test_vec(geps, count=4):
      vec = UOp(Ops.VECTORIZE, dtypes.float.vec(count), geps)
      out = UOp(Ops.STORE, dtypes.void, (d0.index(idx), vec))
      uops = to_uops_list([out])
      if DEBUG >= 4:
        from tinygrad import Device
        print(Device[Device.DEFAULT].renderer.render(uops))
      return uops[-1].src[-1]

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
      self.assertEqual(len(uops), 1)

    for i in [2, 4, 8]:
      var = UOp(Ops.DEFINE_VAR, dtypes.half.vec(i))
      vec = UOp(Ops.VECTORIZE, dtypes.half.vec(i), tuple(UOp.const(dtypes.half, 0.0) for _ in range(i)))
      acc = UOp.variable('acc', 0, 1, dtypes.half.vec(i))
      wmma = UOp(Ops.WMMA, dtypes.half.vec(i), (var, vec, acc))
      uops = to_uops_list([wmma])
      self.assertEqual(uops[0], acc)
      self.assertEqual(len(uops), 1)

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
      self.assertEqual(uops[-1], wmma)

    for i in [4, 8]:
      var = UOp(Ops.DEFINE_VAR, dtypes.half.vec(i), arg=(f'tmp{i}', UOp.const(dtypes.half, 0), UOp.const(dtypes.half, 1)))
      vec = UOp(Ops.VECTORIZE, dtypes.half.vec(i),
                tuple(UOp.const(dtypes.half, 0.0) for _ in range(i//2)) +
                tuple(UOp(Ops.DEFINE_VAR, dtypes.half, arg=(f'tmp{j}', UOp.const(dtypes.half, 0), UOp.const(dtypes.half, 1))) for j in range(i//2)))
      acc = UOp(Ops.DEFINE_VAR, dtypes.half.vec(i), arg=('acc', UOp.const(dtypes.half, 0), UOp.const(dtypes.half, 1)))
      wmma = UOp(Ops.WMMA, dtypes.half.vec(i), (var, vec, acc))
      uops = to_uops_list([wmma])
      self.assertEqual(uops[-1], wmma)

    for i in [2, 4, 8]:
      vec = UOp(Ops.VECTORIZE, dtypes.half.vec(i),
                tuple(UOp.const(dtypes.half, 1.0 if j == 0 else 0.0) for j in range(i)))
      var = UOp(Ops.DEFINE_VAR, dtypes.half.vec(i), arg=(f'tmp{i}', UOp.const(dtypes.half, 0), UOp.const(dtypes.half, 1)))
      acc = UOp(Ops.DEFINE_VAR, dtypes.half.vec(i), arg=('acc', UOp.const(dtypes.half, 0), UOp.const(dtypes.half, 1)))
      wmma = UOp(Ops.WMMA, dtypes.half.vec(i), (vec, var, acc))
      uops = to_uops_list([wmma])
      self.assertEqual(uops[-1], wmma)

    for i in [2, 4, 8]:
      var = UOp(Ops.DEFINE_VAR, dtypes.half.vec(i), arg=(f'tmp{i}', UOp.const(dtypes.half, 0), UOp.const(dtypes.half, 1)))
      vec = UOp(Ops.VECTORIZE, dtypes.half.vec(i),
                tuple(UOp.const(dtypes.half, 1.0 if j == 0 else 0.0) for j in range(i)))
      acc = UOp(Ops.DEFINE_VAR, dtypes.half.vec(i), arg=('acc', UOp.const(dtypes.half, 0), UOp.const(dtypes.half, 1)))
      wmma = UOp(Ops.WMMA, dtypes.half.vec(i), (var, vec, acc))
      uops = to_uops_list([wmma])
      self.assertEqual(uops[-1], wmma)

  def test_cast_alu_fold(self):
    d0 = UOp(Ops.DEFINE_GLOBAL, dtypes.bool.ptr(), arg=0)
    d1 = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), arg=1)
    idx = UOp.const(dtypes.int, 0)
    ld = UOp(Ops.LOAD, dtypes.int, (d1.index(idx),))
    alu = (ld<1).cast(dtypes.bool)
    out = UOp(Ops.STORE, dtypes.void, (d0.index(idx), alu))
    uops = to_uops_list([out])
    self.assertEqual(len([x for x in uops if x.op is Ops.CAST]), 0)

  def test_double_cast_fold(self):
    d0 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0)
    d1 = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), arg=1)
    idx = UOp.const(dtypes.int, 0)
    ld = UOp(Ops.LOAD, dtypes.int, (d1.index(idx),))
    alu = ld.cast(dtypes.float).cast(dtypes.float)
    out = UOp(Ops.STORE, dtypes.void, (d0.index(idx), alu))
    uops = to_uops_list([out])
    self.assertEqual(len([x for x in uops if x.op is Ops.CAST]), 1)

  def test_depth_2_const_fold(self):
    v = UOp.variable("tmp", 0, 1)
    c2 = UOp(Ops.CONST, dtypes.int, arg=2)
    c4 = UOp(Ops.CONST, dtypes.int, arg=4)
    vc = UOp(Ops.ADD, dtypes.int, (v, c2))
    out = UOp(Ops.ADD, dtypes.int, (vc, c4))
    uops = to_uops_list([out])
    self.assertEqual(len(uops), 3)
    out = uops[-1]
    self.assertEqual(out.op, Ops.ADD)
    self.assertEqual(out.src[1].op, Ops.CONST)
    self.assertEqual(out.src[1].arg, 6)

  def test_bitcast_to_same_dtype_fold(self):
    for dt in dtypes.ints + dtypes.floats + (dtypes.bool,):
      d0 = UOp(Ops.DEFINE_GLOBAL, dt.ptr(), arg=0)
      v = UOp(Ops.LOAD, dt, (d0.index(UOp.const(dtypes.int, 0)),))
      uops = to_uops_list([v.bitcast(dt)])
      self.assertEqual(len([x for x in uops if x.op is Ops.BITCAST]), 0, f"dtype = {dt}")

  def test_in_out_of_bounds_access(self):
    with Context(IGNORE_OOB=0):
      glbl0 = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(16), (), 0)
      ld0 = UOp(Ops.LOAD, dtypes.int, (glbl0.index(UOp.const(dtypes.int, 0)),))
      to_uops_list([ld0])
      ld1 = UOp(Ops.LOAD, dtypes.int, (glbl0.index(UOp.const(dtypes.int, 15)),))
      to_uops_list([ld1])
      ld1 = UOp(Ops.LOAD, dtypes.int, (glbl0.index(UOp.const(dtypes.int, 7)),))
      to_uops_list([ld1])

      ld0 = UOp(Ops.LOAD, dtypes.int, (glbl0.index(UOp.const(dtypes.int, 42)),))
      with self.assertRaises(RuntimeError): to_uops_list([ld0])

  def test_in_out_of_bounds_access_symbolic(self):
    with Context(IGNORE_OOB=0):
      glbl0 = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(16), (), 0)
      ld0 = UOp(Ops.LOAD, dtypes.int, (glbl0.index(Variable("i", 1, 10)),))
      to_uops_list([ld0])
      ld0 = UOp(Ops.LOAD, dtypes.int, (glbl0.index(Variable("i", 0, 15)),))
      to_uops_list([ld0])

      ld0 = UOp(Ops.LOAD, dtypes.int, (glbl0.index(Variable("i", 0, 20)),))
      with self.assertRaises(RuntimeError): to_uops_list([ld0])

  @unittest.skip("outdated")
  def test_in_out_of_bounds_access_gated_store(self):
    with Context(IGNORE_OOB=0):
      glbl0 = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(16), (), 0)
      v = Variable("v", 0, 20)
      st0 = UOp(Ops.STORE, dtypes.void, (glbl0.index(v), UOp.const(dtypes.int, 0), v<16))
      to_uops_list([st0])

      st1 = UOp(Ops.STORE, dtypes.void, (glbl0.index(v), v, v<20))
      with self.assertRaises(RuntimeError): to_uops_list([st1])

  @unittest.skip("outdated")
  def test_in_bounds_access_gated_local(self):
    with Context(IGNORE_OOB=0):
      # Define buffers
      gbuf = UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(400), (), 0)
      sbuf = UOp(Ops.DEFINE_LOCAL, dtypes.uint.ptr(8, addrspace=AddrSpace.LOCAL), (), "temp0")

      # Define indices, valids and barrier
      gidx = UOp(Ops.SPECIAL, dtypes.int, (), ("gidx0", 416))
      lidx = UOp(Ops.SPECIAL, dtypes.int, (), ("lidx0", 10))

      gate = (gidx<400) & (lidx<8)

      local_store = UOp(Ops.STORE, dtypes.void, (sbuf.index(lidx), UOp.const(dtypes.uint, 1), lidx<8))

      barrier = UOp(Ops.BARRIER, dtypes.void, (local_store,))
      if_barrier = UOp(Ops.IF, dtypes.void, (gate, barrier))

      # Load from local memory (after the IF/barrier)
      local_load = UOp(Ops.LOAD, dtypes.uint, (sbuf.index(lidx), if_barrier))

      # Store to global memory
      global_store = UOp(Ops.STORE, dtypes.void, (gbuf.index(gidx), local_load))
      to_uops_list([global_store])

  def test_out_of_bounds_off_by_one_access(self):
    with Context(IGNORE_OOB=0):
      glbl0 = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(16), (), 0)
      ld0 = UOp(Ops.LOAD, dtypes.int, (glbl0.index(UOp.const(dtypes.int, 16)),))
      with self.assertRaises(RuntimeError): to_uops_list([ld0])

  def test_in_out_bounds_access_with_mask(self):
    with Context(IGNORE_OOB=0):
      glbl0 = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(16), (), 0)
      gidx0 = UOp(Ops.SPECIAL, dtype=dtypes.int, arg=("gidx0", 42))
      ld0 = UOp(Ops.LOAD, dtypes.int, (glbl0.index(gidx0, (5<gidx0)&(gidx0<16)),))
      ld1 = UOp(Ops.LOAD, dtypes.int, (glbl0.index(gidx0, gidx0<16),))
      to_uops_list([ld0, ld1])

      ld0 = UOp(Ops.LOAD, dtypes.int, (glbl0.index(gidx0, gidx0<17),))
      with self.assertRaises(RuntimeError): to_uops_list([ld0])

  def test_in_out_of_bounds_access_symbolic_mask(self):
    with Context(IGNORE_OOB=0):
      glbl0 = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(16), (), 0)
      i = Variable("i", 1, 80)
      ld0 = UOp(Ops.LOAD, dtypes.int, (glbl0.index(i, i<10),))
      to_uops_list([ld0])
      ld0 = UOp(Ops.LOAD, dtypes.int, (glbl0.index(i, i<15),))
      to_uops_list([ld0])

      ld0 = UOp(Ops.LOAD, dtypes.int, (glbl0.index(i, i<20),))
      with self.assertRaises(RuntimeError): to_uops_list([ld0])

  def test_in_out_of_bounds_access_index_load(self):
    with Context(IGNORE_OOB=0):
      glbl0 = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(16), (), 0)
      glbl1 = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(64), (), 0)
      gidx0 = UOp(Ops.SPECIAL, dtype=dtypes.int, arg=("gidx0", 42))
      ld0 = UOp(Ops.LOAD, dtypes.int, (glbl0.index(gidx0, gidx0<8),))
      ld1 = UOp(Ops.LOAD, dtypes.int, (glbl1.index(ld0*2, (ld0>=0)&(ld0<32)),))
      to_uops_list([ld1])

      ld1 = UOp(Ops.LOAD, dtypes.int, (glbl1.index(ld0*2, (ld0>=0)&(ld0<64)),))
      with self.assertRaises(RuntimeError): to_uops_list([ld1])

  def test_fold_gated_load(self):
    glbl0 = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), (), 0)
    glbl1 = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), (), 1)
    glbl2 = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), (), 2)
    idx = UOp.const(dtypes.int, 0)
    ld0 = UOp(Ops.LOAD, dtypes.int, (glbl1.index(idx, UOp.const(dtypes.bool, False)),))
    ld1 = UOp(Ops.LOAD, dtypes.int, (glbl2.index(idx, UOp.const(dtypes.bool, True)),))
    uops = to_uops_list([UOp(Ops.STORE, dtypes.void, (glbl0.index(idx), ld1+ld0))])
    ld0 = uops[-1].src[-1]
    # the gate and invalid value are deleted from ld1
    self.assertEqual(ld0, UOp.load(glbl2.index(idx), dtype=dtypes.int))

  def test_fold_gated_load_local(self):
    glbl0 = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), (), 0)
    smem = UOp(Ops.DEFINE_LOCAL, dtypes.int.ptr(size=18, addrspace=AddrSpace.LOCAL), (), "temp")
    lidx = UOp(Ops.SPECIAL, dtypes.int, (), ("lidx0", 16))
    st = UOp(Ops.STORE, dtypes.void, (smem.index(lidx), UOp.load(glbl0.index(lidx), dtype=dtypes.int)))
    barrier = UOp(Ops.BARRIER, dtypes.void, (st, ))
    ld0 = UOp(Ops.LOAD, dtypes.int, (smem.index(lidx+1, UOp.const(dtypes.bool, False)), barrier))
    ld1 = UOp(Ops.LOAD, dtypes.int, (smem.index(lidx+2, UOp.const(dtypes.bool, True)), barrier))
    uops = to_uops_list([UOp(Ops.STORE, dtypes.void, (glbl0.index(lidx), ld1+ld0))])

    ld0 = uops[-1].src[-1]
    # the gate and invalid value are deleted from ld1
    self.assertEqual(ld0.src[0], smem.index(lidx+2))

  def test_fold_gated_store(self):
    glbl = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), (), 0)
    idx0 = UOp.const(dtypes.int, 0)
    idx1 = UOp.const(dtypes.int, 0)
    val = UOp.const(dtypes.int, 42)
    st0 = glbl.index(idx0, UOp.const(dtypes.bool, False)).store(val)
    st1 = glbl.index(idx0, UOp.const(dtypes.bool, True)).store(val)
    uops = to_uops_list([st0, st1])
    # only the second store happens
    self.assertEqual(len(uops), 5)
    self.assertEqual(uops[-1], glbl.index(idx1).store(val))

  @unittest.skip("this is a uop type error")
  def test_asserts_bad_gate(self):
    glbl0 = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), (), 0)
    idx = UOp.const(dtypes.int, 0)
    bad_gate = UOp.const(dtypes.int, 1)
    with self.assertRaises(AssertionError): to_uops_list([UOp(Ops.STORE, dtypes.void, (glbl0, idx, UOp.const(dtypes.int, 42), bad_gate))])

  def test_switched_range_order(self):
    glbl = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), (), 0)
    c2 = UOp.const(dtypes.int, 2)
    cf = UOp.const(dtypes.float, 0.0)
    r1 = UOp(Ops.RANGE, dtypes.int, (c2,), 0)
    r2 = UOp(Ops.RANGE, dtypes.int, (c2,), 1)
    alu = UOp(Ops.MUL, dtypes.int, (r2, r1))
    store = UOp(Ops.STORE, dtypes.void, (glbl.index(alu), cf))
    uops = to_uops_list([store])
    ranges = [x for x in uops if x.op is Ops.RANGE]
    endranges = [x for x in uops if x.op is Ops.ENDRANGE]
    # ranges are closed in the right order
    self.assertEqual(endranges[-1].src[0], ranges[0])

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
    e1 = UOp(Ops.DEFINE_VAR, dtypes.int)
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

class TestIFUOps(unittest.TestCase):
  def test_create_ifs(self):
    gbuf = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), (), 0)
    sbuf = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(size=4, addrspace=AddrSpace.LOCAL), (), "smem")
    valid = UOp(Ops.SPECIAL, dtypes.int, (), ("gidx0", 10))<5
    lidx = UOp(Ops.SPECIAL, dtypes.int, (), ("lidx0", 4))
    gate = valid&(lidx.ne(2))
    idx = UOp.const(dtypes.int, 0)
    st = UOp(Ops.STORE, dtypes.void, (sbuf.index(idx), UOp.const(dtypes.float, 42)))
    barrier = UOp(Ops.BARRIER, dtypes.void, (st,))
    lbuf = UOp(Ops.LOAD, dtypes.float, (sbuf.index(UOp.const(dtypes.int, 0)), barrier))
    store = UOp(Ops.STORE, dtypes.void, (gbuf.index(UOp.const(dtypes.int, 0), gate), lbuf))
    sink = UOp(Ops.SINK, dtypes.void, (store,))
    sink = full_rewrite_to_sink(sink)
    if_uops = [u for u in sink.toposort() if u.op is Ops.IF]
    self.assertEqual(len(if_uops), 1)
    self.assertEqual(if_uops[0].src[0], gate)
    for st in sink.src:
      self.assertEqual(len(st.src), 2)

  def test_expand_ifs_one_gate(self):
    gbuf = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), (), 0)
    sbuf = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(size=16, addrspace=AddrSpace.LOCAL), (), "smem")
    valid = UOp(Ops.SPECIAL, dtypes.int, (), ("gidx0", 4))<1
    lidx = UOp(Ops.SPECIAL, dtypes.int, (), ("lidx0", 16))
    gate = valid&(lidx.ne(2))
    st = UOp(Ops.STORE, dtypes.void, (sbuf, lidx, UOp.const(dtypes.float, 42)))
    barrier = UOp(Ops.BARRIER, dtypes.void, (st,))
    lbufs = [UOp(Ops.LOAD, dtypes.float, (sbuf.index(UOp.const(dtypes.int, i)), barrier)) for i in range(4)]
    stores = [UOp(Ops.STORE, dtypes.void, (gbuf.index(UOp.const(dtypes.int, i), gate), lbufs[i])) for i in range(4)]
    sink = UOp(Ops.SINK, dtypes.void, tuple(stores))
    sink = full_rewrite_to_sink(sink)
    if_uops = [u for u in sink.toposort() if u.op is Ops.IF]
    self.assertEqual(len(if_uops), 1)
    self.assertEqual(if_uops[0].src[0], gate)
    for st in sink.src:
      self.assertEqual(len(st.src), 2)

  # this will be fixed with the merge gated stores bounty
  @unittest.expectedFailure
  def test_expand_ifs_dumb(self):
    buf = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), (), 0)
    valid = UOp(Ops.SPECIAL, dtypes.int, (), ("gidx0", 10))<5
    lidx = UOp(Ops.SPECIAL, dtypes.int, (), ("lidx0", 4))
    gate = valid&(lidx.ne(2))
    stores = [UOp(Ops.STORE, dtypes.void, (buf, UOp.const(dtypes.int, i), UOp.const(dtypes.float, i), gate)) for i in range(4)]
    sink = UOp(Ops.SINK, dtypes.void, tuple(stores))
    sink = full_rewrite_to_sink(sink)
    if_uops = [u for u in sink.toposort() if u.op is Ops.IF]
    self.assertEqual(len(if_uops), 1)
    self.assertEqual(if_uops[0].src[0], gate)
    for st in sink.src:
      self.assertEqual(len(st.src), 2)

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
