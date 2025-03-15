from typing import Optional, Tuple, Any, List
import unittest, math
import numpy as np
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.tensor import Tensor, _to_np_dtype
from tinygrad.helpers import CI, DEBUG, getenv, Context, Timing
from tinygrad.dtype import dtypes, DType
from tinygrad.device import Buffer, Device
from tinygrad.ops import Ops, UOp, UPat, KernelInfo, exec_alu, spec # noqa F401
from tinygrad.renderer import ProgramSpec
from tinygrad.engine.schedule import create_schedule, to_si
from tinygrad.engine.realize import CompiledRunner, lower_schedule_item, get_kernel
from tinygrad.codegen.linearize import linearize_uop
from tinygrad.codegen.uopgraph import full_graph_rewrite, sym
from tinygrad.device import is_dtype_supported

def to_uops_list(u:List[UOp], opts=None, skip_check=False) -> List[UOp]: return linearize_uop(full_graph_rewrite(UOp.sink(*u), opts), skip_check)

def _uops_to_prg(uops_list):
  uops = linearize_uop(full_graph_rewrite(UOp.sink(*uops_list), opts=Device[Device.DEFAULT].renderer))
  src = Device[Device.DEFAULT].renderer.render("test", uops)
  has_local = Device[Device.DEFAULT].renderer.has_local
  return CompiledRunner(ProgramSpec("test", src, Device.DEFAULT, uops=uops,
                                global_size=[1,1,1] if has_local else None, local_size=[1,1,1] if has_local else None))

def uop(uops:List[UOp], uop:Ops, dtype:Optional[DType], src:Tuple[UOp, ...], arg:Any=None) -> UOp:
  uops.append(UOp(uop, dtype, tuple(src), arg))
  return uops[-1]

def _test_single_value(vals, op, dts):
  uops = []
  output_dtype = dtypes.bool if op in (Ops.CMPLT, Ops.CMPNE) else dts[-1]
  buf_store = uop(uops, Ops.DEFINE_GLOBAL, output_dtype.ptr(), (), 0)
  buf_loads = [uop(uops, Ops.DEFINE_GLOBAL, dtype.ptr(), (), i+1) for i,dtype in enumerate(dts)]
  loads = (uop(uops, Ops.LOAD, dtype, [buf_loads[i].index(uop(uops, Ops.CONST, dtypes.int32, (), 0))]) for i, dtype in enumerate(dts))
  alu = uop(uops, op, output_dtype, loads)
  out = uop(uops, Ops.STORE, dtypes.void, (buf_store.index(uop(uops, Ops.CONST, dtypes.int32, (), 0)), alu))
  buf = Buffer(Device.DEFAULT, 1, output_dtype).allocate()
  buf2 = [Buffer(Device.DEFAULT, 1, dtype).allocate().copyin(np.array([a], dtype=_to_np_dtype(dtype)).data) for a,dtype in zip(vals, dts)]
  prg = _uops_to_prg([out])
  prg.exec([buf]+buf2)
  ret = np.empty(1, _to_np_dtype(output_dtype))
  buf.copyout(ret.data)
  return ret[0]

def _test_single_value_const(vals, op, dts):
  uops = []
  output_dtype = dtypes.bool if op in (Ops.CMPLT, Ops.CMPNE) else dts[-1]
  buf_store = uop(uops, Ops.DEFINE_GLOBAL, output_dtype.ptr(), (), 0)
  loads = (uop(uops, Ops.CONST, dtype, [], a) for a,dtype in zip(vals, dts))
  alu = uop(uops, op, output_dtype, loads)
  out = uop(uops, Ops.STORE, dtypes.void, (buf_store.index(uop(uops, Ops.CONST, dtypes.int32, (), 0)), alu))
  buf = Buffer(Device.DEFAULT, 1, output_dtype).allocate()
  prg = _uops_to_prg([out])
  prg.exec([buf])
  ret = np.empty(1, _to_np_dtype(output_dtype))
  buf.copyout(ret.data)
  return ret[0]

def _test_uops_result(output_dtype, uops, res):
  # uops = []
  buf_store = uop(uops, Ops.DEFINE_GLOBAL, output_dtype.ptr(), (), 0)
  # res = output_fn(uops)
  out = uop(uops, Ops.STORE, dtypes.void, (buf_store.index(uop(uops, Ops.CONST, dtypes.int32, (), 0)), res))
  buf = Buffer(Device.DEFAULT, 1, output_dtype).allocate()
  prg = _uops_to_prg([out])
  prg.exec([buf])
  ret = np.empty(1, _to_np_dtype(output_dtype))
  buf.copyout(ret.data)
  return ret[0]

class TestUOps(unittest.TestCase):
  def _equal(self, v1, v2):
    assert isinstance(v2, (float, int, bool))
    if isinstance(v2, float):
      np.testing.assert_allclose(v1, v2, rtol=2e-7)
    else:
      np.testing.assert_equal(v1, v2)

  def _test_uop_fxn(self, op, fxn, dts=(dtypes.float32, )):
    for f in [_test_single_value, _test_single_value_const]:
      for a in [-2.0, 0.0, 1.0]:
        a = dtypes.as_const(a, dts[0])
        self._equal(f([a], op, dts), fxn(a))

  def _test_bop_fxn(self, op, fxn, dts=(dtypes.float32, )*2, no_b_zero=False, no_b_neg=False):
    for f in [_test_single_value, _test_single_value_const]:
      for a in [-2.0, 0.0, 1.0]:
        for b in [-3.0, 1.0] + ([] if no_b_zero else [0.0]):
          a = dtypes.as_const(a, dts[0])
          b = dtypes.as_const(abs(b) if no_b_neg else b, dts[1])
          self._equal(f([a,b], op, dts), fxn(a,b))

  def _test_top_fxn(self, op, fxn, dts=(dtypes.float32, )*3):
    for f in [_test_single_value, _test_single_value_const]:
      for a in [-2.0, 0, 1]:
        for b in [-3.0, 3.0]:
          for c in [-4.0, 4.0]:
            a = dtypes.as_const(a, dts[0])
            b = dtypes.as_const(b, dts[1])
            c = dtypes.as_const(c, dts[2])
            self._equal(f([a,b,c], op, dts), fxn(a,b,c))

class TestFloatUOps(TestUOps):
  @unittest.skipIf(Device.DEFAULT == "CLANG", 'not supported as uop')
  def test_exp2(self): self._test_uop_fxn(Ops.EXP2, lambda a: np.exp2(a))
  @unittest.skipIf(Device.DEFAULT == "CLANG", 'not supported as uop')
  def test_log2(self): self._test_uop_fxn(Ops.LOG2, lambda a: math.log2(a) if a > 0 else float('-inf' if a==0 else 'nan'))
  @unittest.skipIf(Device.DEFAULT == "CLANG", 'not supported as uop')
  def test_sin(self): self._test_uop_fxn(Ops.SIN, lambda a: math.sin(a))
  def test_recip(self): self._test_uop_fxn(Ops.RECIP, lambda a: 1/a if a != 0 else float('inf'))
  def test_sqrt(self): self._test_uop_fxn(Ops.SQRT, lambda a: math.sqrt(a) if a >= 0 else float('nan'))

  def test_add(self): self._test_bop_fxn(Ops.ADD, lambda a,b: a+b)
  def test_mul(self): self._test_bop_fxn(Ops.MUL, lambda a,b: a*b)
  def test_max(self): self._test_bop_fxn(Ops.MAX, lambda a,b: max(a,b))
  def test_cmplt(self): self._test_bop_fxn(Ops.CMPLT, lambda a,b: a<b)
  def test_cmpne(self): self._test_bop_fxn(Ops.CMPNE, lambda a,b: a!=b)
  # MOD isn't tested on floats

  def test_where(self):
    self._test_top_fxn(Ops.WHERE, lambda a,b,c: b if a!=0 else c, (dtypes.bool, dtypes.float, dtypes.float))

  @unittest.skipUnless(getenv("PYTHON"), "only python supports MULACC")
  def test_mulacc(self):
    self._test_top_fxn(Ops.MULACC, lambda a,b,c: a*b+c, (dtypes.float, dtypes.float, dtypes.float))

class TestNonFloatUOps(TestUOps):
  def test_add_int32(self): self._test_bop_fxn(Ops.ADD, lambda a,b: int(a)+int(b), (dtypes.int32, dtypes.int32))
  def test_mul_int32(self): self._test_bop_fxn(Ops.MUL, lambda a,b: int(a)*int(b), (dtypes.int32, dtypes.int32))
  @unittest.skipUnless(getenv("PTX"), "only ptx uses bitshifts")
  def test_shr_int32(self): self._test_bop_fxn(Ops.SHR, lambda a,b: int(a)>>int(b), (dtypes.int32, dtypes.int32), no_b_neg=True)
  @unittest.skipUnless(getenv("PTX"), "only ptx uses bitshifts")
  def test_shl_int32(self): self._test_bop_fxn(Ops.SHL, lambda a,b: int(a)<<int(b), (dtypes.int32, dtypes.int32), no_b_neg=True)
  def test_div_int32(self):
    self._test_bop_fxn(Ops.IDIV, lambda a,b: int(a/b), (dtypes.int32, dtypes.int32), no_b_zero=True)
  def test_and_int32(self): self._test_bop_fxn(Ops.AND, lambda a,b: int(a)&int(b), (dtypes.int32, dtypes.int32))
  def test_or_int32(self): self._test_bop_fxn(Ops.OR, lambda a,b: int(a)|int(b), (dtypes.int32, dtypes.int32))
  def test_mod_int32(self):
    self._test_bop_fxn(Ops.MOD,
                       lambda a,b: abs(int(a))%abs(int(b))*(1,-1)[a<0], (dtypes.int32, dtypes.int32), no_b_zero=True)
  def test_cmplt_int32(self): self._test_bop_fxn(Ops.CMPLT, lambda a,b: int(a)<int(b), (dtypes.int32, dtypes.int32))
  def test_cmpne_int32(self): self._test_bop_fxn(Ops.CMPNE, lambda a,b: int(a)!=int(b), (dtypes.int32, dtypes.int32))
  @unittest.skipUnless(is_dtype_supported(dtypes.bool), "dtype not supported")
  def test_mul_bool(self): self._test_bop_fxn(Ops.MUL, lambda a,b: bool(a) and bool(b), (dtypes.bool, dtypes.bool))
  @unittest.skipUnless(is_dtype_supported(dtypes.float16), "dtype not supported")
  def test_where_float16(self):
    self._test_top_fxn(Ops.WHERE, lambda a,b,c: b if a!=0 else c, (dtypes.bool, dtypes.float16, dtypes.float16))

class TestBoolUOps(TestUOps):
  def _test_uop_bool_fxn(self, op, fxn):
    for f in [_test_single_value, _test_single_value_const]:
      for a in [False, True]:
        self._equal(f([a], op, (dtypes.bool, )*1), fxn(a))

  def _test_bop_bool_fxn(self, op, fxn):
    for f in [_test_single_value, _test_single_value_const]:
      for a in [False, True]:
        for b in [False, True]:
          self._equal(f([a,b], op, (dtypes.bool, )*2), fxn(a,b))

  def _test_top_bool_fxn(self, op, fxn):
    for f in [_test_single_value, _test_single_value_const]:
      for a in [False, True]:
        for b in [False, True]:
          for c in [False, True]:
            self._equal(f([a,b,c], op, (dtypes.bool, )*3), fxn(a,b,c))

  def test_add_bool(self): self._test_bop_bool_fxn(Ops.ADD, lambda a,b: a or b)
  def test_mul_bool(self): self._test_bop_bool_fxn(Ops.MUL, lambda a,b: a and b)
  def test_xor_bool(self): self._test_bop_bool_fxn(Ops.XOR, lambda a,b: a != b)
  def test_and_bool(self): self._test_bop_bool_fxn(Ops.AND, lambda a,b: a & b)
  def test_or_bool(self): self._test_bop_bool_fxn(Ops.OR, lambda a,b: a | b)
  def test_cmpne_bool(self): self._test_bop_bool_fxn(Ops.CMPNE, lambda a,b: a != b)
  def test_cmplt_bool(self): self._test_bop_bool_fxn(Ops.CMPLT, lambda a,b: a < b)
  def test_where_bool(self): self._test_top_bool_fxn(Ops.WHERE, lambda a,b,c: b if a else c)

class TestExecALU(TestUOps):
  def test_sqrt(self):
    self.assertEqual(exec_alu(Ops.SQRT, dtypes.float, (0.0,)), 0.0)

  def test_div(self):
    self.assertEqual(exec_alu(Ops.IDIV, dtypes.int8, (8, 2)), 4)
    self.assertEqual(exec_alu(Ops.IDIV, dtypes.int8, (7, 3)), 2)
    self.assertEqual(exec_alu(Ops.IDIV, dtypes.int8, (7, -3)), -2)
    self.assertEqual(exec_alu(Ops.IDIV, dtypes.int8, (-50, 6)), -8)

    np.testing.assert_allclose(exec_alu(Ops.MUL, dtypes.float32, (7.0, exec_alu(Ops.RECIP, dtypes.float32, (3.0,)))), 2+(1.0/3.0))
    np.testing.assert_allclose(exec_alu(Ops.MUL, dtypes.float32, (7.0, exec_alu(Ops.RECIP, dtypes.float32, (-3.0,)))), -2-(1.0/3.0))

  def test_recip(self):
    np.testing.assert_allclose(exec_alu(Ops.RECIP, dtypes.float32, (8,)), 1/8)
    np.testing.assert_allclose(exec_alu(Ops.RECIP, dtypes.float32, (7,)), 1/7)
    np.testing.assert_allclose(exec_alu(Ops.RECIP, dtypes.float32, (-3,)), 1/-3)
    np.testing.assert_allclose(exec_alu(Ops.RECIP, dtypes.float32, (-50,)), 1/-50)

    np.testing.assert_allclose(exec_alu(Ops.RECIP, dtypes.float32, ((32+521+3),)), 1/(32+521+3))
    np.testing.assert_allclose(exec_alu(Ops.RECIP, dtypes.float32, ((34**2),)), 1/(34**2))
    np.testing.assert_allclose(exec_alu(Ops.RECIP, dtypes.float32, (10,)), 1/10)

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
    si = create_schedule([t.lazydata])
    assert len(si) == 0

  def test_bitcast_const(self):
    t = Tensor(1, dtype=dtypes.float).bitcast(dtypes.int)
    si = create_schedule([t.lazydata])
    assert len(si) == 1
    ji = lower_schedule_item(si[-1])
    assert any(uop.op is Ops.BITCAST for uop in ji.prg.p.uops), f"{[uop.op for uop in ji.prg.p.uops]} does not contain bitcast"

class TestGatedStoreRewrite(unittest.TestCase):
  def test_tiny_gate_store(self):
    gmem = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), (), 0)
    gidx0 = UOp(Ops.SPECIAL, dtypes.int, (), ('gidx0', 4))
    idx = UOp(Ops.INDEX, dtypes.float.ptr(), (gmem, gidx0 * UOp.const(dtypes.int, 2)))
    val = UOp.const(dtypes.float, 42.0)
    gate = gidx0<UOp.const(dtypes.int, 1)
    store = UOp(Ops.STORE, dtypes.void, (idx, val, gate))
    uops = to_uops_list([store])
    if DEBUG >= 4: print(Device[Device.DEFAULT].renderer.render("test", uops))
    if_uop = next(u for u in uops if u.op is Ops.IF)
    endif = next(u for u in uops if u.op is Ops.ENDIF)
    assert endif.src[0] is if_uop
    gated_uops = tuple(uops[uops.index(if_uop)+1:uops.index(endif)])
    self.assertEqual(len(gated_uops), 1)
    self.assertIs(gated_uops[-1].op, Ops.STORE)

  def test_gate_some_stores(self):
    gmem0 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), (), 0)
    gmem1 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), (), 1)
    gidx0 = UOp(Ops.SPECIAL, dtypes.int, (), ('gidx0', 4))
    idx = gidx0 * UOp.const(dtypes.int, 2)
    idx0 = UOp(Ops.INDEX, dtypes.float.ptr(), (gmem0, idx))
    idx1 = UOp(Ops.INDEX, dtypes.float.ptr(), (gmem1, idx))
    val = UOp.const(dtypes.float, 42.0)
    gate = gidx0<UOp.const(dtypes.int, 1)
    stores = [UOp.store(idx0, val, gate), UOp.store(idx1, val)]
    uops = to_uops_list(stores)
    if DEBUG >= 4: print(Device[Device.DEFAULT].renderer.render("test", uops))
    if_uop = next(u for u in uops if u.op is Ops.IF)
    endif = next(u for u in uops if u.op is Ops.ENDIF)
    assert endif.src[0] is if_uop
    gated_uops = tuple(uops[uops.index(if_uop)+1:uops.index(endif)])
    self.assertEqual(len(gated_uops), 1)
    self.assertIs(gated_uops[-1].op, Ops.STORE)

  # scaled down version of TestLinearizerDumb.test_unmerged_ifs
  def test_merge_ifs_alt(self):
    gmem0 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), (), 0)
    gmem1 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), (), 1)
    gidx0 = UOp(Ops.SPECIAL, dtypes.int, (), ('gidx0', 4))
    idx = gidx0*UOp.const(dtypes.int, 2)
    idx0 = UOp(Ops.INDEX, dtypes.float.ptr(), (gmem0, idx))
    idx1 = UOp(Ops.INDEX, dtypes.float.ptr(), (gmem1, idx))
    val = UOp.const(dtypes.float, 42.0)
    gate = gidx0<UOp.const(dtypes.int, 1)
    stores = [UOp.store(idx0, val, gate), UOp.store(idx1, val, gate)]
    uops = to_uops_list(stores)
    if DEBUG >= 4: print(Device[Device.DEFAULT].renderer.render("test", uops))
    ifs = [u for u in uops if u.op is Ops.IF]
    endifs = [u for u in uops if u.op is Ops.ENDIF]
    self.assertEqual(len(ifs), 1)
    self.assertEqual(len(endifs), 1)
    gated_uops = tuple(uops[uops.index(ifs[0])+1:uops.index(endifs[0])])
    self.assertEqual(len(gated_uops), 2)
    for x in gated_uops: self.assertIs(x.op, Ops.STORE)

class TestLocalAccess(unittest.TestCase):
  # NOTE: this is failing on METAL CI, no idea why. Works locally.
  @unittest.skipIf(Device.DEFAULT == "METAL" and CI, "failing only in CI")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared memory")
  def test_local_basic(self):
    uops = []
    smem = uop(uops, Ops.DEFINE_LOCAL, dtypes.float32.ptr(local=True), (), ('smem', 16))
    st = uop(uops, Ops.STORE, dtypes.void, (smem.index(uop(uops, Ops.CONST, dtypes.int32, (), 0)), uop(uops, Ops.CONST, dtypes.float32, (), 42.0)))
    barr = uop(uops, Ops.BARRIER, dtypes.void, (st,))
    sres = uop(uops, Ops.LOAD, dtypes.float32, (smem.index(uop(uops, Ops.CONST, dtypes.int32, (), 0)), barr))
    self.assertEqual(_test_uops_result(dtypes.float32, uops, sres), 42)

  # NOTE: webgpu specific, since only webgpu performs bitpacking for uchar
  @unittest.skipUnless(Device.DEFAULT == "WEBGPU", "Test local access with packed data type")
  def test_local_packed(self):
    uops = []
    smem = uop(uops, Ops.DEFINE_LOCAL, dtypes.uint8.ptr(local=True), (), ('smem', 16))
    st = uop(uops, Ops.STORE, dtypes.void, (smem.index(uop(uops, Ops.CONST, dtypes.int32, (), 0)), uop(uops, Ops.CONST, dtypes.uint8, (), 42)))
    barr = uop(uops, Ops.BARRIER, dtypes.void, (st,))
    sres = uop(uops, Ops.LOAD, dtypes.uint8, (smem.index(uop(uops, Ops.CONST, dtypes.int32, (), 0)), barr))
    self.assertEqual(_test_uops_result(dtypes.uint8, uops, sres), 42)

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared memory")
  def test_local_indirect(self):
    uops = []
    smem = uop(uops, Ops.DEFINE_LOCAL, dtypes.int32.ptr(local=True), (), ('smem', 16))
    st1 = uop(uops, Ops.STORE, dtypes.void, (smem.index(uop(uops, Ops.CONST, dtypes.int32, (), 1)), uop(uops, Ops.CONST, dtypes.int32, (), 2)))
    st2 = uop(uops, Ops.STORE, dtypes.void, (smem.index(uop(uops, Ops.CONST, dtypes.int32, (), 2)), uop(uops, Ops.CONST, dtypes.int32, (), 42)))
    barr = uop(uops, Ops.BARRIER, dtypes.void, (st1,st2))
    ofs = uop(uops, Ops.LOAD, dtypes.int32, (smem.index(uop(uops, Ops.CONST, dtypes.int32, (), 1)), barr))
    sres = uop(uops, Ops.LOAD, dtypes.int32, (smem.index(ofs),))
    self.assertEqual(_test_uops_result(dtypes.int32, uops, sres), 42)

@unittest.skipUnless(getenv("PTX"), "This only tests assembly backends")
class TestAssembly(unittest.TestCase):
  def test_bitshift_left(self):
    g1 = UOp(Ops.DEFINE_GLOBAL, dtypes.int32.ptr(), (), 0)
    c1 = UOp(Ops.CONST, dtypes.int, (), 2)
    c2 = UOp(Ops.CONST, dtypes.int, (), 3)
    l1 = UOp(Ops.LOAD, dtypes.int, (g1.index(c1),))
    a1 = UOp(Ops.MUL, dtypes.int, (l1, c1))
    a2 = UOp(Ops.MUL, dtypes.int, (l1, c2))
    uops = to_uops_list([a1,a2], opts=Device[Device.DEFAULT].renderer)
    Device[Device.DEFAULT].renderer.render("test", uops)
    ops = [x.op for x in uops]
    self.assertIn(Ops.SHL, ops)
    self.assertIn(Ops.MUL, ops)

  def test_bitshift_right(self):
    g1 = UOp(Ops.DEFINE_GLOBAL, dtypes.int32.ptr(), (), 0)
    c1 = UOp(Ops.CONST, dtypes.int, (), 2)
    c2 = UOp(Ops.CONST, dtypes.int, (), 3)
    l1 = UOp(Ops.LOAD, dtypes.int, (g1.index(c1),))
    a1 = UOp(Ops.IDIV, dtypes.int, (l1, c1))
    a2 = UOp(Ops.IDIV, dtypes.int, (l1, c2))
    uops = to_uops_list([a1,a2], opts=Device[Device.DEFAULT].renderer)
    Device[Device.DEFAULT].renderer.render("test", uops)
    ops = [x.op for x in uops]
    self.assertIn(Ops.SHR, ops)
    self.assertIn(Ops.IDIV, ops)

class TestUOpMethod(unittest.TestCase):
  @unittest.skip("uops lt no longer ordered")
  def test_compare_alu_same_src_different_arg(self):
    a = UOp(Ops.CONST, dtypes.float, (), 2.0)
    b = UOp(Ops.CONST, dtypes.float, (), 3.0)

    add = UOp(Ops.ADD, dtypes.float, (a, b))
    mul = UOp(Ops.MUL, dtypes.float, (a, b))
    assert (add < mul) or (mul < add), "add and mul with same src should have an order"

  def test_uop_variables(self):
    a = UOp.variable("a", 1, 10)
    uop_var = UOp.const(dtypes.int, a)
    st_var = UOp(Ops.LOAD, dtypes.float, (UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), (), 0),
                                           ShapeTracker.from_shape((2, a)).to_uop()))
    ast_vars = (st_var+uop_var).variables()
    self.assertEqual(len(ast_vars), 1)
    self.assertEqual(ast_vars[0], a)

  def test_const_factor(self):
    gidx0 = UOp(Ops.SPECIAL, dtypes.int, (), ('gidx0', 8))
    self.assertEqual(UOp(Ops.CONST, dtypes.int, (), 17).const_factor(), 17)
    self.assertEqual(gidx0.const_factor(), 1)
    self.assertEqual((gidx0*3).const_factor(), 3)
    self.assertEqual((gidx0*3+6).const_factor(), 3)
    self.assertEqual((gidx0*3+1).const_factor(), 1)

  def test_replace(self):
    x = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), (), 0)
    self.assertIs(x.replace(arg=None).arg, None)
    with self.assertRaises(AssertionError): x.replace(field="a")

  def test_device(self):
    x = UOp(Ops.VIEW, dtypes.int, (UOp.new_buffer(Device.DEFAULT, 1, dtypes.int), UOp.const(dtypes.int, 1)), ShapeTracker.from_shape(()))
    self.assertEqual(x.device, Device.DEFAULT)
    # NOTE: CONST doesn't have device
    buffer, const = x.src
    self.assertEqual(buffer.device, Device.DEFAULT)
    self.assertEqual(const._device, None)
    with self.assertRaises(AssertionError): const.device

  def test_const_arg(self):
    var = UOp.variable("a", 1, 10)
    with self.assertRaises(AssertionError): UOp.const(dtypes.int, var).const_arg
    const = UOp.const(dtypes.int, 1)
    self.assertEqual(const.const_arg, 1)
    tensor_const = UOp(Ops.VIEW, dtypes.int, (UOp.new_buffer(Device.DEFAULT, 1, dtypes.int), const), ShapeTracker.from_shape(()))
    self.assertEqual(tensor_const.const_arg, 1)

class TestUOpStr(unittest.TestCase):
  def test_uop_str(self):
    a = UOp(Ops.CONST, dtypes.float, (), 2.0) + UOp(Ops.CONST, dtypes.float, (), 3.0)
    for _ in range(20): a = a + a
    assert len(str(a)) < 10_000, "exponential string growth"
    assert str(eval(str(a))) == str(a)

    t = Tensor.arange(10)
    t = t + t * Tensor.rand(10)
    # nice big complicated uop
    with Context(NOOPT=1):
      sink = UOp(Ops.SINK, dtypes.void, (get_kernel(Device[Device.DEFAULT].renderer, t.schedule()[-1].ast).linearize().uops[-1],))
    self.assertEqual(sink, eval(str(sink)))

  def test_vectorized_str(self):
    vec = UOp(Ops.VECTORIZE, dtypes.int.vec(4), tuple(UOp.const(dtypes.int, x) for x in range(4)))
    assert str(eval(str(vec))) == str(vec)

@unittest.skip("uop no longer has order like this")
class TestIndexingOrdering(unittest.TestCase):
  # NOTE: these tests skip type_verify since they add dtype to STORE
  @unittest.expectedFailure
  def test_simple_order(self):
    buf = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), (), 0)
    st0 = UOp(Ops.STORE, dtypes.float.vec(4), (buf, UOp.const(dtypes.int, 0), UOp.const(dtypes.float.vec(4), 42)))
    st1 = UOp(Ops.STORE, dtypes.float, (buf, UOp.const(dtypes.int, 4), UOp.const(dtypes.float, 10)))
    uops = to_uops_list([st1, st0], skip_check=True)
    stores = [st for st in uops if st.op is Ops.STORE]
    assert stores[0].src[1] < stores[1].src[1], f"stored at idx {stores[1].src[1].arg} AFTER {stores[0].src[1].arg}"

  @unittest.expectedFailure
  def test_ordering_multi_output(self):
    buf0 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), (), 0)
    buf1 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), (), 1)
    st0_0 = UOp(Ops.STORE, dtypes.float.vec(4), (buf0, UOp.const(dtypes.int, 0), UOp.const(dtypes.float.vec(4), 42)))
    st1_0 = UOp(Ops.STORE, dtypes.float, (buf0, UOp.const(dtypes.int, 4), UOp.const(dtypes.float, 10)))
    st0_1 = UOp(Ops.STORE, dtypes.float.vec(4), (buf1, UOp.const(dtypes.int, 0), UOp.const(dtypes.float.vec(4), 42)))
    st1_1 = UOp(Ops.STORE, dtypes.float, (buf1, UOp.const(dtypes.int, 4), UOp.const(dtypes.float, 10)))
    uops = to_uops_list([st0_0, st1_0, st0_1, st1_1], skip_check=True)
    stores = [st for st in uops if st.op is Ops.STORE]
    print("\n".join(map(str, stores)))
    # buf0 stores come first
    self.assertEqual(stores[0].src[0].arg, stores[1].src[0].arg)
    # buf1 stores come next
    self.assertEqual(stores[2].src[0].arg, stores[3].src[0].arg)
    # both stores are aligned based on idx
    assert stores[0].src[1] < stores[1].src[1], f"stored at idx {stores[1].src[1].arg} AFTER {stores[0].src[1].arg}"
    assert stores[2].src[1] < stores[3].src[1], f"stored at idx {stores[1].src[1].arg} AFTER {stores[0].src[1].arg}"

  def test_simple_order_with_special(self):
    buf = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), (), 0)
    gidx0 = UOp(Ops.SPECIAL, dtypes.int, (), ('gidx0', 4))
    st0 = UOp(Ops.STORE, dtypes.float.vec(4), (buf, gidx0+UOp.const(dtypes.int, 0), UOp.const(dtypes.float.vec(4), 42)))
    st1 = UOp(Ops.STORE, dtypes.float, (buf, UOp.const(dtypes.int, 4), UOp.const(dtypes.float, 10)))
    uops = linearize_uop(UOp.sink(st1, st0), skip_check=True)
    stores = [st for st in uops if st.op is Ops.STORE]
    assert stores[0].src[1] < stores[1].src[1], f"stored at idx {stores[1].src[1].arg} AFTER {stores[0].src[1].arg}"

class TestUPatHelpers(unittest.TestCase):
  def test_location(self):
    self.assertEqual(sym.patterns[-1][0].location[0].replace("\\", "/").split("/")[-1], "uopgraph.py")
    self.assertEqual(to_si.patterns[0][0].location[0].replace("\\", "/").split("/")[-1], "schedule.py")
    self.assertEqual(spec.patterns[0][0].location[0].replace("\\", "/").split("/")[-1], "ops.py")
    with self.assertRaises(AssertionError): # TODO: location UPat files created in test/*?
      test_upat = UPat(Ops.CONST, dtypes.bool)
      self.assertEqual(test_upat.location[0].split("/")[-1], __file__.replace("\\", "/").split("/")[-1])

class TestUopsObject(unittest.TestCase):
  # LOL, running this test breaks all instances of "4"
  """
  @unittest.expectedFailure
  def test_immutable(self):
    const_4 = UOp.const(dtypes.int, 4)
    with self.assertRaises(Exception):
      const_4.arg = 5
  """

  def test_timing(self):
    with Timing("create 10k uops:"): ret = [UOp(Ops.CONST, dtypes.int, arg=10000000+i) for i in range(10000)]
    assert len(ret) == 10000

if __name__ == '__main__':
  unittest.main(verbosity=2)
