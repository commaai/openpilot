from typing import Optional, Any
import unittest, math
import numpy as np
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View # noqa F401
from tinygrad.tensor import Tensor, _to_np_dtype
from tinygrad.helpers import CI, DEBUG, getenv, Timing
from tinygrad.dtype import dtypes, DType, AddrSpace
from tinygrad.device import Buffer, Device
from tinygrad.uop.ops import Ops, UOp, UPat, KernelInfo, exec_alu # noqa F401
from tinygrad.uop.spec import spec
from tinygrad.renderer import ProgramSpec
from tinygrad.engine.realize import CompiledRunner, get_program
from tinygrad.codegen import full_rewrite
from tinygrad.uop.symbolic import sym
from tinygrad.device import is_dtype_supported
from tinygrad.codegen.opt.kernel import Opt, OptOps

def to_uops_list(u:list[UOp], opts=None, skip_check=False) -> list[UOp]: return full_rewrite(UOp.sink(*u), opts)

def _uops_to_prg(uops_list):
  uops = full_rewrite(ast:=UOp.sink(*uops_list), opts=Device[Device.DEFAULT].renderer)
  src = Device[Device.DEFAULT].renderer.render(uops)
  has_local = Device[Device.DEFAULT].renderer.has_local
  return CompiledRunner(ProgramSpec("test", src, Device.DEFAULT, ast, uops=uops,
                                global_size=[1,1,1] if has_local else None, local_size=[1,1,1] if has_local else None))

def uop(uops:list[UOp], uop:Ops, dtype:Optional[DType], src:tuple[UOp, ...], arg:Any=None) -> UOp:
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
  @unittest.skipIf(Device.DEFAULT == "CPU", 'not supported as uop')
  def test_exp2(self): self._test_uop_fxn(Ops.EXP2, lambda a: np.exp2(a))
  @unittest.skipIf(Device.DEFAULT == "CPU", 'not supported as uop')
  def test_log2(self): self._test_uop_fxn(Ops.LOG2, lambda a: math.log2(a) if a > 0 else float('-inf' if a==0 else 'nan'))
  @unittest.skipIf(Device.DEFAULT == "CPU", 'not supported as uop')
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
    si = t.schedule()
    assert len(si) == 0

class TestGatedStoreRewrite(unittest.TestCase):
  def test_tiny_gate_store(self):
    gmem = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), (), 0)
    gidx0 = UOp(Ops.SPECIAL, dtypes.int, (), ('gidx0', 4))
    gate = gidx0<UOp.const(dtypes.int, 1)
    idx = UOp(Ops.INDEX, dtypes.float.ptr(), (gmem, gidx0 * UOp.const(dtypes.int, 2), gate))
    val = UOp.const(dtypes.float, 42.0)
    store = UOp(Ops.STORE, dtypes.void, (idx, val))
    uops = to_uops_list([store])
    if DEBUG >= 4: print(Device[Device.DEFAULT].renderer.render(uops))
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
    idx0 = UOp(Ops.INDEX, dtypes.float.ptr(), (gmem0, idx, gidx0<UOp.const(dtypes.int, 1)))
    idx1 = UOp(Ops.INDEX, dtypes.float.ptr(), (gmem1, idx))
    val = UOp.const(dtypes.float, 42.0)
    stores = [UOp.store(idx0, val), UOp.store(idx1, val)]
    uops = to_uops_list(stores)
    if DEBUG >= 4: print(Device[Device.DEFAULT].renderer.render(uops))
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
    gate = gidx0<UOp.const(dtypes.int, 1)
    idx0 = UOp(Ops.INDEX, dtypes.float.ptr(), (gmem0, idx, gate))
    idx1 = UOp(Ops.INDEX, dtypes.float.ptr(), (gmem1, idx, gate))
    val = UOp.const(dtypes.float, 42.0)
    stores = [UOp.store(idx0, val), UOp.store(idx1, val)]
    uops = to_uops_list(stores)
    if DEBUG >= 4: print(Device[Device.DEFAULT].renderer.render(uops))
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
    smem = uop(uops, Ops.DEFINE_LOCAL, dtypes.float32.ptr(size=16, addrspace=AddrSpace.LOCAL), (), 'smem')
    st = uop(uops, Ops.STORE, dtypes.void, (smem.index(uop(uops, Ops.CONST, dtypes.int32, (), 0)), uop(uops, Ops.CONST, dtypes.float32, (), 42.0)))
    barr = uop(uops, Ops.BARRIER, dtypes.void, (st,))
    sres = uop(uops, Ops.LOAD, dtypes.float32, (smem.index(uop(uops, Ops.CONST, dtypes.int32, (), 0)), barr))
    self.assertEqual(_test_uops_result(dtypes.float32, uops, sres), 42)

  # NOTE: webgpu specific, since only webgpu performs bitpacking
  @unittest.skipUnless(Device.DEFAULT == "WEBGPU", "Test local access with packed data type")
  def test_local_packed(self):
    uops = []
    smem = uop(uops, Ops.DEFINE_LOCAL, dtypes.uint8.ptr(size=16, addrspace=AddrSpace.LOCAL), (), 'smem')
    st = uop(uops, Ops.STORE, dtypes.void, (smem.index(uop(uops, Ops.CONST, dtypes.int32, (), 0)), uop(uops, Ops.CONST, dtypes.uint8, (), 42)))
    barr = uop(uops, Ops.BARRIER, dtypes.void, (st,))
    sres = uop(uops, Ops.LOAD, dtypes.uint8, (smem.index(uop(uops, Ops.CONST, dtypes.int32, (), 0)), barr))
    self.assertEqual(_test_uops_result(dtypes.uint8, uops, sres), 42)

  # NOTE: webgpu specific, since only webgpu performs bitpacking
  @unittest.skipUnless(Device.DEFAULT == "WEBGPU", "Test local memory size for packed data types")
  def test_packed_smem_size(self):
    _dtypes = [dtypes.char, dtypes.uchar, dtypes.short, dtypes.ushort, dtypes.half]
    size = 16
    for dtype in _dtypes:
      temp = UOp(Ops.DEFINE_LOCAL, dtype.ptr(size=size, addrspace=AddrSpace.LOCAL), (), 'smem')
      uops = to_uops_list([temp], opts=Device[Device.DEFAULT].renderer)
      out = Device[Device.DEFAULT].renderer.render(uops)
      # half is supported in wgsl, so it doesn't have to be packed
      corrected_size = size//(4//dtype.itemsize) if dtype != dtypes.half else size
      self.assertIn(f"temp0: array<{Device[Device.DEFAULT].renderer.buf_map(dtype)},{corrected_size}>;", out)

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared memory")
  @unittest.skip("tinygrad doesn't support this behavior")
  def test_local_indirect(self):
    uops = []
    smem = uop(uops, Ops.DEFINE_LOCAL, dtypes.int32.ptr(size=16, addrspace=AddrSpace.LOCAL), (), 'smem')
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
    Device[Device.DEFAULT].renderer.render(uops)
    ops = [x.op for x in uops]
    self.assertIn(Ops.SHL, ops)
    self.assertIn(Ops.MUL, ops)

  def test_division_power_of_two(self):
    for dt in (dtypes.int32, dtypes.uint32):
      g = UOp(Ops.DEFINE_GLOBAL, dt.ptr(), (), 0)
      c = UOp(Ops.CONST, dt, (), 2)
      l = UOp(Ops.LOAD, dt, (g.index(c),))
      a = UOp(Ops.IDIV, dt, (l, c))
      uops = to_uops_list([a], opts=Device[Device.DEFAULT].renderer)
      Device[Device.DEFAULT].renderer.render(uops)
      ops = [x.op for x in uops]
      self.assertIn(Ops.SHR, ops, f"For dtype={dt} divison by power of two did not simplify to shift")
      self.assertNotIn(Ops.IDIV, ops, f"For dtype={dt} divison by power of two did not simplify to shift")

  def test_fast_idiv_and_mod(self):
    g = UOp(Ops.DEFINE_GLOBAL, dtypes.uint32.ptr(), (), 0)
    c = UOp(Ops.CONST, dtypes.uint, (), 3)
    l = UOp(Ops.LOAD, dtypes.uint, (g.index(c),))
    a = UOp(Ops.IDIV, dtypes.uint, (l, c))
    uops = to_uops_list([a], opts=Device[Device.DEFAULT].renderer)
    Device[Device.DEFAULT].renderer.render(uops)
    ops = [x.op for x in uops]
    self.assertIn(Ops.SHR, ops)
    self.assertNotIn(Ops.IDIV, ops)

    b = UOp(Ops.MOD, dtypes.uint, (l, c))
    uops = to_uops_list([b], opts=Device[Device.DEFAULT].renderer)
    Device[Device.DEFAULT].renderer.render(uops)
    ops = [x.op for x in uops]
    self.assertIn(Ops.SHR, ops)
    self.assertNotIn(Ops.MOD, ops)

  @unittest.expectedFailure
  def test_fast_idiv_overflow(self):
    # This will be possible with a slightly different method for fast_idiv
    g = UOp(Ops.DEFINE_GLOBAL, dtypes.uint32.ptr(), (), 0)
    c = UOp(Ops.CONST, dtypes.uint, (), 7)
    l = UOp(Ops.LOAD, dtypes.uint, (g.index(c),))
    a = UOp(Ops.IDIV, dtypes.uint, (l, c))
    uops = to_uops_list([a], opts=Device[Device.DEFAULT].renderer)
    Device[Device.DEFAULT].renderer.render(uops)
    ops = [x.op for x in uops]
    self.assertIn(Ops.SHR, ops)
    self.assertNotIn(Ops.IDIV, ops)

  def test_mulacc_unrolled(self):
    # test that     acc = acc + a0*b0 + a1*b1 + a2*b2 + a3*b3
    # is not        acc = acc + (a0*b0 + a1*b1 + a2*b2 + a3*b3)
    a = Tensor.empty(1024)
    b = Tensor.empty(1024)
    c = (a*b).sum()
    ast = c.schedule()[-1].ast
    opts_to_apply = [Opt(OptOps.UNROLL, 0, 4)]
    ast = ast.replace(arg=KernelInfo(opts_to_apply=tuple(opts_to_apply)))
    program = get_program(ast, Device[Device.DEFAULT].renderer)
    uops = program.uops
    self.assertEqual(len([x.op for x in uops if x.op is Ops.MULACC]), 4)

  def test_use_cmpeq(self):
    g = UOp(Ops.DEFINE_GLOBAL, dtypes.uint32.ptr(), (), 0)
    c = UOp(Ops.CONST, dtypes.uint, (), 7)
    l = UOp(Ops.LOAD, dtypes.uint, (g.index(c),))
    comp = l.ne(c).ne(True)
    uops = to_uops_list([comp], opts=Device[Device.DEFAULT].renderer)
    Device[Device.DEFAULT].renderer.render(uops)
    ops = [x.op for x in uops]
    self.assertIn(Ops.CMPEQ, ops)
    self.assertNotIn(Ops.CMPNE, ops)

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
    uop_var = Tensor(a.bind(1))
    st_var = Tensor.empty((2, 1)).reshape((2, a.bind(1)))
    _, var_vals = (uop_var+st_var).schedule_with_vars()
    self.assertEqual(len(var_vals), 1)
    self.assertEqual(list(var_vals)[0], a)

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

class TestUOpStr(unittest.TestCase):
  def test_uop_str(self):
    a = UOp(Ops.CONST, dtypes.float, (), 2.0) + UOp(Ops.CONST, dtypes.float, (), 3.0)
    for _ in range(20): a = a + a
    assert len(str(a)) < 10_000, "exponential string growth"
    assert str(eval(str(a))) == str(a)

  def test_vectorized_str(self):
    vec = UOp(Ops.VECTORIZE, dtypes.int.vec(4), tuple(UOp.const(dtypes.int, x) for x in range(4)))
    assert str(eval(str(vec))) == str(vec)

  def test_device_arg(self):
    device = UOp(Ops.DEVICE, arg="GPU")
    assert str(eval(str(device))) == str(device)

  def test_reduceop_arg(self):
    sum_uop = Tensor.empty(32, 32).sum().uop
    assert str(eval(str(sum_uop))) == str(sum_uop)

class TestUPatHelpers(unittest.TestCase):
  def test_location(self):
    self.assertEqual(sym.patterns[-1][0].location[0].replace("\\", "/").split("/")[-1], "symbolic.py")
    self.assertEqual(spec.patterns[0][0].location[0].replace("\\", "/").split("/")[-1], "spec.py")
    test_upat = UPat(Ops.CONST, dtypes.bool)
    self.assertEqual(test_upat.location[0].split("/")[-1], __file__.replace("\\", "/").split("/")[-1])
    test_upat_named = test_upat.named("test_name")
    self.assertEqual(test_upat.location[0], test_upat_named.location[0])
    self.assertNotEqual(test_upat.location[1], test_upat_named.location[1])

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


class TestShapeSpec(unittest.TestCase):
  # ** CONST is CONST(VIEW(DEVICE)) -> RESHPAE -> EXPAND

  def test_expanded_const(self):
    a = Tensor(1).uop
    self.assertEqual(a.st, ShapeTracker.from_shape(()))
    a = Tensor.ones((4, 4)).uop
    self.assertEqual(a.st, ShapeTracker.from_shape(()).reshape((1,1)).expand((4,4)))

  # NOTE: CONST ShapeTracker comes from its source
  def test_scalar_const(self):
    a = Tensor(0).uop
    self.assertEqual(a.st, ShapeTracker.from_shape(()))

  def test_scalar_var(self):
    vv = UOp.variable("a", 1, 4).bind(2)
    t = Tensor(vv).uop
    self.assertEqual(t.st, ShapeTracker.from_shape(()))

  # ** ASSIGN is ASSIGN(VIEW(BUFFER), new_val)

  def test_assign_flat(self):
    buffer = Tensor.arange(4).realize()
    a = buffer.assign(Tensor.zeros((4,), dtype=dtypes.int))
    assign_pattern = UPat(Ops.ASSIGN, src=(UPat(Ops.BUFFER), UPat()))
    assert assign_pattern.match(a.uop, {})
    a.realize()
    self.assertEqual(buffer.tolist(), [0, 0, 0, 0])

  def test_assign_permuted(self):
    buffer = Tensor.arange(4).reshape(2, 1, 2).contiguous().realize()
    a = buffer.permute((1, 2, 0)).assign(Tensor.arange(4).reshape(1, 2, 2).contiguous())
    a.realize()
    self.assertEqual(buffer.tolist(), [[[0, 2]], [[1, 3]]])

  def test_assign_reshaped(self):
    buffer = Tensor.ones((4,)).contiguous().realize()
    a = buffer.reshape((2, 2)).assign(Tensor.zeros((2, 2)))
    assign_pattern = UPat(Ops.ASSIGN, src=(UPat(Ops.RESHAPE, src=(UPat(Ops.BUFFER))), UPat()))
    assert assign_pattern.match(a.uop, {})
    a.realize()
    self.assertEqual(buffer.tolist(), [0, 0, 0, 0])

  # setitem is a partial assign
  def test_setitem(self):
    a = Tensor.ones((4,)).contiguous().realize()
    assign = a.shrink(((1, 2),)).assign(Tensor.zeros((1,)))
    # the ASSIGN UOp has size=1
    self.assertEqual(assign.uop.size, 1)
    # the ASSIGN views the buffer with a shrunk st
    self.assertEqual(assign.uop.src[0].st, ShapeTracker.from_shape((4,)).shrink(((1, 2),)))
    # the underlying BUFFER has a size=4
    self.assertEqual(assign.uop.buf_uop.size, 4)
    # NOTE: output shape is different from the BUFFER shape
    self.assertNotEqual(assign.uop.shape, a.uop.shape)
    assign.realize()
    self.assertEqual(a.tolist(), [1, 0, 1, 1])

  def test_buffer_st(self):
    a = UOp.new_buffer(Device.DEFAULT, 10, dtypes.float)
    self.assertEqual(a.st, ShapeTracker.from_shape((10,)))

  def test_ops_st(self):
    # view / mop
    a = Tensor.empty(4, 2, 1).permute((1, 2, 0)).uop
    self.assertEqual(a.st, ShapeTracker.from_shape((4, 2, 1)).permute((1, 2, 0)))
    # alu / reduce
    alu = a*2
    self.assertEqual(alu.st, ShapeTracker.from_shape((2, 1, 4)))
    r = Tensor.empty(4, 4).sum(axis=1)
    self.assertEqual(r.uop.st, ShapeTracker.from_shape((4,)))

  def test_st_wmma_none(self):
    A = UOp(Ops.DEFINE_VAR, dtypes.float.vec(16), arg=('a', UOp.const(dtypes.float, 0), UOp.const(dtypes.float, 1)))
    B = UOp(Ops.DEFINE_VAR, dtypes.float.vec(16), arg=('b', UOp.const(dtypes.float, 0), UOp.const(dtypes.float, 2)))
    C = UOp(Ops.DEFINE_VAR, dtypes.float.vec(16), arg=('c', UOp.const(dtypes.float, 0), UOp.const(dtypes.float, 3)))
    wmma = UOp(Ops.WMMA, dtypes.float.vec(16), (A, B, C))
    assert wmma.st is None

class TestUOpChildren(unittest.TestCase):
  def test_children_exist(self):
    a = UOp.variable("weird_name_234", 0, 10)
    b = a*a
    self.assertEqual(len(a.children), 1)
    self.assertIs(list(a.children)[0](), b)

  def test_children_cleaned_up(self):
    a = UOp.variable("weird_name_235", 0, 10)
    b = a*a
    self.assertEqual(len(a.children), 1)
    del b
    self.assertEqual(len(a.children), 0)

  def test_children_cleaned_up_two(self):
    a = UOp.variable("weird_name_236", 0, 10)
    b = a*a
    c = a*2
    self.assertEqual(len(a.children), 2)
    del b
    self.assertEqual(len(a.children), 1)
    del c
    self.assertEqual(len(a.children), 0)

if __name__ == '__main__':
  unittest.main(verbosity=2)
