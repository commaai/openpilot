from typing import Optional, Any
import unittest, math
import numpy as np
from tinygrad.tensor import Tensor, _to_np_dtype
from tinygrad.helpers import Context
from tinygrad.dtype import dtypes, DType, AddrSpace, ConstFloat  # noqa: F401
from tinygrad.device import Buffer, Device
from tinygrad.uop.ops import Ops, UOp, KernelInfo, AxisType, buffers
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.engine.realize import run_linear
from tinygrad.codegen import to_program
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.renderer.ptx import PTXRenderer
from test.helpers import to_uops_list

def run_uops(uops_list:list[UOp], bufs:list[Buffer]):
  buf_uops = [UOp.new_buffer(b.device, b.size, b.dtype) for b in bufs]
  for u,b in zip(buf_uops, bufs): buffers[u] = b
  run_linear(UOp(Ops.LINEAR, src=(UOp.sink(*uops_list, arg=KernelInfo()).call(*buf_uops),)))

def uop(uops:list[UOp], op:Ops, dtype:Optional[DType], src:tuple[UOp, ...], arg:Any=None) -> UOp:
  if op is Ops.CONST: uops.append(UOp.const(dtype, arg))
  elif op is Ops.PARAM: uops.append(UOp.param(arg, dtype, shape=(1,)))
  else: uops.append(UOp(op, dtype, tuple(src), arg))
  return uops[-1]

def _test_single_value(vals, op, dts):
  uops = []
  output_dtype = dtypes.bool if op in (Ops.CMPLT, Ops.CMPNE) else dts[-1]
  buf_store = uop(uops, Ops.PARAM, output_dtype, (), 0)
  buf_loads = [uop(uops, Ops.PARAM, dtype, (), i+1) for i,dtype in enumerate(dts)]
  loads = (buf_loads[i].index(uop(uops, Ops.CONST, dtypes.int32, (), 0)) for i, dtype in enumerate(dts))
  alu = uop(uops, op, output_dtype, loads)
  out = uop(uops, Ops.STORE, dtypes.void, (buf_store.index(uop(uops, Ops.CONST, dtypes.int32, (), 0)), alu))
  buf = Buffer(Device.DEFAULT, 1, output_dtype).allocate()
  buf2 = [Buffer(Device.DEFAULT, 1, dtype, initial_value=np.array([a], dtype=_to_np_dtype(dtype)).tobytes()) for a,dtype in zip(vals, dts)]
  run_uops([out], [buf]+buf2)
  return np.frombuffer(buf.as_memoryview(), _to_np_dtype(output_dtype))[0]

def _test_single_value_const(vals, op, dts):
  uops = []
  output_dtype = dtypes.bool if op in (Ops.CMPLT, Ops.CMPNE) else dts[-1]
  buf_store = uop(uops, Ops.PARAM, output_dtype, (), 0)
  loads = (uop(uops, Ops.CONST, dtype, [], a) for a,dtype in zip(vals, dts))
  alu = uop(uops, op, output_dtype, loads)
  out = buf_store[UOp.const(dtypes.int32, 0)].store(alu)
  buf = Buffer(Device.DEFAULT, 1, output_dtype).allocate()
  run_uops([out], [buf])
  return np.frombuffer(buf.as_memoryview(), _to_np_dtype(output_dtype))[0]

def _test_uops_result(output_dtype, uops, res):
  # uops = []
  buf_store = uop(uops, Ops.PARAM, output_dtype, (), 0)
  # res = output_fn(uops)
  out = uop(uops, Ops.STORE, dtypes.void, (buf_store.index(uop(uops, Ops.CONST, dtypes.int32, (), 0)), res))
  buf = Buffer(Device.DEFAULT, 1, output_dtype).allocate()
  run_uops([out], [buf])
  return np.frombuffer(buf.as_memoryview(), _to_np_dtype(output_dtype))[0]

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
        a = dts[0].const(a)
        self._equal(f([a], op, dts), fxn(a))

  def _test_bop_fxn(self, op, fxn, dts=(dtypes.float32, )*2, no_b_zero=False, no_b_neg=False):
    for f in [_test_single_value, _test_single_value_const]:
      for a in [-2.0, 0.0, 1.0]:
        for b in [-3.0, 1.0] + ([] if no_b_zero else [0.0]):
          a = dts[0].const(a)
          b = dts[1].const(abs(b) if no_b_neg else b)
          self._equal(f([a,b], op, dts), fxn(a,b))

  def _test_top_fxn(self, op, fxn, dts=(dtypes.float32, )*3):
    for f in [_test_single_value, _test_single_value_const]:
      for a in [-2.0, 0, 1]:
        for b in [-3.0, 3.0]:
          for c in [-4.0, 4.0]:
            a = dts[0].const(a)
            b = dts[1].const(b)
            c = dts[2].const(c)
            self._equal(f([a,b,c], op, dts), fxn(a,b,c))

class TestFloatUOps(TestUOps):
  @unittest.skipIf(Device.DEFAULT == "CPU", 'not supported as uop')
  def test_exp2(self): self._test_uop_fxn(Ops.EXP2, lambda a: np.exp2(a))
  @unittest.skipIf(Device.DEFAULT == "CPU", 'not supported as uop')
  def test_log2(self): self._test_uop_fxn(Ops.LOG2, lambda a: math.log2(a) if a > 0 else float('-inf' if a==0 else 'nan'))
  @unittest.skipIf(Device.DEFAULT == "CPU", 'not supported as uop')
  def test_sin(self): self._test_uop_fxn(Ops.SIN, lambda a: math.sin(a))
  def test_recip(self): self._test_uop_fxn(Ops.RECIPROCAL, lambda a: 1/a if a != 0 else float('inf'))
  def test_sqrt(self): self._test_uop_fxn(Ops.SQRT, lambda a: math.sqrt(a) if a >= 0 else float('nan'))

  def test_add(self): self._test_bop_fxn(Ops.ADD, lambda a,b: a+b)
  def test_mul(self): self._test_bop_fxn(Ops.MUL, lambda a,b: a*b)
  def test_max(self): self._test_bop_fxn(Ops.MAX, lambda a,b: max(a,b))
  def test_cmplt(self): self._test_bop_fxn(Ops.CMPLT, lambda a,b: a<b)
  def test_cmpne(self): self._test_bop_fxn(Ops.CMPNE, lambda a,b: a!=b)
  @unittest.skipIf(Device.DEFAULT == "WEBGPU", "WEBGPU doesn't support NaN comparison correctly")
  def test_cmpne_nan(self):  # NaN != x for any x (IEEE 754)
    for a, b in [(math.nan, 1.0), (1.0, math.nan), (math.nan, math.nan)]:
      self.assertTrue(_test_single_value(
        [dtypes.float32.const(a), dtypes.float32.const(b)],
        Ops.CMPNE, (dtypes.float32, dtypes.float32)))
  # MOD isn't tested on floats

  def test_where(self):
    self._test_top_fxn(Ops.WHERE, lambda a,b,c: b if a!=0 else c, (dtypes.bool, dtypes.float, dtypes.float))

  @unittest.skipUnless(Device.DEFAULT == "PYTHON", "only python supports MULACC")
  def test_mulacc(self):
    self._test_top_fxn(Ops.MULACC, lambda a,b,c: a*b+c, (dtypes.float, dtypes.float, dtypes.float))

class TestNonFloatUOps(TestUOps):
  def test_add_int32(self): self._test_bop_fxn(Ops.ADD, lambda a,b: int(a)+int(b), (dtypes.int32, dtypes.int32))
  def test_mul_int32(self): self._test_bop_fxn(Ops.MUL, lambda a,b: int(a)*int(b), (dtypes.int32, dtypes.int32))
  @unittest.skipUnless(isinstance(Device[Device.DEFAULT].renderer, (PTXRenderer, CStyleLanguage)), "only ptx and cstyle use bitshifts")
  def test_shr_int32(self): self._test_bop_fxn(Ops.SHR, lambda a,b: int(a)>>int(b), (dtypes.int32, dtypes.int32), no_b_neg=True)
  @unittest.skipUnless(isinstance(Device[Device.DEFAULT].renderer, (PTXRenderer, CStyleLanguage)), "only ptx and cstyle use bitshifts")
  def test_shl_int32(self): self._test_bop_fxn(Ops.SHL, lambda a,b: int(a)<<int(b), (dtypes.int32, dtypes.int32), no_b_neg=True)
  def test_div_int32(self):
    self._test_bop_fxn(Ops.CDIV, lambda a,b: int(a/b), (dtypes.int32, dtypes.int32), no_b_zero=True)
  def test_and_int32(self): self._test_bop_fxn(Ops.AND, lambda a,b: int(a)&int(b), (dtypes.int32, dtypes.int32))
  def test_or_int32(self): self._test_bop_fxn(Ops.OR, lambda a,b: int(a)|int(b), (dtypes.int32, dtypes.int32))
  def test_mod_int32(self):
    self._test_bop_fxn(Ops.CMOD,
                       lambda a,b: abs(int(a))%abs(int(b))*(1,-1)[a<0], (dtypes.int32, dtypes.int32), no_b_zero=True)
  def test_cmplt_int32(self): self._test_bop_fxn(Ops.CMPLT, lambda a,b: int(a)<int(b), (dtypes.int32, dtypes.int32))
  def test_cmpne_int32(self): self._test_bop_fxn(Ops.CMPNE, lambda a,b: int(a)!=int(b), (dtypes.int32, dtypes.int32))
  def test_mul_bool(self): self._test_bop_fxn(Ops.MUL, lambda a,b: bool(a) and bool(b), (dtypes.bool, dtypes.bool))
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

class TestLocalAccess(unittest.TestCase):
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared memory")
  def test_local_basic(self):
    uops = []
    smem = UOp.placeholder((16,), dtypes.float32, slot=0, addrspace=AddrSpace.LOCAL)
    uops.append(smem)
    st = uop(uops, Ops.STORE, dtypes.void, (smem.index(uop(uops, Ops.CONST, dtypes.int32, (), 0)), uop(uops, Ops.CONST, dtypes.float32, (), 42.0)))
    barr = uop(uops, Ops.BARRIER, dtypes.void, (st,))
    sres = uop(uops, Ops.LOAD, dtypes.float32, (smem.after(barr).index(uop(uops, Ops.CONST, dtypes.int32, (), 0)),))
    self.assertEqual(_test_uops_result(dtypes.float32, uops, sres), 42)

  # NOTE: webgpu specific, since only webgpu performs bitpacking
  @unittest.skipUnless(Device.DEFAULT == "WEBGPU", "Test local access with packed data type")
  def test_local_packed(self):
    uops = []
    smem = UOp.placeholder((16,), dtypes.uint8, slot=0, addrspace=AddrSpace.LOCAL)
    uops.append(smem)
    st = uop(uops, Ops.STORE, dtypes.void, (smem.index(uop(uops, Ops.CONST, dtypes.int32, (), 0)), uop(uops, Ops.CONST, dtypes.uint8, (), 42)))
    barr = uop(uops, Ops.BARRIER, dtypes.void, (st,))
    sres = smem.after(barr).index(uop(uops, Ops.CONST, dtypes.int32, (), 0))
    self.assertEqual(_test_uops_result(dtypes.uint8, uops, sres), 42)

  # NOTE: webgpu specific, since only webgpu performs bitpacking
  @unittest.skipUnless(Device.DEFAULT == "WEBGPU", "Test local memory size for packed data types")
  def test_packed_smem_size(self):
    _dtypes = [dtypes.char, dtypes.uchar, dtypes.short, dtypes.ushort, dtypes.half]
    size = 16
    for dtype in _dtypes:
      temp = UOp.placeholder((size,), dtype, slot=0, addrspace=AddrSpace.LOCAL)
      uops = to_uops_list([temp], ren=Device[Device.DEFAULT].renderer)
      out = Device[Device.DEFAULT].renderer.render(uops)
      # half is supported in wgsl, so it doesn't have to be packed
      corrected_size = size//(4//dtype.itemsize) if dtype != dtypes.half else size
      # temp0: array<{Device[Device.DEFAULT].renderer.buf_map(dtype)},{corrected_size}>;
      self.assertIn(f",{corrected_size}>;", out)

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared memory")
  @unittest.skip("tinygrad doesn't support this behavior")
  def test_local_indirect(self):
    uops = []
    smem = UOp.placeholder((16,), dtypes.int32, slot=0, addrspace=AddrSpace.LOCAL)
    uops.append(smem)
    st1 = uop(uops, Ops.STORE, dtypes.void, (smem.index(uop(uops, Ops.CONST, dtypes.int32, (), 1)), uop(uops, Ops.CONST, dtypes.int32, (), 2)))
    st2 = uop(uops, Ops.STORE, dtypes.void, (smem.index(uop(uops, Ops.CONST, dtypes.int32, (), 2)), uop(uops, Ops.CONST, dtypes.int32, (), 42)))
    barr = uop(uops, Ops.BARRIER, dtypes.void, (st1,st2))
    ofs = uop(uops, Ops.LOAD, dtypes.int32, (smem.index(uop(uops, Ops.CONST, dtypes.int32, (), 1)), barr))
    sres = uop(uops, Ops.LOAD, dtypes.int32, (smem.index(ofs),))
    self.assertEqual(_test_uops_result(dtypes.int32, uops, sres), 42)

@unittest.skipUnless(isinstance(Device[Device.DEFAULT].renderer, PTXRenderer), "This only tests assembly backends")
class TestAssembly(unittest.TestCase):
  def test_bitshift_left(self):
    g1 = UOp.param(0, dtypes.int32, shape=(3,))
    out = UOp.param(1, dtypes.int32, shape=(2,))
    c1 = UOp.const(dtypes.int, 2)
    c2 = UOp.const(dtypes.int, 3)
    l1 = g1.index(c1)
    a1 = UOp(Ops.MUL, src=(l1, c1))
    a2 = UOp(Ops.MUL, src=(l1, c2))
    uops = to_uops_list([out.index(UOp.const(dtypes.int, 0)).store(a1), out.index(UOp.const(dtypes.int, 1)).store(a2)],
                        ren=Device[Device.DEFAULT].renderer)
    Device[Device.DEFAULT].renderer.render(uops)
    ops = [x.op for x in uops]
    self.assertIn(Ops.SHL, ops)
    self.assertIn(Ops.MUL, ops)

  @unittest.skip("this is a questionable microoptimization i won't enforce")
  def test_mulacc_unrolled(self):
    # test that     acc = acc + a0*b0 + a1*b1 + a2*b2 + a3*b3
    # is not        acc = acc + (a0*b0 + a1*b1 + a2*b2 + a3*b3)
    a = Tensor.empty(1024)
    b = Tensor.empty(1024)
    c = (a*b).sum()
    ast = c.schedule_linear().src[-1].src[0]
    opts_to_apply = [Opt(OptOps.UNROLL, 0, 4)]
    ast = ast.replace(arg=KernelInfo(opts_to_apply=tuple(opts_to_apply)))
    program = to_program(ast, Device[Device.DEFAULT].renderer)
    uops = tuple(program.src[1].src)
    self.assertGreaterEqual(len([x.op for x in uops if x.op is Ops.MULACC]), 4)

  def test_mulacc_shl(self):
    g1 = UOp.param(0, dtypes.int32, shape=(2,))
    c1 = UOp.const(dtypes.int, 0)
    c2 = UOp.const(dtypes.int, 1)
    expr = g1.index(c1) * UOp.const(dtypes.int, 4096) + g1.index(c2)
    uops = to_uops_list([expr], ren=Device[Device.DEFAULT].renderer)
    Device[Device.DEFAULT].renderer.render(uops)
    self.assertIn(Ops.MULACC, [x.op for x in uops])

  def test_use_cmpeq(self):
    g = UOp.param(0, dtypes.uint32, shape=(8,))
    c = UOp.const(dtypes.uint, 7)
    comp = g.index(c).ne(c).ne(True)
    uops = to_uops_list([comp], ren=Device[Device.DEFAULT].renderer)
    Device[Device.DEFAULT].renderer.render(uops)
    ops = [x.op for x in uops]
    self.assertIn(Ops.CMPEQ, ops)
    self.assertNotIn(Ops.CMPNE, ops)

class TestZeroRange(unittest.TestCase):
  def test_reduce_variable(self):
    for i in range(3,-1,-1):
      v = UOp.variable("i", 0, 5).bind(i)
      out = Tensor.ones(10, dtype=dtypes.int).contiguous().shrink(((0,v),)).sum()
      self.assertEqual(out.item(), i)

class TestUOpPrograms(unittest.TestCase):
  def _run(self, prog:UOp, *tensors:Tensor):
    run_linear(UOp(Ops.LINEAR, src=(prog.call(*[t.uop.buf_uop for t in tensors]),)), update_stats=False)

  def test_simple(self):
    out = Tensor.empty(10,10,dtype=dtypes.int)

    ptr = UOp.placeholder(out.shape, out.dtype, slot=0)
    i, j = UOp.range(10, axis_id=0), UOp.range(10, axis_id=1)
    prog = ptr[i,j].set(42).end(i,j)
    self._run(prog.sink(arg=KernelInfo()), out)

    with Context(DEBUG=0): self.assertTrue((out == 42).all().item())

  def test_matmul(self):
    a = Tensor.randn(10,10)
    b = Tensor.randn(10,10)
    c = Tensor.empty(10,10)
    ref = (a@b)
    with Context(DEBUG=0): Tensor.realize(a, b, c, ref)

    # C[i,j] = sum_k A[i,k] * B[k,j]
    # Shapes: A[M,K], B[K,N], C[M,N]
    M = N = K = 10
    DT = dtypes.float32

    # Placeholders (bind slots explicitly)
    A = UOp.placeholder((M, K), DT, slot=0)
    B = UOp.placeholder((K, N), DT, slot=1)
    C = UOp.placeholder((M, N), DT, slot=2)

    # Axes: i,j are spatial; k is a reduction axis over the shared dim K
    i = UOp.range(M, axis_id=0)                             # rows of A/C
    j = UOp.range(N, axis_id=1)                             # cols of B/C
    k = UOp.range(K, axis_id=2, axis_type=AxisType.REDUCE)  # reduction over K

    # Zero-init: write a scalar 0 to each (i,j).
    C = C[i, j].set(0.0)

    # Accumulate: C_after(k) enforces the dependency along the reduction axis
    C = C[i, j].set(C.after(k)[i, j] + A[i, k] * B[k, j])

    # Finalize the loop nest / schedule in (i, j, k) order
    prog = C.end(i, j, k)

    # run program
    # TODO: make this work with opts_to_apply
    self._run(prog.sink(arg=KernelInfo(opts_to_apply=())), a, b, c)

    with Context(DEBUG=0): self.assertLessEqual((c-ref).square().mean().item(), 1e-6)

  def test_matmul_relu(self):
    a, b, c = Tensor.randn(10,10), Tensor.randn(10,10), Tensor.empty(10,10)
    ref = (a@b).relu()
    with Context(DEBUG=0): Tensor.realize(a, b, c, ref)

    A, B, C = a.uop.placeholder_like(0), b.uop.placeholder_like(1), c.uop.placeholder_like(2)
    i, j, k = UOp.range(10, 0), UOp.range(10, 1), UOp.range(10, 2, axis_type=AxisType.REDUCE)

    C = C[i, j].set(0.0)
    C = C[i, j].set(C.after(k)[i, j] + A[i, k] * B[k, j], end=k)
    C = C[i, j].set(C[i, j].maximum(0.0))

    prog = C.end(i, j)

    self._run(prog.sink(arg=KernelInfo(opts_to_apply=())), a, b, c)
    with Context(DEBUG=0): self.assertLessEqual((c-ref).square().mean().item(), 1e-6)

if __name__ == '__main__':
  unittest.main(verbosity=2)
