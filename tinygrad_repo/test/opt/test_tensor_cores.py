import numpy as np
import unittest

from tinygrad import Device, Tensor, dtypes
from tinygrad.tensor import _to_np_dtype
from tinygrad.uop.ops import Ops, UOp, AxisType
from tinygrad.dtype import DType
from tinygrad.device import Buffer
from tinygrad.helpers import Context, TC_SELECT, TC_OPT
from test.helpers import slow, replace_opts
from tinygrad.engine.realize import run_linear
from tinygrad.codegen import to_program
from tinygrad.codegen.opt import Opt, OptOps, KernelOptError
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.renderer.tc import amd_cdna_1616128
from tinygrad.renderer.llvmir import LLVMRenderer, AMDLLVMRenderer

# TODO: write a clean version of this
from test.backend.test_linearizer import helper_realized_ast, helper_linearizer_opt

# NOTE: to_program always passes in Device[Device.DEFAULT].renderer explicitly for process_replay!!!

def _tc_rand(*shape, dtype:DType) -> Tensor:
  return Tensor.randint(*shape, low=dtype.min, high=dtype.max+1, dtype=dtype) if dtypes.is_int(dtype) else Tensor.rand(*shape, dtype=dtype)

def run_program(prg:UOp, bufs:list[Buffer]):
  buf_uops = [UOp.from_buffer(b) for b in bufs]
  run_linear(UOp(Ops.LINEAR, src=(prg.call(*buf_uops),)))

def _skip_unsupported_tc_dtypes(dtype_in:DType, dtype_out:DType):
  supported_dtypes = Device[Device.DEFAULT].renderer.supported_dtypes()
  if unsupported := [f"{name}={dtype}" for name,dtype in (("dtype_in", dtype_in), ("dtype_out", dtype_out)) if dtype not in supported_dtypes]:
    raise unittest.SkipTest(f"tensor core requires unsupported renderer dtype: {', '.join(unsupported)}")

def tc_reduce_axis(r:Tensor) -> int:
  sche = Scheduler(r.schedule_linear().src[-1].src[0], Device[Device.DEFAULT].renderer)
  sche.apply_opt(Opt(OptOps.TC, 0, (TC_SELECT.value, TC_OPT.value, 1)))
  return sche.axis_types.index(AxisType.REDUCE)

def helper_tc_ensure_uops_and_opts_count(N: int, M:int, K:int, dtype_in:DType, dtype_out:DType, axis:int=0, tc_select:int=-1, tc_opt:int=0,
                                         ensure_triggered:bool=True):
  _skip_unsupported_tc_dtypes(dtype_in, dtype_out)
  a, b = _tc_rand(M, K, dtype=dtype_in), _tc_rand(K, N, dtype=dtype_in)
  r = a.matmul(b, dtype=dtype_out)
  sched = r.schedule_linear()
  realized_ast = sched.src[-1].src[0]
  opts_to_apply = [Opt(OptOps.TC, axis, (tc_select, tc_opt, 1))]

  if ensure_triggered:
    program = to_program(replace_opts(realized_ast, opts_to_apply), Device[Device.DEFAULT].renderer)
    wmmas = len([uop for uop in tuple(program.src[1].src) if uop.op is Ops.WMMA])
    tcs = len([x for x in program.src[0].arg.applied_opts if x.op is OptOps.TC])
    assert wmmas > 0, "tensor core not triggered"
    assert tcs == 1, "tensor core opt not included"
  else:
    try:
      program = to_program(replace_opts(realized_ast, opts_to_apply), Device[Device.DEFAULT].renderer)
      assert False, "OptOps.TC triggered, expected KernelOptError"
    except KernelOptError: pass

def helper_tc_allclose(N:int, M:int, K:int, dtype_in:DType, dtype_out:DType, axis:int=0, tc_select:int=-1, tc_opt:int=0, use_tensor_cores:int=1,
                       extra_opts:list[Opt]=[]):
  _skip_unsupported_tc_dtypes(dtype_in, dtype_out)
  a, b = _tc_rand(M, K, dtype=dtype_in), _tc_rand(K, N, dtype=dtype_in)
  np_a, np_b = a.numpy(), b.numpy()
  r = a.matmul(b, dtype=dtype_out)
  if dtype_in == dtypes.bfloat16: r = r.float()
  realized_ast, bufs = helper_realized_ast(r)
  opts = [Opt(op=OptOps.TC, axis=axis, arg=(tc_select, tc_opt, use_tensor_cores))] + extra_opts
  ast = replace_opts(realized_ast, opts)
  pu = to_program(ast, Device[Device.DEFAULT].renderer)
  if use_tensor_cores == 1: assert len([uop for uop in pu.src[1].src if uop.op is Ops.WMMA]) > 0, "wmma not triggered"
  assert len([x for x in pu.src[0].arg.applied_opts if x.op is OptOps.TC]) == 1, "tensor core opt not included"
  run_program(ast, bufs)
  if dtype_in == dtypes.half: tc_atol, tc_rtol = 1e-2, 1e-3
  elif dtype_in == dtypes.bfloat16: tc_atol, tc_rtol = (1e-1, 2e-2) if dtype_out == dtypes.bfloat16 else (1e-2, 1e-2)
  elif not dtypes.is_float(dtype_in): tc_atol, tc_rtol = 0, 0
  else: tc_atol, tc_rtol = 5e-3, 1e-4
  c = bufs[0].numpy().reshape((M,N))
  ref = (np_a.astype(np.int32) @ np_b.astype(np.int32)) if not dtypes.is_float(dtype_in) else (np_a @ np_b)
  np.testing.assert_allclose(c, ref, atol=tc_atol, rtol=tc_rtol)

class TestTensorCores(unittest.TestCase):
  # TODO: don't skip bf16 for real device (METAL, AMD)
  @Context(ALLOW_TF32=1)
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_tensor_cores(self):
    for tc in Device[Device.DEFAULT].renderer.tensor_cores:
      with self.subTest(tc=tc):
        helper_tc_allclose(tc.dims[0], tc.dims[1], tc.dims[2], tc.dtype_in, tc.dtype_out, axis=0, tc_opt=0)

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_tensor_cores_nan(self):
    for tc in [tc for tc in Device[Device.DEFAULT].renderer.tensor_cores if dtypes.is_float(tc.dtype_in)]:
      with self.subTest(tc=tc):
        _skip_unsupported_tc_dtypes(tc.dtype_in, tc.dtype_out)
        a, b = Tensor.full((tc.dims[1], tc.dims[2]), float("nan"), dtype=tc.dtype_in), Tensor.ones(tc.dims[2], tc.dims[0], dtype=tc.dtype_in)
        realized_ast, bufs = helper_realized_ast(a.matmul(b, dtype=tc.dtype_out))
        run_program(replace_opts(realized_ast, [Opt(OptOps.TC, 0, (-1, 0, 1))]), bufs)
        self.assertTrue(np.isnan(bufs[0].numpy()).all())

  @unittest.skipUnless(Device.DEFAULT == "PYTHON" and Device[Device.DEFAULT].renderer.tensor_cores, "test requires emulated tensor cores")
  def test_tensor_cores_emulated_half(self):
    # the fragment layout is the instruction's, a dtype decomp only changes what carries the operands
    for tc in [tc for tc in Device[Device.DEFAULT].renderer.tensor_cores if dtypes.half in (tc.dtype_in, tc.dtype_out)]:
      with self.subTest(tc=tc), Context(EMULATED_DTYPES="half", SPEC=2):
        helper_tc_allclose(tc.dims[0], tc.dims[1], tc.dims[2], tc.dtype_in, tc.dtype_out, axis=0, tc_opt=0)

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_tensor_cores_partial_sum_in_accumulator(self):
    # the heuristic tiles M, N and K after the TC opt: every partial sum enters the next WMMA's accumulator, never an add after it
    for i, tc in enumerate(Device[Device.DEFAULT].renderer.tensor_cores):
      with self.subTest(tc=tc):
        _skip_unsupported_tc_dtypes(tc.dtype_in, tc.dtype_out)
        with Context(ALLOW_TF32=1, TC_SELECT=i, TC_OPT=2):
          a = _tc_rand(tc.dims[1]*8, tc.dims[2]*8, dtype=tc.dtype_in)
          b = _tc_rand(tc.dims[2]*8, tc.dims[0]*8, dtype=tc.dtype_in)
          ast = a.matmul(b, dtype=tc.dtype_out).schedule_linear().src[-1].src[0]
          wmmas = [u for u in to_program(ast, Device[Device.DEFAULT].renderer).src[1].src if u.op is Ops.WMMA]
        self.assertGreater(len(wmmas), 0)
        for u in wmmas: self.assertTrue(any(x.op is Ops.LOAD for x in u.src[2].toposort()), f"accumulator is {u.src[2]}")

  @Context(ALLOW_TF32=1)
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  @unittest.skipIf(Device.DEFAULT == "AMD" and Device[Device.DEFAULT].renderer.target.arch.startswith("gfx9"),
                   "TODO: crashes the worker on MOCKKFD gfx950 in CI, passes locally")
  def test_tensor_cores_extra_locals(self):
    # LOCAL splits after the TC opt: the WARP must keep a whole hardware local dim, its lanes are consecutive threads
    for tc in Device[Device.DEFAULT].renderer.tensor_cores:
      with self.subTest(tc=tc):
        helper_tc_allclose(tc.dims[0]*8, tc.dims[1]*8, tc.dims[2], tc.dtype_in, tc.dtype_out,
                           extra_opts=[Opt(OptOps.SPLIT, 0, (2, AxisType.LOCAL))]*3)

  @Context(ALLOW_TF32=1)
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_tensor_cores_upcast_shared_axis(self):
    # same operand shapes
    tc = next(tc for tc in Device[Device.DEFAULT].renderer.tensor_cores if tc.dtype_in not in dtypes.fp8s)
    N, M, K = tc.dims
    a, b = Tensor.rand(3, M*2, K*2, dtype=tc.dtype_in), Tensor.rand(3, K*2, N*2, dtype=tc.dtype_in)
    helper_linearizer_opt(a.matmul(b, dtype=tc.dtype_out), [[Opt(OptOps.TC, 0, (-1, 0, 1)), Opt(OptOps.SPLIT, 0, (0, AxisType.UPCAST))]],
                          atol=3e-2, rtol=1e-3, check_default_opt=False)

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_tensor_cores_padto_warp(self):
    # the WARP is the hardware simdgroup width, it can't be padded
    tc = Device[Device.DEFAULT].renderer.tensor_cores[0]
    sche = Scheduler(Tensor.empty(64, 64, dtype=tc.dtype_in).matmul(Tensor.empty(64, 64, dtype=tc.dtype_in), dtype=tc.dtype_out)
                     .schedule_linear().src[-1].src[0], Device[Device.DEFAULT].renderer)
    sche.apply_opt(Opt(OptOps.TC, 0, (-1, 0, 1)))
    with self.assertRaises(KernelOptError): sche.apply_opt(Opt(OptOps.PADTO, sche.axis_types.index(AxisType.WARP), 7))

  @Context(ALLOW_TF32=1)
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_tensor_cores_group_reduce(self):
    tc = next(tc for tc in Device[Device.DEFAULT].renderer.tensor_cores if tc.dtype_in not in dtypes.fp8s)
    sche = Scheduler(Tensor.empty(16, 64, dtype=tc.dtype_in).matmul(Tensor.empty(64, 16, dtype=tc.dtype_in), dtype=tc.dtype_out)
                      .schedule_linear().src[-1].src[0], Device[Device.DEFAULT].renderer)
    sche.apply_opt(Opt(OptOps.TC, 0, (-1, 0, 1)))
    axis = sche.axis_types.index(AxisType.REDUCE)
    if AxisType.UNROLL in sche.axis_types:
      # this tc keeps an unrolled reduce outside the WMMA, grouping inside it must be rejected
      with self.assertRaises(KernelOptError): sche.apply_opt(Opt(OptOps.SPLIT, axis, (2, AxisType.GROUP_REDUCE)))
    else:
      x, y = Tensor.rand(16, 64, dtype=tc.dtype_in), Tensor.rand(64, 16, dtype=tc.dtype_in)
      helper_linearizer_opt(x.matmul(y, dtype=tc.dtype_out),
                            [[Opt(OptOps.SPLIT, axis, (amt, AxisType.GROUP_REDUCE, top))] for amt in (2, 4) for top in (False, True)],
                            apply_tc=True, atol=3e-2, rtol=1e-3, check_default_opt=False)

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_tensor_cores_failed_padto(self):
    N, M, K = (tc:=Device[Device.DEFAULT].renderer.tensor_cores[0]).dims
    sche = Scheduler(Tensor.empty(M//4, K, dtype=tc.dtype_in).matmul(Tensor.empty(K, N+N//2, dtype=tc.dtype_in), dtype=tc.dtype_out)
                     .schedule_linear().src[-1].src[0], Device[Device.DEFAULT].renderer)
    # N pads, then M is too small to pad. the failed attempt leaves the ast untouched
    ast = sche.ast
    with self.assertRaises(KernelOptError): sche.apply_opt(Opt(OptOps.TC, 0, (-1, 2, 1)))
    self.assertIs(sche.ast, ast)

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_tensor_cores_nested_reduce(self):
    tc = Device[Device.DEFAULT].renderer.tensor_cores[0]
    a, b = Tensor.empty(tc.dims[1]*2, tc.dims[2], dtype=tc.dtype_in), Tensor.empty(tc.dims[2], tc.dims[0], dtype=tc.dtype_in)
    ast = replace_opts(a.matmul(b, dtype=tc.dtype_out).sum(0).schedule_linear().src[-1].src[0], [Opt(OptOps.TC, 0, (-1, 0, 1))])
    with self.assertRaises(KernelOptError): to_program(ast, Device[Device.DEFAULT].renderer)

  @Context(ALLOW_TF32=1)
  @unittest.skipIf(Device.DEFAULT == "PYTHON", "not generated on EMULATED device")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_tensor_cores_codegen(self):
    for tc in Device[Device.DEFAULT].renderer.tensor_cores:
      n, m, k = tc.dims
      a, b = _tc_rand(m, k, dtype=tc.dtype_in), _tc_rand(k, n, dtype=tc.dtype_in)
      r = a.matmul(b, dtype=tc.dtype_out)
      prg = to_program(replace_opts(r.schedule_linear().src[-1].src[0],
                        [Opt(op=OptOps.TC, axis=0, arg=(-1, 2, 1))]), Device[Device.DEFAULT].renderer)
      if isinstance(Device[Device.DEFAULT].renderer, AMDLLVMRenderer):
        # RDNA emits wmma intrinsics, CDNA emits mfma intrinsics
        assert ("@llvm.amdgcn.wmma" in prg.src[2].arg) or ("@llvm.amdgcn.mfma" in prg.src[2].arg)
      elif isinstance(Device[Device.DEFAULT].renderer, LLVMRenderer):
        assert "0x201000" in prg.src[2].arg
      elif Device[Device.DEFAULT].renderer.suffix == "PTX":
        assert "mma.sync.aligned" in prg.src[2].arg
      else:
        assert "__WMMA_" in prg.src[2].arg

  @Context(ALLOW_TF32=1)
  @unittest.skipIf((Device.DEFAULT == "AMD") or (Device.DEFAULT == "PYTHON" and Device.default.renderer.target.device == "AMD"), "broken for AMD")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_tensor_cores_padded(self):
    for tc in Device[Device.DEFAULT].renderer.tensor_cores:
      helper_tc_allclose(tc.dims[0]+(pad:=1), tc.dims[1]+pad, tc.dims[2]+pad, tc.dtype_in, tc.dtype_out, tc_opt=2)

  @Context(ALLOW_TF32=1)
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_tensor_cores_padded_uops(self):
    for tc in Device[Device.DEFAULT].renderer.tensor_cores:
      pad = 1

      # check that TC is triggered for TC_OPT=2
      helper_tc_ensure_uops_and_opts_count(tc.dims[0]+pad, tc.dims[1]+pad, tc.dims[2]+pad,
                                           tc.dtype_in, tc.dtype_out, tc_opt=2, ensure_triggered=True)

      # check that TC is not triggered for TC_OPT<2
      helper_tc_ensure_uops_and_opts_count(tc.dims[0]+pad, tc.dims[1]+pad, tc.dims[2]+pad,
                                           tc.dtype_in, tc.dtype_out, tc_opt=1, ensure_triggered=False)
      helper_tc_ensure_uops_and_opts_count(tc.dims[0]+pad, tc.dims[1]+pad, tc.dims[2]+pad,
                                           tc.dtype_in, tc.dtype_out, tc_opt=0, ensure_triggered=False)

      # check excessive padding doesn't trigger padded TC in TC_OPT=2
      helper_tc_ensure_uops_and_opts_count(tc.dims[0]//4, tc.dims[1], tc.dims[2], tc.dtype_in, tc.dtype_out, tc_opt=2, ensure_triggered=False)
      helper_tc_ensure_uops_and_opts_count(tc.dims[0], tc.dims[1]//4, tc.dims[2], tc.dtype_in, tc.dtype_out, tc_opt=2, ensure_triggered=False)
      if tc not in amd_cdna_1616128:
        helper_tc_ensure_uops_and_opts_count(tc.dims[0], tc.dims[1], tc.dims[2]//8, tc.dtype_in, tc.dtype_out, tc_opt=2, ensure_triggered=False)

  @Context(ALLOW_TF32=1)
  @unittest.skipUnless(any(tc.dtype_in in (dtypes.half, dtypes.float) for tc in Device[Device.DEFAULT].renderer.tensor_cores),
                       "test requires half or float tensor cores")
  def test_tensor_cores_padto_unroll(self):
    # a padded then fully unrolled reduce makes both operands of one WMMA constant, its output is still a register
    tc = next(tc for tc in Device[Device.DEFAULT].renderer.tensor_cores if tc.dtype_in in (dtypes.half, dtypes.float))
    Tensor.manual_seed(3)
    a = Tensor.rand(tc.dims[1]*2+1, tc.dims[2]*3-1, dtype=tc.dtype_in).realize()
    b = Tensor.rand(tc.dims[2]*3-1, tc.dims[0]*2+1, dtype=tc.dtype_in).realize()
    sche = Scheduler(a.matmul(b, dtype=tc.dtype_out).schedule_linear().src[-1].src[0], Device[Device.DEFAULT].renderer)
    sche.apply_opt(tc_opt:=Opt(OptOps.TC, 0, (-1, 2, 1)))
    axis = sche.axis_types.index(AxisType.REDUCE)
    helper_linearizer_opt(a.matmul(b, dtype=tc.dtype_out), [[tc_opt, Opt(OptOps.PADTO, axis, 4), Opt(OptOps.SPLIT, axis, (2, AxisType.UNROLL)),
                                                            Opt(OptOps.SPLIT, axis, (0, AxisType.UNROLL))]],
                          check_default_opt=False, atol=3e-2, rtol=1e-3)

  @Context(ALLOW_TF32=1)
  @unittest.skipUnless(any(tc.dtype_in in (dtypes.half, dtypes.float) for tc in Device[Device.DEFAULT].renderer.tensor_cores),
                       "test requires half or float tensor cores")
  def test_tensor_cores_padto_masked_operand(self):
    # tc_opt=2 pads K. an ALU between the load and the multiply is fine, a where with a defined false arm is not
    tc = next(tc for tc in Device[Device.DEFAULT].renderer.tensor_cores if tc.dtype_in in (dtypes.half, dtypes.float))
    Tensor.manual_seed(3)
    a = Tensor.rand(tc.dims[1]*2+1, tc.dims[2]*3-1, dtype=tc.dtype_in).realize()
    b = Tensor.rand(tc.dims[2]*3-1, tc.dims[0]*2+1, dtype=tc.dtype_in).realize()
    tc_opt = Opt(OptOps.TC, 0, (-1, 2, 1))
    helper_linearizer_opt((a+1).matmul(b+1, dtype=tc.dtype_out), [[tc_opt]], check_default_opt=False, atol=3e-2, rtol=1e-3)
    one = Tensor(1, dtype=tc.dtype_in)
    ma = (Tensor.rand(a.shape[0], 1) > 0.5).expand(a.shape).where(a, one)
    mb = (Tensor.rand(1, b.shape[1]) > 0.5).expand(b.shape).where(b, one)
    helper_linearizer_opt(ma.matmul(mb, dtype=tc.dtype_out), [[tc_opt]], check_default_opt=False, atol=3e-2, rtol=1e-3)

  @Context(ALLOW_TF32=1)
  @unittest.skipIf(Device.DEFAULT == "PYTHON", "not generated on EMULATED device")
  @slow
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_tensor_cores_multi_reduce(self):
    for tc in Device[Device.DEFAULT].renderer.tensor_cores:
      if tc.dtype_in is dtypes.bfloat16: continue # <-- broken with numpy
      # this will be a M=G16, N=G32, M=G16, M=G16, K=R16, K=R16, K=R16 with 9 choices of TC MNK axes
      golden_result = None
      a = Tensor.rand(16, 16, 29, 29, dtype=tc.dtype_in).realize()
      b = Tensor.rand(32, 16, 16, 16, dtype=tc.dtype_in).realize()
      for axis in range(9):
        c = a.conv2d(b, padding=1, dtype=tc.dtype_out)
        realized_ast, real_bufs = helper_realized_ast(c)

        ast = replace_opts(realized_ast, [Opt(OptOps.TC, axis, (-1, 2, 1))])
        program = to_program(ast, Device[Device.DEFAULT].renderer)
        assert len([uop for uop in tuple(program.src[1].src) if uop.op is Ops.WMMA]) > 0, "tensor core not triggered"
        assert len([x for x in program.src[0].arg.applied_opts if x.op is OptOps.TC]) == 1, "tensor core opt not included"

        # TODO: support this even if numpy doesn't
        if _to_np_dtype(real_bufs[0].dtype) is None: continue
        # Zero to check that all values are filled
        real_bufs[0].copy_from(Buffer("PYTHON", real_bufs[0].size, real_bufs[0].dtype, opaque=memoryview(bytearray(real_bufs[0].nbytes))))
        run_program(ast, real_bufs)
        result = np.frombuffer(real_bufs[0].as_memoryview(), _to_np_dtype(real_bufs[0].dtype))

        # ensure the results for each choice of axis matches
        if golden_result is None: golden_result = result.copy()
        np.testing.assert_allclose(result, golden_result, atol=0.1, rtol=0.2)

  @Context(ALLOW_TF32=1)
  @unittest.skipIf(Device.DEFAULT == "PYTHON", "slow on EMULATED device")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_tensor_cores_unroll_phi(self):
    # skip fp8 tcs: the unoptimized ALU baseline quantizes products to fp8 (JAX promotion), which legitimately
    # differs from the MFMA path (f32 accumulation), so the baseline-vs-TC numerical gate can't hold for fp8.
    tc = next(tc for tc in Device[Device.DEFAULT].renderer.tensor_cores if tc.dtype_in not in dtypes.fp8s)
    x, y = Tensor.rand(16, 64, dtype=tc.dtype_in).realize(), Tensor.rand(64, 16, dtype=tc.dtype_in).realize()
    opts = [Opt(OptOps.TC, 0, (-1, 0, 1)), Opt(OptOps.SPLIT, tc_reduce_axis(x.matmul(y, dtype=tc.dtype_out)), (2, AxisType.UNROLL))]
    r = x.matmul(y, dtype=tc.dtype_out)
    ast = helper_linearizer_opt(r, [opts[1:]], apply_tc=True, atol=3e-2, rtol=1e-3, check_default_opt=False)
    wmmas = [u for u in tuple(to_program(replace_opts(ast, opts), Device[Device.DEFAULT].renderer).src[1].src) if u.op is Ops.WMMA]
    self.assertGreater(len(wmmas), 0)
    for u in wmmas: assert u.src[-1].src[0].op != Ops.STORE

  @Context(ALLOW_TF32=1)
  @unittest.skipIf(Device.DEFAULT == "PYTHON", "slow on EMULATED device")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  @unittest.skipIf(Device.DEFAULT in {"CPU"}, "CPU does not support using a different type for accumulation")
  def test_tensor_cores_unroll_casted_phi(self):
    tc = [tc for tc in Device[Device.DEFAULT].renderer.tensor_cores if tc.dtype_in != tc.dtype_out and tc.dtype_in not in dtypes.fp8s][0]
    x, y = Tensor.rand(16, 64, dtype=tc.dtype_in).realize(), Tensor.rand(64, 16, dtype=tc.dtype_in).realize()
    opts = [Opt(OptOps.TC, 0, (-1, 0, 1)), Opt(OptOps.SPLIT, tc_reduce_axis(x.matmul(y, dtype=tc.dtype_out)), (2, AxisType.UNROLL))]
    r = x.matmul(y, dtype=tc.dtype_out)
    ast = helper_linearizer_opt(r, [opts[1:]], apply_tc=True, atol=3e-2, rtol=1e-3, check_default_opt=False)
    wmmas = [u for u in tuple(to_program(replace_opts(ast, opts), Device[Device.DEFAULT].renderer).src[1].src) if u.op is Ops.WMMA]
    self.assertGreater(len(wmmas), 0)
    for u in wmmas: assert u.src[-1].src[0].op != Ops.STORE

  @Context(ALLOW_TF32=1)
  @unittest.skipIf(Device.DEFAULT == "PYTHON", "slow on EMULATED device")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  @unittest.skipIf(Device.DEFAULT in {"CPU"}, "CPU does not support using a different type for accumulation")
  def test_tensor_cores_unroll_casted_phi_with_children(self):
    # all STORE children are outside the loop
    tc = [tc for tc in Device[Device.DEFAULT].renderer.tensor_cores if tc.dtype_in != tc.dtype_out and tc.dtype_in not in dtypes.fp8s][0]
    x, y = Tensor.rand(16, 64, dtype=tc.dtype_in).realize(), Tensor.rand(64, 16, dtype=tc.dtype_in).realize()
    opts = [Opt(OptOps.TC, 0, (-1, 0, 1)), Opt(OptOps.SPLIT, tc_reduce_axis(x.matmul(y, dtype=tc.dtype_out).relu()), (2, AxisType.UNROLL))]
    r = x.matmul(y, dtype=tc.dtype_out).relu()
    ast = helper_linearizer_opt(r, [opts[1:]], apply_tc=True, atol=3e-2, rtol=1e-3, check_default_opt=False)
    wmmas = [u for u in tuple(to_program(replace_opts(ast, opts), Device[Device.DEFAULT].renderer).src[1].src) if u.op is Ops.WMMA]
    self.assertGreater(len(wmmas), 0)
    for u in wmmas: assert u.src[-1].src[0].op != Ops.STORE

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  @unittest.skipUnless(any(tc.dtype_in == tc.dtype_out == dtypes.half for tc in Device[Device.DEFAULT].renderer.tensor_cores),
                      "test requires tensor cores with accumulation in half") # testing with half suffices.
  @unittest.skipIf(Device.DEFAULT == "PYTHON", "slow on EMULATED device")
  def test_tensor_core_opts(self):
    N = 128
    Tensor.manual_seed(1552)
    a, b = Tensor.rand(N, N, dtype=dtypes.half).realize(), Tensor.rand(N, N, dtype=dtypes.half).realize()
    R = tc_reduce_axis(a.matmul(b, dtype=dtypes.half))
    r = a.matmul(b, dtype=dtypes.half)
    atol, rtol = 0.25, 0.01
    helper_linearizer_opt(r, [
      [],
      [Opt(OptOps.SPLIT, 0, (4, AxisType.UPCAST))],
      [Opt(OptOps.SPLIT, 1, (4, AxisType.UPCAST))],
      [Opt(OptOps.SPLIT, 0, (4, AxisType.UPCAST)), Opt(OptOps.SPLIT, 1, (4, AxisType.UPCAST))], # check upcasts
      [Opt(OptOps.SPLIT, R, (2, AxisType.UNROLL))], # check unroll
      [Opt(OptOps.SPLIT, 0, (4, AxisType.UPCAST)), Opt(OptOps.SPLIT, R+1, (2, AxisType.UNROLL))], # check combo of unroll and upcast
      [Opt(OptOps.SPLIT, 0, (4, AxisType.UPCAST)), Opt(OptOps.SPLIT, 1, (4, AxisType.UPCAST)), Opt(OptOps.SPLIT, R+2, (2, AxisType.UNROLL))],
      [Opt(OptOps.SPLIT, 0, (4, AxisType.UPCAST)), Opt(OptOps.SPLIT, 1, (4, AxisType.UPCAST)), Opt(OptOps.SPLIT, R+2, (4, AxisType.UNROLL))],
    ], apply_tc=True, atol=atol, rtol=rtol)

  @unittest.skipUnless(any(tc.dtype_in in (dtypes.half, dtypes.float) for tc in Device[Device.DEFAULT].renderer.tensor_cores),
                       "test requires half or float tensor cores")
  def test_tc_shape_padded(self):
    tc = next(tc for tc in Device[Device.DEFAULT].renderer.tensor_cores if tc.dtype_in in (dtypes.half, dtypes.float))
    Tensor.manual_seed(3)
    a, b = Tensor.rand(17, 23, dtype=tc.dtype_in).realize(), Tensor.rand(23, 29, dtype=tc.dtype_in).realize()
    with Context(ALLOW_TF32=1):
      helper_linearizer_opt(a.matmul(b, dtype=tc.dtype_out), [[Opt(OptOps.TC, 0, (-1, 2, 2))]], check_default_opt=False, atol=3e-2, rtol=1e-3)

  @unittest.skipUnless(any(tc.dtype_in in (dtypes.half, dtypes.float) for tc in Device[Device.DEFAULT].renderer.tensor_cores),
                       "test requires half or float tensor cores")
  @unittest.skipIf(Device.DEFAULT == "AMD" and Device[Device.DEFAULT].renderer.target.arch.startswith(("gfx11", "gfx12")),
                   "TODO: LLVM AMDGPU miscompiles RDNA WMMA with masked operands, passes on PYTHON::gfx1100")
  def test_tc_padto_full_upcast(self):
    # a fully upcast pad lane is gated to 0 on the WMMA operand
    tc = next(tc for tc in Device[Device.DEFAULT].renderer.tensor_cores if tc.dtype_in in (dtypes.half, dtypes.float))
    Tensor.manual_seed(3)
    a, b = Tensor.rand(17, 23, dtype=tc.dtype_in).realize(), Tensor.rand(23, 29, dtype=tc.dtype_in).realize()
    with Context(ALLOW_TF32=1):
      helper_linearizer_opt(a.matmul(b, dtype=tc.dtype_out),
                            [[Opt(OptOps.TC, 0, (-1, 2, 1)), Opt(OptOps.PADTO, 0, 4), Opt(OptOps.SPLIT, 0, (0, AxisType.UPCAST))]],
                            check_default_opt=False, atol=3e-2, rtol=1e-3)

if __name__ == '__main__':
  unittest.main()
