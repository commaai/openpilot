import numpy as np
import unittest
from dataclasses import replace

from tinygrad import Device, Tensor, dtypes
from tinygrad.tensor import _to_np_dtype
from tinygrad.uop.ops import Ops
from tinygrad.dtype import DType
from tinygrad.device import is_dtype_supported
from tinygrad.helpers import AMX, AMD_LLVM, CPU_LLVM, Context
from test.helpers import slow
from tinygrad.engine.realize import CompiledRunner, get_program
from tinygrad.codegen.opt import Opt, OptOps, KernelOptError
from tinygrad.codegen.opt.tc import amd_cdna_1616128

# TODO: write a clean version of this
from test.test_linearizer import helper_realized_ast, helper_linearizer_opt

# NOTE: get_program always passes in Device[Device.DEFAULT].renderer explicitly for process_replay!!!

def helper_tc_ensure_uops_and_opts_count(N: int, M:int, K:int, dtype_in:DType, dtype_out:DType, axis:int=0, tc_select:int=-1, tc_opt:int=0,
                                         ensure_triggered:bool=True):
  a, b = Tensor.rand(M, K, dtype=dtype_in), Tensor.rand(K, N, dtype=dtype_in)
  r = a.matmul(b, dtype=dtype_out)
  sched = r.schedule()
  realized_ast = sched[-1].ast
  opts_to_apply = [Opt(OptOps.TC, axis, (tc_select, tc_opt, 1))]

  if ensure_triggered:
    program = get_program(realized_ast, Device[Device.DEFAULT].renderer, opts=opts_to_apply)
    wmmas = len([uop for uop in program.uops if uop.op is Ops.WMMA])
    tcs = len([x for x in program.applied_opts if x.op is OptOps.TC])
    assert wmmas > 0, "tensor core not triggered"
    assert tcs == 1, "tensor core opt not included"
  else:
    try:
      program = get_program(realized_ast, Device[Device.DEFAULT].renderer, opts=opts_to_apply)
      assert False, "OptOps.TC triggered, expected KernelOptError"
    except KernelOptError: pass

def helper_tc_allclose(N:int, M:int, K:int, dtype_in:DType, dtype_out:DType, axis:int=0, tc_select:int=-1, tc_opt:int=0, use_tensor_cores:int=1):
  a, b = Tensor.rand(M, K, dtype=dtype_in), Tensor.rand(K, N, dtype=dtype_in)
  np_a, np_b = a.numpy(), b.numpy()
  r = a.matmul(b, dtype=dtype_out)
  if dtype_in == dtypes.bfloat16: r = r.float()
  realized_ast, bufs = helper_realized_ast(r)
  opts = [Opt(op=OptOps.TC, axis=axis, arg=(tc_select, tc_opt, use_tensor_cores))]
  prg = CompiledRunner(replace(get_program(realized_ast, Device[Device.DEFAULT].renderer, opts=opts), device=Device.DEFAULT))
  if use_tensor_cores == 1: assert len([uop for uop in prg.p.uops if uop.op is Ops.WMMA]) > 0, "wmma not triggered"
  assert len([x for x in prg.p.uops[-1].arg.applied_opts if x.op is OptOps.TC]) == 1, "tensor core opt not included"
  prg.exec(bufs)
  if dtype_in == dtypes.half: tc_atol, tc_rtol = 1e-2, 1e-3
  elif dtype_in == dtypes.bfloat16: tc_atol, tc_rtol = (1e-1, 2e-2) if dtype_out == dtypes.bfloat16 else (1e-2, 1e-2)
  else: tc_atol, tc_rtol = 5e-3, 1e-4
  c = bufs[0].numpy().reshape((M,N))
  np.testing.assert_allclose(c, np_a @ np_b, atol=tc_atol, rtol=tc_rtol)

class TestTensorCores(unittest.TestCase):
  # TODO: don't skip bf16 for real device (METAL, AMD)
  @Context(ALLOW_TF32=1)
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_tensor_cores(self):
    for tc in Device[Device.DEFAULT].renderer.tensor_cores:
      if not is_dtype_supported(tc.dtype_in) or not is_dtype_supported(tc.dtype_out): continue
      # for AMX, tc.dims[2] == 1 so reduceop is None thus tensor_cores are not triggered
      helper_tc_allclose(tc.dims[0], tc.dims[1], 2 if AMX else tc.dims[2], tc.dtype_in, tc.dtype_out, axis=0, tc_opt=0)

  @Context(ALLOW_TF32=1)
  @unittest.skipIf(Device.DEFAULT == "PYTHON", "not generated on EMULATED device")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_tensor_cores_codegen(self):
    for tc in Device[Device.DEFAULT].renderer.tensor_cores:
      if not is_dtype_supported(tc.dtype_in) or not is_dtype_supported(tc.dtype_out): continue
      n, m, k = tc.dims[0], tc.dims[1], 2 if AMX else tc.dims[2]
      a, b = Tensor.rand(m, k, dtype=tc.dtype_in), Tensor.rand(k, n, dtype=tc.dtype_in)
      r = a.matmul(b, dtype=tc.dtype_out)
      prg = get_program(r.schedule()[-1].ast, Device[Device.DEFAULT].renderer, opts=[Opt(op=OptOps.TC, axis=0, arg=(-1, 2, 1))])
      if Device.DEFAULT == "CPU" and CPU_LLVM:
        assert "0x201000" in prg.src
      elif Device.DEFAULT == "AMD" and AMD_LLVM:
        assert "@llvm.amdgcn.wmma" in prg.src
      elif Device[Device.DEFAULT].renderer.suffix == "PTX":
        assert "mma.sync.aligned" in prg.src
      else:
        assert "__WMMA_" in prg.src

  @Context(ALLOW_TF32=1)
  @unittest.skipIf((Device.DEFAULT == "AMD") or (Device.DEFAULT == "PYTHON" and Device.default.renderer.device == "AMD"), "broken for AMD")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_tensor_cores_padded(self):
    for tc in Device[Device.DEFAULT].renderer.tensor_cores:
      if not is_dtype_supported(tc.dtype_in) or not is_dtype_supported(tc.dtype_out): continue
      helper_tc_allclose(tc.dims[0]+(pad:=1), tc.dims[1]+pad, tc.dims[2]+pad, tc.dtype_in, tc.dtype_out, tc_opt=2)

  # AMD compiler bug: AMD miscompiles non-zero padded tc kernels with -O3, producing wrong results, nans or hang (see #9606)
  # Internal bug: zero-stride dimensions combined with a mask may produce wrong index/valid for pad == 1 on AMD
  @unittest.skipUnless((Device.DEFAULT == "AMD") or (Device.DEFAULT == "PYTHON" and Device.default.renderer.device == "AMD"), "test for AMD's tc")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  @unittest.skip("warp elements not duplicated properly across lanes")
  def test_tensor_cores_padded_amd(self):
    for tc in Device[Device.DEFAULT].renderer.tensor_cores:
      if not is_dtype_supported(tc.dtype_in) or not is_dtype_supported(tc.dtype_out): continue
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
      if not AMX and tc not in amd_cdna_1616128: # AMX tc.dims[2] == 1
        helper_tc_ensure_uops_and_opts_count(tc.dims[0], tc.dims[1], tc.dims[2]//8, tc.dtype_in, tc.dtype_out, tc_opt=2, ensure_triggered=False)

  @Context(ALLOW_TF32=1)
  @unittest.skipIf(Device.DEFAULT == "PYTHON", "not generated on EMULATED device")
  @slow
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_tensor_cores_multi_reduce(self):
    for tc in Device[Device.DEFAULT].renderer.tensor_cores:
      if not is_dtype_supported(tc.dtype_in) or not is_dtype_supported(tc.dtype_out): continue
      if tc.dtype_in is dtypes.bfloat16: continue # <-- broken with numpy
      # this will be a M=G16, N=G32, M=G16, M=G16, K=R16, K=R16, K=R16 with 9 choices of TC MNK axes
      golden_result = None
      for axis in range(9):
        a = Tensor.rand(16, 16, 29, 29, dtype=tc.dtype_in).realize()
        b = Tensor.rand(32, 16, 16, 16, dtype=tc.dtype_in).realize()
        c = a.conv2d(b, padding=1, dtype=tc.dtype_out)
        realized_ast, real_bufs = helper_realized_ast(c)

        program = get_program(realized_ast, Device[Device.DEFAULT].renderer, opts=[Opt(OptOps.TC, axis, (-1, 2, 1))])
        assert len([uop for uop in program.uops if uop.op is Ops.WMMA]) > 0, "tensor core not triggered"
        assert len([x for x in program.applied_opts if x.op is OptOps.TC]) == 1, "tensor core opt not included"

        prg = CompiledRunner(program)
        # TODO: support this even if numpy doesn't
        if _to_np_dtype(real_bufs[0].dtype) is None: continue
        real_bufs[0].copyin(np.zeros((real_bufs[0].size, ), dtype=_to_np_dtype(real_bufs[0].dtype)).data) # Zero to check that all values are filled
        prg.exec(real_bufs)
        result = np.frombuffer(real_bufs[0].as_buffer(), _to_np_dtype(real_bufs[0].dtype))

        # ensure the results for each choice of axis matches
        if golden_result is None: golden_result = np.frombuffer(real_bufs[0].as_buffer(), _to_np_dtype(real_bufs[0].dtype))
        np.testing.assert_allclose(result, golden_result, atol=0.1, rtol=0.2)

  @Context(ALLOW_TF32=1)
  @unittest.skipIf(Device.DEFAULT == "PYTHON", "slow on EMULATED device")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_tensor_cores_unroll_phi(self):
    tc = Device[Device.DEFAULT].renderer.tensor_cores[0]
    x, y = Tensor.rand(128, 128, dtype=tc.dtype_in), Tensor.rand(128, 128, dtype=tc.dtype_in)
    r = x.matmul(y, dtype=tc.dtype_out)
    opts = [Opt(OptOps.UNROLL, 0, 4)]
    ast = helper_linearizer_opt(r, [opts], apply_tc=True, atol=3e-2, rtol=1e-3)
    for u in get_program(ast, Device[Device.DEFAULT].renderer, opts=opts).uops:
      if u.op is Ops.WMMA:
        assert u.src[-1].src[0].op != Ops.STORE

  @Context(ALLOW_TF32=1)
  @unittest.skipIf(Device.DEFAULT == "PYTHON", "slow on EMULATED device")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  @unittest.skipIf(Device.DEFAULT in {"CPU"}, "CPU does not support using a different type for accumulation")
  def test_tensor_cores_unroll_casted_phi(self):
    tc = [tc for tc in Device[Device.DEFAULT].renderer.tensor_cores if tc.dtype_in != tc.dtype_out][0]
    x, y = Tensor.rand(128, 128, dtype=tc.dtype_in), Tensor.rand(128, 128, dtype=tc.dtype_in)
    r = x.matmul(y, dtype=tc.dtype_out)
    opts = [Opt(OptOps.UNROLL, 0, 4)]
    ast = helper_linearizer_opt(r, [opts], apply_tc=True, atol=3e-2, rtol=1e-3)
    for u in get_program(ast, Device[Device.DEFAULT].renderer, opts=opts).uops:
      if u.op is Ops.WMMA:
        #assert u.src[-1].dtype == dtypes.float.vec(prod(tc.thread_local_sizes[2]))
        assert u.src[-1].src[0].op != Ops.STORE

  @Context(ALLOW_TF32=1)
  @unittest.skipIf(Device.DEFAULT == "PYTHON", "slow on EMULATED device")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  @unittest.skipIf(Device.DEFAULT in {"CPU"}, "CPU does not support using a different type for accumulation")
  def test_tensor_cores_unroll_casted_phi_with_children(self):
    # all STORE children are outside the loop
    tc = [tc for tc in Device[Device.DEFAULT].renderer.tensor_cores if tc.dtype_in != tc.dtype_out][0]
    x, y = Tensor.rand(128, 128, dtype=tc.dtype_in), Tensor.rand(128, 128, dtype=tc.dtype_in)
    r = x.matmul(y, dtype=tc.dtype_out).relu()
    opts = [Opt(OptOps.UNROLL, 0, 4)]
    ast = helper_linearizer_opt(r, [opts], apply_tc=True, atol=3e-2, rtol=1e-3)
    for u in get_program(ast, Device[Device.DEFAULT].renderer, opts=opts).uops:
      if u.op is Ops.WMMA:
        #assert u.src[-1].dtype == dtypes.float.vec(prod(tc.thread_local_sizes[2]))
        assert u.src[-1].src[0].op != Ops.STORE

if __name__ == '__main__':
  unittest.main()
