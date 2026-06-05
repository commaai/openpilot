import numpy as np
import unittest

from tinygrad import Device, Tensor, dtypes
from tinygrad.tensor import _to_np_dtype
from tinygrad.uop.ops import Ops, UOp, buffers
from tinygrad.dtype import DType
from tinygrad.device import Buffer
from tinygrad.helpers import DEV, Context
from test.helpers import slow, replace_opts
from tinygrad.engine.realize import run_linear
from tinygrad.codegen import to_program
from tinygrad.codegen.opt import Opt, OptOps, KernelOptError
from tinygrad.codegen.opt.tc import amd_cdna_1616128

# TODO: write a clean version of this
from test.backend.test_linearizer import helper_realized_ast, helper_linearizer_opt

# NOTE: to_program always passes in Device[Device.DEFAULT].renderer explicitly for process_replay!!!

AMX = "AMX" in DEV.arch

def run_program(prg:UOp, bufs:list[Buffer]):
  buf_uops = [UOp.new_buffer(b.device, b.size, b.dtype) for b in bufs]
  for u,b in zip(buf_uops, bufs): buffers[u] = b
  run_linear(UOp(Ops.LINEAR, src=(prg.call(*buf_uops),)))

def helper_tc_ensure_uops_and_opts_count(N: int, M:int, K:int, dtype_in:DType, dtype_out:DType, axis:int=0, tc_select:int=-1, tc_opt:int=0,
                                         ensure_triggered:bool=True):
  a, b = Tensor.rand(M, K, dtype=dtype_in), Tensor.rand(K, N, dtype=dtype_in)
  r = a.matmul(b, dtype=dtype_out)
  sched = r.schedule_linear()
  realized_ast = sched.src[-1].src[0]
  opts_to_apply = [Opt(OptOps.TC, axis, (tc_select, tc_opt, 1))]

  if ensure_triggered:
    program = to_program(replace_opts(realized_ast, opts_to_apply), Device[Device.DEFAULT].renderer)
    wmmas = len([uop for uop in tuple(program.src[2].src) if uop.op is Ops.WMMA])
    tcs = len([x for x in program.src[0].arg.applied_opts if x.op is OptOps.TC])
    assert wmmas > 0, "tensor core not triggered"
    assert tcs == 1, "tensor core opt not included"
  else:
    try:
      program = to_program(replace_opts(realized_ast, opts_to_apply), Device[Device.DEFAULT].renderer)
      assert False, "OptOps.TC triggered, expected KernelOptError"
    except KernelOptError: pass

def helper_tc_allclose(N:int, M:int, K:int, dtype_in:DType, dtype_out:DType, axis:int=0, tc_select:int=-1, tc_opt:int=0, use_tensor_cores:int=1):
  a, b = Tensor.rand(M, K, dtype=dtype_in), Tensor.rand(K, N, dtype=dtype_in)
  np_a, np_b = a.numpy(), b.numpy()
  r = a.matmul(b, dtype=dtype_out)
  if dtype_in == dtypes.bfloat16: r = r.float()
  realized_ast, bufs = helper_realized_ast(r)
  opts = [Opt(op=OptOps.TC, axis=axis, arg=(tc_select, tc_opt, use_tensor_cores))]
  ast = replace_opts(realized_ast, opts)
  pu = to_program(ast, Device[Device.DEFAULT].renderer)
  if use_tensor_cores == 1: assert len([uop for uop in pu.src[2].src if uop.op is Ops.WMMA]) > 0, "wmma not triggered"
  assert len([x for x in pu.src[0].arg.applied_opts if x.op is OptOps.TC]) == 1, "tensor core opt not included"
  run_program(ast, bufs)
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
      # for AMX, tc.dims[2] == 1 so reduceop is None thus tensor_cores are not triggered
      helper_tc_allclose(tc.dims[0], tc.dims[1], 2 if AMX else tc.dims[2], tc.dtype_in, tc.dtype_out, axis=0, tc_opt=0)

  @Context(ALLOW_TF32=1)
  @unittest.skipIf(Device.DEFAULT == "PYTHON", "not generated on EMULATED device")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_tensor_cores_codegen(self):
    for tc in Device[Device.DEFAULT].renderer.tensor_cores:
      n, m, k = tc.dims[0], tc.dims[1], 2 if AMX else tc.dims[2]
      a, b = Tensor.rand(m, k, dtype=tc.dtype_in), Tensor.rand(k, n, dtype=tc.dtype_in)
      r = a.matmul(b, dtype=tc.dtype_out)
      prg = to_program(replace_opts(r.schedule_linear().src[-1].src[0],
                        [Opt(op=OptOps.TC, axis=0, arg=(-1, 2, 1))]), Device[Device.DEFAULT].renderer)
      if Device.DEFAULT == "CPU" and DEV.renderer == "LLVM":
        assert "0x201000" in prg.src[3].arg
      elif Device.DEFAULT == "AMD" and DEV.renderer == "LLVM":
        assert "@llvm.amdgcn.wmma" in prg.src[3].arg
      elif Device[Device.DEFAULT].renderer.suffix == "PTX":
        assert "mma.sync.aligned" in prg.src[3].arg
      else:
        assert "__WMMA_" in prg.src[3].arg

  @Context(ALLOW_TF32=1)
  @unittest.skipIf((Device.DEFAULT == "AMD") or (Device.DEFAULT == "PYTHON" and Device.default.renderer.target.device == "AMD"), "broken for AMD")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_tensor_cores_padded(self):
    for tc in Device[Device.DEFAULT].renderer.tensor_cores:
      helper_tc_allclose(tc.dims[0]+(pad:=1), tc.dims[1]+pad, tc.dims[2]+pad, tc.dtype_in, tc.dtype_out, tc_opt=2)

  # AMD compiler bug: AMD miscompiles non-zero padded tc kernels with -O3, producing wrong results, nans or hang (see #9606)
  # Internal bug: zero-stride dimensions combined with a mask may produce wrong index/valid for pad == 1 on AMD
  @unittest.skipUnless((Device.DEFAULT == "AMD") or (Device.DEFAULT == "PYTHON" and Device.default.renderer.target.device == "AMD"),
                       "test for AMD's tc")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  @unittest.skip("warp elements not duplicated properly across lanes")
  def test_tensor_cores_padded_amd(self):
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
      if not AMX and tc not in amd_cdna_1616128: # AMX tc.dims[2] == 1
        helper_tc_ensure_uops_and_opts_count(tc.dims[0], tc.dims[1], tc.dims[2]//8, tc.dtype_in, tc.dtype_out, tc_opt=2, ensure_triggered=False)

  @Context(ALLOW_TF32=1)
  @unittest.skipIf(Device.DEFAULT == "PYTHON", "not generated on EMULATED device")
  @slow
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_tensor_cores_multi_reduce(self):
    for tc in Device[Device.DEFAULT].renderer.tensor_cores:
      if tc.dtype_in is dtypes.bfloat16: continue # <-- broken with numpy
      # this will be a M=G16, N=G32, M=G16, M=G16, K=R16, K=R16, K=R16 with 9 choices of TC MNK axes
      golden_result = None
      for axis in range(9):
        a = Tensor.rand(16, 16, 29, 29, dtype=tc.dtype_in).realize()
        b = Tensor.rand(32, 16, 16, 16, dtype=tc.dtype_in).realize()
        c = a.conv2d(b, padding=1, dtype=tc.dtype_out)
        realized_ast, real_bufs = helper_realized_ast(c)

        ast = replace_opts(realized_ast, [Opt(OptOps.TC, axis, (-1, 2, 1))])
        program = to_program(ast, Device[Device.DEFAULT].renderer)
        assert len([uop for uop in tuple(program.src[2].src) if uop.op is Ops.WMMA]) > 0, "tensor core not triggered"
        assert len([x for x in program.src[0].arg.applied_opts if x.op is OptOps.TC]) == 1, "tensor core opt not included"

        # TODO: support this even if numpy doesn't
        if _to_np_dtype(real_bufs[0].dtype) is None: continue
        real_bufs[0].copyin(np.zeros((real_bufs[0].size, ), dtype=_to_np_dtype(real_bufs[0].dtype)).data) # Zero to check that all values are filled
        run_program(ast, real_bufs)
        result = np.frombuffer(real_bufs[0].as_memoryview(), _to_np_dtype(real_bufs[0].dtype))

        # ensure the results for each choice of axis matches
        if golden_result is None: golden_result = np.frombuffer(real_bufs[0].as_memoryview(), _to_np_dtype(real_bufs[0].dtype))
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
    for u in tuple(to_program(replace_opts(ast, opts), Device[Device.DEFAULT].renderer).src[2].src):
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
    for u in tuple(to_program(replace_opts(ast, opts), Device[Device.DEFAULT].renderer).src[2].src):
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
    for u in tuple(to_program(replace_opts(ast, opts), Device[Device.DEFAULT].renderer).src[2].src):
      if u.op is Ops.WMMA:
        #assert u.src[-1].dtype == dtypes.float.vec(prod(tc.thread_local_sizes[2]))
        assert u.src[-1].src[0].op != Ops.STORE

if __name__ == '__main__':
  unittest.main()
