import random, traceback, ctypes, argparse, os
from typing import Any
import numpy as np
from collections import defaultdict
from extra.optimization.helpers import load_worlds, ast_str_to_lin, kern_str_to_lin
from tinygrad.engine.realize import get_program

# We need to insert ioctl before opening devices.
if os.getenv("VALIDATE_HCQ", 0) != 0:
  try:
    import extra.nv_gpu_driver.nv_ioctl
    from tinygrad import Device
    _, _ = Device["NV"], Device["CUDA"]
  except Exception: pass

  try:
    import extra.qcom_gpu_driver.opencl_ioctl
    from tinygrad import Device
    _, _ = Device["QCOM"], Device["GPU"]
  except Exception: pass

from tinygrad import Tensor, Device, dtypes
from tinygrad.tensor import _to_np_dtype
from tinygrad.codegen.opt.kernel import Kernel
from tinygrad.codegen.opt.kernel import Opt, OptOps
from tinygrad.codegen.opt.search import get_kernel_actions, bufs_from_lin
from tinygrad.engine.realize import CompiledRunner
from tinygrad.helpers import getenv, from_mv, prod, colored, Context, DEBUG, Timing
from tinygrad.uop.ops import UOp, Ops
from tinygrad.device import is_dtype_supported

def on_linearizer_will_run(): pass
def on_linearizer_did_run(): pass
def compare_states(x, y): return (True, "")

if getenv("VALIDATE_HCQ"):
  if Device.DEFAULT == "NV":
    print("VALIDATE_HCQ: Comparing NV to CUDA")
    import extra.nv_gpu_driver.nv_ioctl
    validate_device = Device["CUDA"]
    on_linearizer_will_run = extra.nv_gpu_driver.nv_ioctl.before_launch
    on_linearizer_did_run = extra.nv_gpu_driver.nv_ioctl.collect_last_launch_state
    compare_states = extra.nv_gpu_driver.nv_ioctl.compare_launch_state
  elif Device.DEFAULT == "QCOM":
    print("VALIDATE_HCQ: Comparing QCOM to GPU")
    import extra.qcom_gpu_driver.opencl_ioctl
    validate_device = Device["GPU"]
    on_linearizer_will_run = extra.qcom_gpu_driver.opencl_ioctl.before_launch
    on_linearizer_did_run = extra.qcom_gpu_driver.opencl_ioctl.collect_last_launch_state
    compare_states = extra.qcom_gpu_driver.opencl_ioctl.compare_launch_state
  else:
    print(colored("VALIDATE_HCQ options is ignored", 'red'))

def tuplize_uops(uops:list[UOp]) -> tuple:
  return tuple([(x.op, x.dtype, tuple(uops.index(x) for x in x.src), x.arg) for x in uops])

def get_fuzz_rawbufs(lin):
  rawbufs = bufs_from_lin(lin)

  # Reallocate output buffer with additional area to detect out-of-bounds writes.
  RED_AREA_SIZE = 1024
  # setting output  # TODO: multi-output kernel
  rawbufs[0] = get_fuzz_rawbuf_like(rawbufs[0], zero=True, size=rawbufs[0].size+RED_AREA_SIZE)
  # setting inputs
  with Context(DEBUG=0):
    for rawbuf in rawbufs[1:]:
      if dtypes.is_unsigned(rawbuf.dtype):
        data = np.random.randint(0, 100, size=rawbuf.size, dtype=_to_np_dtype(rawbuf.dtype))
      elif dtypes.is_int(rawbuf.dtype):
        data = np.random.randint(-100, 100, size=rawbuf.size, dtype=_to_np_dtype(rawbuf.dtype))
      elif rawbuf.dtype == dtypes.bool:
        data = np.random.choice([True, False], size=rawbuf.size)
      elif rawbuf.dtype == dtypes.half:
        data = np.random.uniform(-1, 1, size=rawbuf.size).astype(dtype=_to_np_dtype(rawbuf.dtype))
      else:
        data = np.random.uniform(-10, 10, size=rawbuf.size).astype(dtype=_to_np_dtype(rawbuf.dtype))
      rawbuf.copyin(Tensor(data, device=lin.opts.device).realize().uop.base.realized.as_buffer())
  return rawbufs

def get_fuzz_rawbuf_like(old_rawbuf, zero=False, copy=False, size=None, force_device=None):
  rawbuf = type(old_rawbuf)(force_device or old_rawbuf.device, old_rawbuf.size if size is None else size, old_rawbuf.dtype).allocate()
  if copy:
    with Context(DEBUG=0): rawbuf.copyin(old_rawbuf.as_buffer())
  elif zero:
    with Context(DEBUG=0):
      mv = memoryview(bytearray(rawbuf.size * rawbuf.dtype.itemsize))
      ctypes.memset(from_mv(mv), 0, len(mv))
      rawbuf.copyin(mv)
  return rawbuf

def run_linearizer(lin: Kernel, rawbufs=None, var_vals=None) -> tuple[str, Any]: # (error msg, run state)
  if rawbufs is None: rawbufs = bufs_from_lin(lin)
  if var_vals is None: var_vals = {v: v.min for v in lin.vars}

  # TODO: images needs required_optimization
  try:
    prg = CompiledRunner(get_program(lin.get_optimized_ast(), lin.opts))
  except KeyboardInterrupt: raise
  except Exception:
    traceback.print_exc()
    return "COMPILE_ERROR", None

  if getenv("VALIDATE_HCQ"): on_linearizer_will_run()
  try:
    prg(rawbufs, var_vals, wait=True)
  except KeyboardInterrupt: raise
  except Exception:
    traceback.print_exc()
    return "EXEC_ERROR", None

  if getenv("VALIDATE_HCQ"): run_state = on_linearizer_did_run()
  else: run_state = None

  return "PASS", run_state

def compare_linearizer(lin: Kernel, rawbufs=None, var_vals=None, ground_truth=None, rtol=1e-2, atol=1e-2):
  # TODO: for bfloat16 it compiles linearizer, but it does not run because numpy cannot generate bf16 buffer.
  has_bf16 = any(b.dtype.base == dtypes.bfloat16 for b in lin.bufs)

  # TODO: raise specific fuzzing errors instead of str, and propagate the error message
  try:
    if rawbufs is None:
      rawbufs = get_fuzz_rawbufs(lin)
    else:
      rawbufs[0] = get_fuzz_rawbuf_like(rawbufs[0], zero=True) # get a new output buffer
  except KeyboardInterrupt: raise
  except BaseException:
    return ("RAWBUFS_ERROR", rawbufs, var_vals, ground_truth, None)

  if var_vals is None:
    # TODO: handle symbolic max case
    var_vals = {v: random.randint(v.vmin, v.vmax) for v in lin.ast.variables()}

  if ground_truth is None and not has_bf16:
    unoptimized = Kernel(lin.ast)
    if run_linearizer(unoptimized, rawbufs, var_vals)[0] != "PASS":
      return ("BASELINE_ERROR", rawbufs, var_vals, ground_truth, None)
    ground_truth = np.frombuffer(rawbufs[0].as_buffer(), _to_np_dtype(rawbufs[0].dtype)).copy()

  rawbufs[0] = get_fuzz_rawbuf_like(rawbufs[0], zero=True) # get a new output buffer
  run_msg, run_state = run_linearizer(lin, rawbufs, var_vals)
  if run_msg != "PASS": return (run_msg, rawbufs, var_vals, ground_truth, run_state)

  try:
    if not has_bf16:
      result = np.frombuffer(rawbufs[0].as_buffer(), _to_np_dtype(rawbufs[0].dtype))
      np.testing.assert_allclose(result, ground_truth, rtol=rtol, atol=atol)
  except KeyboardInterrupt: raise
  except AssertionError as e:
    if DEBUG >= 2:
      print(f"COMPARE_ERROR details: {e}")
      if getenv("DEBUG_VALUES") > 0:
        mismatch_indices = np.where(~np.isclose(result, ground_truth, rtol=rtol, atol=atol))
        mismatched_result = result[mismatch_indices]
        mismatched_ground_truth = ground_truth[mismatch_indices]
        for i, idx in enumerate(mismatch_indices[0]):
          print(f"mismatch at {idx=}: result={mismatched_result[i]} <> ground_truth={mismatched_ground_truth[i]}")
    return ("COMPARE_ERROR", rawbufs, var_vals, ground_truth, run_state)

  return ("PASS", rawbufs, var_vals, ground_truth, run_state)

def fuzz_linearizer(lin: Kernel, rtol=1e-2, atol=1e-2, opts_list=None):
  SEED = getenv("SEED", 42)
  random.seed(SEED)
  np.random.seed(SEED)
  print(lin.ast)
  print(lin.colored_shape())
  seen_uops = {}
  last_lins = [lin]
  failures:defaultdict[str, list[tuple[tuple[UOp, ...], list[Opt]]]] = defaultdict(list)
  rawbufs, var_vals, ground_truth, validate_rawbufs = None, None, None, None

  FUZZ_ALL_ACTIONS = getenv("FUZZ_ALL_ACTIONS", 0)
  FUZZ_MAX_SIZE = getenv("FUZZ_MAX_SIZE", 0)
  FUZZ_IGNORE_SIMPLE_OPS = getenv("FUZZ_IGNORE_SIMPLE_OPS", 1)

  if FUZZ_MAX_SIZE > 0 and prod(lin.full_shape) > FUZZ_MAX_SIZE:
    print("skipping large kernel")
    return failures
  if FUZZ_IGNORE_SIMPLE_OPS and _is_simple(lin):
    print("skipping simple kernel")
    return failures

  test_depth = 1 if opts_list is not None else getenv("DEPTH", 1 if FUZZ_ALL_ACTIONS else 10)
  for depth in range(test_depth):
    next_lins = []
    for lin in last_lins:
      if opts_list is None: actions = get_kernel_actions(lin, include_0=False)
      else:
        actions = {}
        for oi,opts in enumerate(opts_list):
          lin2 = lin.copy()
          for o in opts: lin2.apply_opt(o)
          actions[oi] = lin2

      if not actions: continue
      if depth == 0 and getenv("FUZZ_REQUIRE_TC", 0):
        tc_acts = {i: k for k in actions.values() if k.applied_opts[0].op == OptOps.TC}
        if len(tc_acts) == 0: return failures
        else: actions = tc_acts

      test_lins = list(actions.values())
      if FUZZ_ALL_ACTIONS: print(f"testing {lin.applied_opts=} with {len(actions)} actions")
      elif opts_list is None: test_lins = [random.choice(test_lins)]

      for test_lin in test_lins:
        if not FUZZ_ALL_ACTIONS and test_lin.applied_opts: print(f"applied opts: {test_lin.applied_opts}")

        # stop if kernel uops repeat
        try: tuops = tuplize_uops(get_program(test_lin.get_optimized_ast(), test_lin.opts).uops)
        except KeyboardInterrupt: raise
        except BaseException as e:
          print(test_lin.ast)
          print(test_lin.applied_opts)
          print(e)
          failures["LINEARIZE_ERROR"].append((test_lin.ast, test_lin.applied_opts))
          continue

        if tuops in seen_uops: continue
        seen_uops[tuops] = tuple(test_lin.applied_opts)

        if not FUZZ_ALL_ACTIONS: print(test_lin.colored_shape())

        (msg, rawbufs, var_vals, ground_truth, state1) = compare_linearizer(test_lin, rawbufs, var_vals, ground_truth, rtol=rtol, atol=atol)
        if state1 is not None and validate_device is not None:
          validate_lin = test_lin.copy()
          validate_lin.opts = validate_device.renderer
          if validate_rawbufs is None:
            validate_rawbufs = [get_fuzz_rawbuf_like(x, copy=True, force_device=validate_device.device) for x in rawbufs]
          (_msg, _, _, _, state2) = compare_linearizer(validate_lin, validate_rawbufs, var_vals, ground_truth, rtol=rtol, atol=atol)

          if _msg != "PASS": failures[f"VALIDATE_DEV_{_msg}"].append((validate_lin.ast, validate_lin.applied_opts))

          ok, err_msg = compare_states(state1, state2)
          if not ok: failures["HCQ_COMPARE_FAILURE"].append((err_msg, test_lin.ast, test_lin.applied_opts, state1, state2))

        if msg != "PASS":
          print(test_lin.ast)
          print(test_lin.applied_opts)
          print(msg)
          failures[msg].append((test_lin.ast, test_lin.applied_opts))
          continue

        next_lins.append(test_lin)

    last_lins = next_lins
    if FUZZ_ALL_ACTIONS: print(f"depth={depth} total_lins={len(last_lins)} {failures=}")
  return failures

def _is_simple(lin: Kernel) -> bool:
  if len(lin.ast.src) > 1: return False
  ast:UOp = lin.ast.src[0]
  if ast.src[0].op is Ops.CAST and ast.src[0].src[0].op is Ops.LOAD: return True
  return False

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run a fuzz testing on one or more kernels", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--ast", type=str, default=None, help="the ast for the kernel to be optimized")
  parser.add_argument("--file", type=str, default=None, help="a file containing asts to be optimized, one per line")
  parser.add_argument("--beamreplay", type=str, default=None, help="replay asts and opts got from beam with CAPTURE_BEAM")
  parser.add_argument("--logfile", type=str, default=None, help="a file containing a tuple of ast and applied_opts, one per line")
  parser.add_argument("--expected-failures", type=int, default=0, help="the number of expected failed kernels")
  parser.add_argument("--rtol", type=float, default=1e-2, help="relative tolerance for numerical comparison")
  parser.add_argument("--atol", type=float, default=1e-2, help="absolute tolerance for numerical comparison")
  args = parser.parse_args()

  opts_list = None
  if args.ast is not None:
    print("loaded AST from CLI")
    ast_strs = [args.ast]
  elif args.file is not None:
    print(f"loading ASTs from file '{args.file}'")
    with open(args.file, 'r') as file:
      ast_strs = file.readlines()
  elif args.beamreplay is not None:
    print(f"loading BEAM replay from file '{args.beamreplay}'")
    with open(args.beamreplay, 'r') as file: fdata = file.readlines()
    ast_strs, opts_list = [x.split(' :: ')[0] for x in fdata if not x.startswith("#")], [x.split(' :: ')[1] for x in fdata if not x.startswith("#")]

    # dedup ast_strs and opts_list
    dct = defaultdict(list)
    for i in range(len(ast_strs)): dct[ast_strs[i]].append(eval(opts_list[i]))
    ast_strs_items = list(dct.keys())
    opts_list = [dct[c] for c in ast_strs_items]
  elif args.logfile is not None:
    print(f"loading ASTs from LOGKERNS file '{args.file}'")
    with open(args.logfile, 'r') as file:
      kern_strs = file.readlines()
      test_lins = [kern_str_to_lin(kern_str) for kern_str in kern_strs]
      ast_strs = [f"{lin.ast}" for lin in test_lins]
  else:
    print("loading ASTs from world")
    ast_strs = load_worlds(filter_reduce=False, filter_novariable=False)

  print(f"{len(ast_strs)=}")
  tested = 0
  failed_ids = []
  failures = defaultdict(list)
  seen_ast_strs = set()

  try:
    for i, ast in enumerate(ast_strs[:getenv("FUZZ_N", len(ast_strs))]):
      if (nth := getenv("FUZZ_NTH", -1)) != -1 and i != nth: continue
      if getenv("FUZZ_IMAGEONLY") and "dtypes.image" not in ast: continue
      if "dtypes.image" in ast and Device.DEFAULT not in {"GPU", "QCOM"}: continue  # IMAGE is only for GPU
      if ast in seen_ast_strs: continue
      seen_ast_strs.add(ast)

      lin = ast_str_to_lin(ast)
      if not all(is_dtype_supported(buf.dtype) for buf in lin.bufs):
        print("skipping kernel due to not supported dtype")
        continue

      with Timing(f"tested ast {i}: "):
        tested += 1
        fuzz_failures = fuzz_linearizer(lin, rtol=args.rtol, atol=args.atol, opts_list=(opts_list[i] if opts_list else None))
        if fuzz_failures: failed_ids.append(i)
        for k, v in fuzz_failures.items():
          for f in v:
            failures[k].append(f)
  except KeyboardInterrupt: print(colored("STOPPING...", 'red'))

  for msg, errors in failures.items():
    for i, payload in enumerate(errors):
      print(f"{msg} {i} kernel: {payload}") # easier to use with output with verify_kernel.py

  print(f"{tested=}")
  if failures:
    print(f"{failed_ids=}")
    for msg, errors in failures.items():
      print(f"{msg}: {len(errors)}")
    if len(failed_ids) == args.expected_failures:
      print(colored(f"{len(failed_ids)} failed as expected", "yellow"))
  if len(failed_ids) != args.expected_failures:
    print(colored(f"failed on {len(failed_ids)} kernels, expected {args.expected_failures}", "red"))
    # TODO: fix this
    # raise RuntimeError(f"failed on {len(failed_ids)} kernels, expected {args.expected_failures}")
  else:
    print(colored("all passed", "green"))
