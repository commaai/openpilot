from typing import cast, Optional, Callable
import itertools, functools, random, math, time, multiprocessing, traceback, signal, atexit
from collections import defaultdict
from dataclasses import replace
from tinygrad.ops import UOp, Ops, Variable, sym_infer
from tinygrad.device import Device, Buffer, Compiler
from tinygrad.helpers import prod, flatten, DEBUG, CACHELEVEL, diskcache_get, diskcache_put, getenv, Context, colored, time_to_str
from tinygrad.helpers import IGNORE_BEAM_CACHE, TC_SEARCH_OVER_SHAPE
from tinygrad.dtype import ImageDType, PtrDType
from tinygrad.codegen.kernel import Kernel, Opt, OptOps, KernelOptError
from tinygrad.tensor import Tensor
from tinygrad.engine.realize import CompiledRunner
from tinygrad.renderer import ProgramSpec

actions = [Opt(op=OptOps.UPCAST, axis=axis, arg=amt) for amt in [0,2,3,4,5,7] for axis in range(6)]
actions += [Opt(op=OptOps.UNROLL, axis=axis, arg=amt) for amt in [0,4,7] for axis in range(5)]
actions += [Opt(op=OptOps.LOCAL, axis=axis, arg=amt) for amt in [2,3,4,8,13,16,29] for axis in range(6)]
actions += [Opt(op=OptOps.GROUPTOP, axis=axis, arg=amt) for amt in [13,16,28,29,32,49,64,256] for axis in range(3)]
actions += [Opt(op=OptOps.GROUP, axis=axis, arg=amt) for amt in [0,4,8,16] for axis in range(3)]
if getenv("BEAM_PADTO", 1): actions += [Opt(op=OptOps.PADTO, axis=axis, arg=amt) for amt in [32] for axis in range(7)]
actions += [Opt(op=OptOps.LOCAL, axis=0, arg=32), Opt(op=OptOps.LOCAL, axis=6, arg=2)]
actions += [Opt(op=OptOps.TC, axis=0, arg=(-1, 0))]
actions += [Opt(op=OptOps.TC, axis=axis, arg=(-1, getenv("TC_OPT", 2))) for axis in range(9)] # covers resnet kernels (3 global * 3 reduce)
actions += [Opt(op=OptOps.SWAP, axis=axis_0, arg=axis_1) for axis_0 in range(5) for axis_1 in range(axis_0+1, 5)]
if getenv("NOLOCALS"): actions += [Opt(op=OptOps.NOLOCALS)]

def _get_test_global_size(global_size, max_global_size, var_vals):
  test_global_size, factor = [sym_infer(sz, var_vals) for sz in global_size], 1
  while prod(test_global_size) > max_global_size:
    for j in range(len(global_size)-1,-1,-1):
      if test_global_size[j] > 16:
        test_global_size[j] //= 2
        factor *= 2
        break
  return test_global_size, factor

def _time_program(p:ProgramSpec, lib:bytes, var_vals:dict[Variable, int], rawbufs:list[Buffer], early_stop:Optional[float]=None,
                  max_global_size:Optional[int]=65536, clear_l2=False, cnt=3, name="test") -> list[float]:
  factor = 1
  if p.global_size is not None and max_global_size is not None:
    global_size, factor = _get_test_global_size(p.global_size, max_global_size, var_vals)
    p = replace(p, global_size=global_size)
  try: car = CompiledRunner(p, precompiled=lib)
  except AssertionError: return [math.inf] * cnt
  tms = []
  input_bufs = [rawbufs[i] for i in car.p.globals]
  for _ in range(cnt):
    if clear_l2:
      if hasattr(dev:=Device[p.device], 'invalidate_caches'): dev.invalidate_caches()
      else:
        with Context(DEBUG=0, BEAM=0, CAPTURING=0, TRACK_MATCH_STATS=0): Tensor.ones(1024,1024).contiguous().realize(do_update_stats=False)
    tms.append(cast(float, car(input_bufs, var_vals, wait=True))*factor)
    if early_stop is not None and early_stop < min(tms): break
  return tms

class TimeoutException(Exception): pass
def timeout_handler(signum, frame): raise TimeoutException()

def _try_compile_linearized_w_idx(x:tuple[int,Kernel], compiler:Compiler) -> tuple[int, Optional[tuple[ProgramSpec, bytes, float]]]:
  if hasattr(signal, "alarm"):
    signal.signal(getattr(signal, 'SIGALRM'), timeout_handler)
    # set timeout
    signal.alarm(getenv("BEAM_TIMEOUT_SEC", 10))
  ret = None
  try:
    p = x[1].to_program(name_override="test")
    assert p.uops is not None, "uop list wasn't generated?"
    if len(p.uops) >= getenv("BEAM_UOPS_MAX", 3000) > 0: raise RuntimeError("too many uops")
    st = time.perf_counter()
    prog = compiler.compile(p.src)
    et = time.perf_counter() - st
    ret = (p, prog, et)
  except RuntimeError:
    if DEBUG >= 4: traceback.print_exc()
  except Exception as e:
    if getenv("BEAM_STRICT_MODE"): raise e
  finally:
    if hasattr(signal, "alarm"): signal.alarm(0)
  return x[0], ret

# workers should ignore ctrl c
def _init_worker(): signal.signal(signal.SIGINT, signal.SIG_IGN)

def _ensure_buffer_alloc(bufs:list[Buffer]) -> list[Buffer]: return [buf.ensure_allocated() for buf in bufs]

# *** external API ***

# get (scrap) buffers for timing the linearizer
def bufs_from_lin(lin:Kernel, allocate:bool=True) -> list[Buffer]:
  bufsts: defaultdict[int, list[UOp]] = defaultdict(list)
  for x in lin.bufs:
    if x.src[0].op is Ops.DEFINE_GLOBAL: bufsts[x.src[0].arg].append(x)
  rawbufs: list[Optional[Buffer]] = [None]*len(bufsts)
  for k,lx in bufsts.items():
    buf_size = prod(dtype.shape) if isinstance(dtype:=lx[0].src[0].dtype, ImageDType) else max(y.st_arg.real_size() for y in lx)
    assert isinstance(dtype, (PtrDType, ImageDType))
    if buf_size == 0: buf_size = 1  # create a size 1 buffer if no cell is accessed in kernel. # TODO: remove from kernel input in this case.
    buf_dtype = dtype if isinstance(dtype, ImageDType) else dtype.base
    rawbufs[k] = Buffer(lin.opts.device, buf_size, buf_dtype).allocate() if allocate else Buffer(lin.opts.device, buf_size, buf_dtype)
  assert all(r is not None for r in rawbufs)
  return cast(list[Buffer], rawbufs)

# get dictionary of all possible actions
def get_kernel_actions(lin:Kernel, include_0=True) -> dict[int, Kernel]:
  acted_lins, max_up, max_lcl = {0:lin} if include_0 else {}, getenv("BEAM_UPCAST_MAX", 256), getenv("BEAM_LOCAL_MAX", 1024)
  kernel_actions = actions.copy()

  if TC_SEARCH_OVER_SHAPE and len(lin.applied_opts) == 0: # tensor core opts must be first
    for i, action in enumerate(kernel_actions):
      if action.op == OptOps.TC and (tc_arg := cast(tuple, action.arg))[0] == -1:
        # replace every tc_action with default tc with one tc_action for each available tc
        kernel_actions[i:i+1] = [Opt(op=OptOps.TC, axis=action.axis, arg=(tc_select, tc_arg[1])) for tc_select,_ in enumerate(lin.opts.tensor_cores)]

  for i,a in enumerate(kernel_actions):
    if a.axis is not None and a.op is not OptOps.TC:
      if ((ax:=lin.real_axis(a)) >= lin.shape_len) or (lin.full_shape[ax] == a.arg and Opt(a.op, ax, 0) in kernel_actions): continue
    lin2 = lin.copy()
    try:
      lin2.apply_opt(a)
      up, lcl, tc_up = 1, 1, prod(tc.dims)//tc.threads if (tc:=lin2.tensor_core) else 1
      for s,c in zip(lin2.full_shape, lin2.colors()):
        if c in {"magenta", "yellow"}: up *= s
        elif c in {"cyan", "green", "white"}: lcl *= s
      if up//tc_up > max_up or lcl > max_lcl: continue
      acted_lins[i+1] = lin2
    except KernelOptError: pass
  return acted_lins

beam_pool, BEAM_DEBUG = None, getenv("BEAM_DEBUG")
def beam_search(lin:Kernel, rawbufs:list[Buffer], amt:int, allow_test_size=True, disable_cache=IGNORE_BEAM_CACHE.value) -> Kernel:
  global beam_pool
  key = {"ast": lin.ast.key, "amt": amt, "allow_test_size": allow_test_size, "device": lin.opts.device, "suffix": lin.opts.suffix}
  if not disable_cache and CACHELEVEL >= 1 and (val:=diskcache_get("beam_search", key)) is not None:
    ret = lin.copy()
    for o in val[len(lin.applied_opts):]: ret.apply_opt(o)
    return ret

  beam: list[tuple[Kernel, float]] = [(lin, float("inf"))]
  seen_libs = set()

  default_parallel = multiprocessing.cpu_count() if lin.opts.device in {"CUDA", "AMD", "NV", "METAL"} else 0
  if beam_pool is None and (workers := getenv("PARALLEL", default_parallel)):
    beam_pool = multiprocessing.get_context("spawn").Pool(workers, _init_worker, (), getenv("BEAM_MAX_TASKS_PER_CHILD", 16))
    @atexit.register
    def close_pool(): beam_pool.close()

  min_progress = getenv("BEAM_MIN_PROGRESS", 0.01)/1e6
  if BEAM_DEBUG: print(f"BEAM_SEARCH:\n{lin.ast}")
  if DEBUG >= 2: print(f"   0.00s:                from   1 ->   1 actions {lin.colored_shape()}")

  try:
    rawbufs = _ensure_buffer_alloc(rawbufs)
    var_vals: dict[Variable, int] = {k:int(k.vmax+k.vmin)//2 for k in lin.ast.variables()}
    exiting, st = False, time.perf_counter()
    dev = Device[lin.opts.device]
    while not exiting:
      acted_lins: list[Kernel] = flatten([get_kernel_actions(lin, include_0=False).values() for lin,_ in beam])
      timed_lins: list[tuple[Kernel, float]] = []
      _compile_fn = functools.partial(_try_compile_linearized_w_idx, compiler=dev.compiler)
      least_compute_ops = math.inf
      for i,proc in (map(_compile_fn, enumerate(acted_lins)) if beam_pool is None else beam_pool.imap_unordered(_compile_fn, enumerate(acted_lins))):
        if proc is None: continue
        p, lib, compile_et = proc
        if lib in seen_libs: continue
        # filter out kernels that use 1000x more compute than the smallest
        least_compute_ops = min(this_compute_ops:=sym_infer(p.estimates.ops, var_vals), least_compute_ops)
        if least_compute_ops*1000 < this_compute_ops: continue
        seen_libs.add(lib)
        try: tms = _time_program(p, lib, var_vals, rawbufs, early_stop=beam[0][1]*3 if len(beam) else 1.0, clear_l2=hasattr(dev, 'invalidate_caches'))
        except RuntimeError: continue # for runtime issues
        timed_lins.append((acted_lins[i], min(tms)))
        if BEAM_DEBUG > 1: print(f"{time.perf_counter() - st:7.2f}s: {i:5d} {len(cast(list, p.uops)):5d} uops {time_to_str(compile_et, w=12)} compile/{time_to_str(timed_lins[-1][1], w=12)} run       {len(timed_lins):4d}/{len(acted_lins):4d}         {timed_lins[-1][0].colored_shape()}")  # noqa: E501
        elif DEBUG >= 2: print(f"\r{time.perf_counter() - st:7.2f}s: {time_to_str(timed_lins[-1][1], w=12)}       {len(timed_lins):4d}/{len(acted_lins):4d}         {timed_lins[-1][0].colored_shape()}\033[K", end="")  # noqa: E501

      # done
      opts = sorted(timed_lins, key=lambda x: x[1])
      exiting = len(opts) == 0 or (opts[0][1] < min_progress) or (len(beam) > 0 and ((beam[0][1]-opts[0][1]) < min_progress))
      if not exiting: beam = opts[:amt]
      elif len(opts) > 0 and opts[0][1] < beam[0][1]: beam = opts[:1]
      if DEBUG >= 2: print(f"\r{time.perf_counter() - st:7.2f}s:", colored(time_to_str(beam[0][1], w=12), "green" if exiting else None), f"from {len(acted_lins):3d} -> {len(opts):3d} actions\033[K", beam[0][0].colored_shape())  # noqa: E501
  except KeyboardInterrupt as e:
    if beam_pool is not None: beam_pool.terminate()
    raise e

  if CACHELEVEL >= 1: diskcache_put("beam_search", key, beam[0][0].applied_opts)
  if BEAM_DEBUG: print(f"BEAM_SEARCH: final tm={time_to_str(beam[0][1], w=0)}, applied_opts={beam[0][0].applied_opts}")
  return beam[0][0]

def optimize_local_size(_prg:Callable, global_size:list[int], rawbufs:list[Buffer]) -> list[int]:
  test_rawbuffers = [Buffer(rawbufs[0].device, rawbufs[0].size, rawbufs[0].dtype).allocate(), *rawbufs[1:]] if rawbufs[0] in rawbufs[1:] else rawbufs
  MAX_WORKGROUP = 1024
  local_dims = [[x for x in set([sz, 1, 2, 4, 8, 16, 32, 64, 128, 256, MAX_WORKGROUP]) if x<=sz] for sz in global_size]
  local_sizes = [list(x) for x in itertools.product(*local_dims) if prod(x) <= MAX_WORKGROUP] * 2  # try each valid size twice
  def try_exec(local_size):
    try: return _prg(*[x._buf for x in test_rawbuffers], global_size=[g//l if g%l == 0 else g/l for g,l in zip(global_size, local_size)], local_size=local_size, wait=True)  # noqa: E501
    except Exception: return float('inf')
  ret = min([(try_exec(local_size), local_size) for local_size in random.sample(local_sizes, len(local_sizes))])
  assert not math.isinf(ret[0]), "all optimize_local_size exec failed"
  return ret[1]
