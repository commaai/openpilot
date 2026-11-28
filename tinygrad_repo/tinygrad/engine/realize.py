from __future__ import annotations
from typing import cast, Iterator, Any, Sequence
import decimal
from dataclasses import dataclass, replace, field
from tinygrad.helpers import colored, DEBUG, GlobalCounters, ansipad, Context, to_tuple, unwrap
from tinygrad.helpers import time_to_str, PROFILE, ProfilePointEvent, cpu_events, perf_counter_us, cpu_profile
from tinygrad.uop.ops import Ops, PatternMatcher, UOp, UPat, sym_infer
from tinygrad.device import Device, Buffer, ProfileGraphEntry, HOST_DEVS
from tinygrad.dtype import AddrSpace
from tinygrad.renderer import TinyELF
from tinygrad.codegen import to_program
from tinygrad.codegen.compile import compile_linear, get_call_arg_uops, get_call_outs_ins, get_call_name, estimate_uop
from tinygrad.runtime.support.hcq2 import hcq_link

# **************** Helpers ****************

def get_call_kernel_args(call:UOp, prg:UOp, devs:tuple[str, ...]) -> list[UOp]:
  return [call.scalar_arg(p).ccast(p.dtype) if p.addrspace is AddrSpace.ALU else
          call.src[1+p.arg.slot].getaddr(devs) for p in prg.kernel_params]

def get_call_kernels(call:UOp) -> list[tuple[str, UOp, tuple|None]]:
  if call.arg.hcq is not None: # the submitter itself, then every kernel it enqueues
    kernels:list[tuple[str, UOp, tuple|None]] = [(Device[call.arg.hcq.device[0]].host, call, None)]
    return kernels + [(d, call, (name, estimates, key, bufs, io)) for d,name,estimates,_,key,bufs,io in call.arg.hcq.kernels]
  ast = call.body
  if ast.op is Ops.NATIVE and ast.arg.code == "validate": return []
  return [(d, call, None) for d in to_tuple(call.device if ast.op is Ops.STORE else get_call_arg_uops(call)[0].device)]

# **************** Stat ****************

first_run_cache:set[bytes] = set()
def track_stats(ctx:ExecContext, call:UOp, st:decimal.Decimal, ets:list[float|None]):
  if ctx.update_stats:
    is_hcq = call.arg.hcq is not None
    estimates, n = estimate_uop(call), 1 if is_hcq else len(get_call_kernels(call))
    GlobalCounters.kernel_count += len(call.arg.hcq.kernels) if is_hcq else n
    GlobalCounters.global_ops += n*sym_infer(estimates.ops, ctx.var_vals)
    GlobalCounters.global_mem += n*sym_infer(estimates.mem, ctx.var_vals)
    GlobalCounters.time_sum_s += sum(et for et in ets if et is not None)
  if DEBUG < 2 and not PROFILE: return

  kernels = get_call_kernels(call) # everything below is the per kernel display: exec events for the profiler and DEBUG=2 lines
  args = [] if call.arg.hcq is not None else get_call_arg_uops(call)
  if args and call.body.op is Ops.PROGRAM: args = [call.src[1+g] for g in call.body.arg.globals]
  for i, (device, kcall, stats) in enumerate(kernels):
    et = ets[i] if i < len(ets) else None
    bufs = [_buffer(b, ctx.input_uops) for b in args] if stats is None else [cast(Buffer, ctx.input_uops[s].buffer) for s in stats[3]]
    display_name, estimates, key = (get_call_name(kcall, bufs, ctx.var_vals), estimate_uop(kcall), kcall.body.key) if stats is None else stats[:3]
    op_est, mem_est, lds_est = (sym_infer(x, ctx.var_vals) for x in (estimates.ops, estimates.mem, estimates.lds))
    if PROFILE: # backdate the event to the start of the call, the viz matches a device range with the exec event before it
      outputs, inputs = get_call_outs_ins(kcall) if stats is None else stats[4]
      cpu_events.append(ProfilePointEvent(device, "exec", len(cpu_events), {"program": key, "estimates": (op_est, mem_est, lds_est),
        "bufs": [b.trace_num for b in bufs], "name": display_name, "outputs": outputs, "inputs": inputs}, ts=st))
    if DEBUG < (3 if stats is None and call.arg.hcq is not None else 2) or not ctx.update_stats: continue
    if et is None and not (call.arg.hcq is not None and call.arg.hcq.skip_wait):
      Device[device].synchronize()
      et, st = float(perf_counter_us() - st)*1e-6, perf_counter_us()
      GlobalCounters.time_sum_s += et

    header_color = 'magenta' if ctx.jit else ('green' if key not in first_run_cache else None)
    ptm = colored(time_to_str(et, w=9), "yellow" if et > 0.01 else None) if et is not None else ""
    flops, membw, ldsbw = op_est/(et or 1e-20), mem_est/(et or 1e-20), lds_est/(et or 1e-20)
    flops_str = f"{flops*1e-9:7.0f} GFLOPS" if flops < 1e14 else colored(f"{flops*1e-12:7.0f} TFLOPS", 'green')
    mem_str = f"{membw*1e-9:4.0f}|{ldsbw*1e-9:<6.0f} GB/s" if membw < 1e13 and ldsbw < 1e15 else \
      colored(f"{membw*1e-12:4.0f}|{ldsbw*1e-12:<6.0f} TB/s", 'green')
    print(f"{colored(f'*** {device[:7]:7s} {GlobalCounters.kernel_count:4d}', header_color)}"+
      f" {ansipad(display_name, 46)} arg {len(bufs):2d} mem {GlobalCounters.mem_used/1e9:6.2f} GB"+
      ("" if et is None else f" tm {ptm}/{GlobalCounters.time_sum_s*1e3:9.2f}ms ({flops_str} {mem_str})"))
    first_run_cache.add(key)

# **************** runtime cache ****************

runtime_cache: dict[tuple[bytes, str], Any] = {}
def get_runtime(device:str, ast:UOp, cache=True):
  if (runtime:=runtime_cache.get(key:=(ast.key, device))) is None:
    links = {call.body.src[0].arg.function_name:get_runtime(device, call.body, cache)
             for call in ast.src[1].src if call.op is Ops.CALL and call.body.op is Ops.PROGRAM}
    runtime = Device[device].runtime(TinyELF.from_program(ast), links=links)
    if cache: runtime_cache[key] = runtime
  return runtime

# **************** run linear ****************

capturing: list = []  # put classes with an add_export method in here

@dataclass
class ExecContext:
  var_vals: dict[str, int] = field(default_factory=dict)
  input_uops: tuple[UOp, ...] = ()
  update_stats: bool = True
  jit: bool = False
  wait: bool = False
  timeout: int|None = None
  cache: bool = True

def _resolve(b:UOp, inputs:tuple[UOp, ...]) -> UOp:
  if b.op in (Ops.MSELECT, Ops.SHRINK, Ops.BITCAST, Ops.GETADDR): return b.replace(src=(_resolve(b.src[0], inputs), *b.src[1:]))
  if b.op is Ops.MSTACK: return b.replace(src=tuple(_resolve(x, inputs) for x in b.src))
  return inputs[b.arg.slot] if b.op is Ops.PARAM else b
def _buffer(b:UOp, inputs:tuple[UOp, ...]) -> Buffer: return cast(Buffer, _resolve(b, inputs).buffer)

def exec_copy(ctx:ExecContext, call:UOp, ast:UOp) -> list[float|None]:
  dest, src = [_buffer(b, ctx.input_uops).ensure_allocated() for b in get_call_arg_uops(call)[:2]]
  if dest.get_storage().host is not None and src.get_storage().host is not None:
    for b in (dest, src): b.allocator.dev.synchronize()
    with cpu_profile(f"{src.device} -> {dest.device}", f"{src.device}:COPY"): dest.host[:] = src.host[:]
  elif dest._host_mv() is not None: src.allocator._copyout(dest.as_memoryview(allow_zero_copy=True), src._buf)
  else: dest.allocator._copyin(dest._buf, src.as_memoryview(allow_zero_copy=True))
  return []

def exec_kernel(ctx:ExecContext, call:UOp, ast:UOp, device:str|None=None) -> list[float|None]:
  if device is None:
    device = to_tuple(get_call_arg_uops(call)[0].device)[0]
    if device.split(":")[0] in HOST_DEVS: Device[device].synchronize(ctx.timeout)
  prg_bufs = {i:_buffer(call.src[1+i], ctx.input_uops).ensure_allocated() for i in ast.arg.globals}
  rt = get_runtime(device, ast, cache=ctx.cache)
  def scalar(a:UOp): return ctx.var_vals[a.expr] if a.is_variable else _resolve(a, ctx.input_uops).val
  try:
    args = [scalar(call.scalar_arg(p)) if p.addrspace is AddrSpace.ALU else prg_bufs[p.arg.slot].get_buf(device) for p in ast.kernel_params]
    global_sz, local_sz = (tuple(sym_infer(x, ctx.var_vals) for x in dims) for dims in ast.arg.launch_dims(call))
  except KeyError as e: raise RuntimeError(f"unbound Variable {e}") from None
  return [rt(*args, global_size=global_sz, local_size=local_sz, wait=ctx.wait, timeout=ctx.timeout)]

def exec_validate(ctx:ExecContext, call:UOp, ast:UOp) -> list[float|None]:
  import numpy as np
  from tinygrad.dtype import _to_np_dtype
  Device["CPU"].synchronize(ctx.timeout)
  exec_kernel(ctx, call, prg:=to_program(ast.src[0], Device["CPU"].renderer)) # the shadows sit at the call's slots, so it runs like any call
  pos = [k for k, s in enumerate(call.src[1:]) if s.addrspace is not AddrSpace.ALU]
  args = [_buffer(b, ctx.input_uops) for b in get_call_arg_uops(call)]
  bufs, dev_bufs = dict(zip(pos, args[:len(args)//2])), dict(zip(pos, args[len(args)//2:]))
  for i in prg.arg.outs:
    dt = _to_np_dtype(call.src[1+i].dtype)
    assert dt is not None, f"no np dtype for {call.src[1+i].dtype}"
    np.testing.assert_allclose(np.frombuffer(dev_bufs[i].ensure_allocated().as_memoryview(), dt),
                               np.frombuffer(bufs[i].as_memoryview(), dt), rtol=1e-3, atol=1e-3)
  return []

def exec_hcq(ctx:ExecContext, call:UOp, ast:UOp) -> list[float|None]:
  info = call.arg.hcq
  ctx = replace(ctx, wait=ctx.wait and not info.skip_wait,
                var_vals={**ctx.var_vals, **{k: v for d in info.device for k, v in Device[d].var_vals.items()}})
  slots = {d: cast(Buffer, call.src[1 + i].buffer) for d, i in info.slots}
  if PROFILE:
    for d in {d for d, _, _, prof, *_ in info.kernels if prof and (slots[d], prof[0]) in Device[d].prof_ents}: Device[d].synchronize()
  ets = exec_kernel(ctx, call, ast, device=Device[info.device[0]].host)
  if not (ctx.wait or PROFILE): return ets

  for d, name, _, prof, pkey, *_ in info.kernels:
    if prof: Device[d].prof_ents[(slots[d], prof[0])] = ProfileGraphEntry(d, name, prof[0], prof[1], pkey)
  if ctx.wait:
    for device in info.device: Device[device].synchronize(timeout=ctx.timeout)
  def _prof_tm(device:str, prof:tuple[int, ...]) -> float:
    st, en = (slots[device].host.view(fmt='Q')[x] for x in prof)
    return float(en-st) / Device[device].timestamp_divider / 1e6
  return ets + [_prof_tm(d, prof) if ctx.wait else None for d, _, _, prof, *_ in info.kernels if prof]

pm_exec = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.STORE, name="ast"),), name="call", allow_any_len=True), exec_copy),
  (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, name="ast"),), name="call", allow_any_len=True),
   lambda ctx, call, ast: exec_hcq(ctx, call, ast) if call.arg.hcq is not None else exec_kernel(ctx, call, ast)),
  (UPat(Ops.CALL, src=(UPat.native("validate", name="ast"),), name="call", allow_any_len=True), exec_validate),
])

def devices_of(linear:UOp) -> list[str]:
  return sorted({d for u in linear.toposort() if u.op in {Ops.BUFFER, Ops.PARAM, Ops.ALLOC, Ops.EXTERN} for d in to_tuple(u.device) if d is not None})

def link_linear(linear:UOp, input_uops:list[UOp]|None=None) -> UOp:
  # link opens the devices the linear uses, opening a device registers its link buffers and rules
  for d in devices_of(linear): Device[d]
  return hcq_link(linear, input_uops=input_uops)

def run_linear(linear:UOp, var_vals:dict[str, int]|None=None, input_uops:Sequence[UOp]=(), update_stats=True, jit=False, wait=False):
  assert linear.op is Ops.LINEAR, f"run_linear calls on {linear.op}"
  if len(linear.src) == 0: return
  ctx = ExecContext(var_vals or {}, tuple(input_uops), update_stats, jit, wait or DEBUG>=2)
  for call in linear.src: track_stats(ctx, call.without_after, perf_counter_us(), pm_exec.rewrite(call.without_after, ctx))

def time_call(call:UOp, var_vals:dict[str, int]|None=None, timeout:int|None=None, clear_l2:bool=False) -> Iterator[float]:
  ctx = ExecContext(var_vals or {}, update_stats=False, wait=True, timeout=timeout, cache=False)
  linear = link_linear(compile_linear(UOp(Ops.LINEAR, src=(call,)), beam=0, profile=True))
  while True:
    if clear_l2:
      if hasattr(dev:=Device[call.device], 'invalidate_caches'): dev.invalidate_caches()
      else:
        from tinygrad.tensor import Tensor
        with Context(DEBUG=0, BEAM=0, CAPTURING=0, TRACK_MATCH_STATS=0): Tensor.ones(1024, 1024).contiguous().realize(do_update_stats=False)
    yield max(map(unwrap, pm_exec.rewrite(c:=linear.src[0].without_after, ctx)[c.arg.hcq is not None:]))
