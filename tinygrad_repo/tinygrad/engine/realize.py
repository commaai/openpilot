from __future__ import annotations
from typing import cast, Iterator, Any, Sequence
import weakref, decimal, array
from dataclasses import dataclass, replace, field
from tinygrad.helpers import colored, DEBUG, GlobalCounters, ansipad, prod, flatten, Context, to_tuple, tqdm, dedup
from tinygrad.helpers import BEAM, size_to_str, time_to_str, VALIDATE_WITH_CPU, PROFILE, ProfilePointEvent, cpu_events, perf_counter_us, cpu_profile
from tinygrad.uop.ops import Ops, PatternMatcher, UOp, UPat, AxisType, sym_infer, graph_rewrite, ProgramInfo
from tinygrad.device import Device, Buffer, MultiBuffer, ProfileGraphEntry
from tinygrad.renderer import Estimates, Renderer
from tinygrad.codegen import to_program, to_program_cache, to_program_key, to_program_context
from tinygrad.engine.worker import get_worker_pool, terminate_worker_pool

# **************** Helpers ****************

def get_call_arg_uops(call:UOp) -> tuple[UOp, ...]: return tuple(s for s in call.src[1:] if not s.is_bound_var)
def get_call_var_uops(call:UOp, prg:UOp) -> list[UOp]:
  bound = {s.src[0].expr: s.src[1].src[1] for s in call.src[1:] if s.is_bound_var}
  return [bound.get(v.expr, v) for v in prg.arg.vars]

def get_call_outs_ins(call:UOp) -> tuple[tuple[int, ...], tuple[int, ...]]:
  ast = call.body
  if isinstance(call.arg.aux, HCQInfo): return (), ()
  if ast.op is Ops.PROGRAM: return tuple(ast.arg.outs), tuple(ast.arg.ins)
  if ast.op is Ops.COPY: return (0,), (1,)
  if ast.op is Ops.CUSTOM_FUNCTION and ast.arg == "encdec": return (0,), tuple(range(1, len(get_call_arg_uops(call))))
  return (), ()

def get_call_written_bufs(call:UOp) -> list[UOp]:
  if isinstance(call.arg.aux, HCQInfo): return list(call.arg.aux.written_bufs)
  arg_uops, (outs, ins) = get_call_arg_uops(call), get_call_outs_ins(call)
  bufs = [b.src[0].storage_base if (b:=arg_uops[k].storage_base).op is Ops.MSELECT else b for k in outs if k not in ins]
  return dedup([b for b in bufs if b.op is Ops.BUFFER])

def get_call_kernels(call:UOp) -> list[tuple[str, UOp, tuple[str, Estimates, bytes]|None]]:
  if isinstance(call.arg.aux, HCQInfo): # the submitter itself, then every kernel it enqueues
    kernels:list[tuple[str, UOp, tuple[str, Estimates, bytes]|None]] = [(Device[call.arg.aux.device[0]].host, call, None)]
    return kernels + [(d, call, (name, estimates, profile_key)) for devices,name,estimates,_,profile_key in call.arg.aux.kernels for d in devices]
  ast = call.body
  if ast.op is Ops.CUSTOM_FUNCTION and ast.arg == "graph": return [(to_tuple(ast.device)[0], call, None)]
  if ast.op is Ops.CUSTOM_FUNCTION and ast.arg == "validate": return []
  return [(d, call, None) for d in to_tuple(call.src[1].device)]

def get_call_name(call:UOp, bufs:Sequence[Buffer|UOp], var_vals:dict[str, int]|None=None) -> str:
  def _uop_sz_to_str(uop:UOp) -> str: return size_to_str(sym_infer(prod(uop.shape) * uop.dtype.itemsize, var_vals or {}))
  def _dev_str(buf:Buffer|UOp) -> str: return ', '.join(d[:7] for d in to_tuple(buf.device))

  ast, arg_uops = call.body, get_call_arg_uops(call)
  if ast.op is Ops.PROGRAM: return ast.src[0].arg.name
  if ast.op is Ops.COPY: return colored(f"copy {_uop_sz_to_str(arg_uops[0]):>10}, {_dev_str(bufs[0]):>7s} <- {_dev_str(bufs[1]):7s}", "yellow")
  if ast.op is Ops.CUSTOM_FUNCTION and ast.arg == "encdec": return colored(f"enc/dec {_uop_sz_to_str(arg_uops[0])}", "yellow")
  if ast.op is Ops.CUSTOM_FUNCTION and ast.arg == "graph": return colored(f"batched {len(ast.src[0].src)}", "cyan")
  raise NotImplementedError("get_call_name is not implemented")

# **************** Stat ****************

def estimate_uop(call:UOp) -> Estimates:
  call = call.without_after
  if isinstance(call.arg.aux, HCQInfo): return call.arg.aux.estimates
  if (ast:=call.body).op is Ops.PROGRAM: return ast.src[0].arg.estimates or Estimates()
  if ast.op is Ops.COPY or (ast.op is Ops.CUSTOM_FUNCTION and ast.arg == "encdec"):
    return Estimates(lds=(nbytes:=prod(call.src[1].shape) * call.src[1].dtype.itemsize), mem=nbytes)
  if ast.op is Ops.CUSTOM_FUNCTION and ast.arg == "graph": return get_graph_runtime(ast).estimates
  return Estimates()

first_run_cache:set[bytes] = set()
def track_stats(ctx:ExecContext, call:UOp, st:decimal.Decimal, ets:list[float|None]):
  if ctx.update_stats:
    is_hcq = isinstance(call.arg.aux, HCQInfo)
    estimates, n = estimate_uop(call), 1 if is_hcq else len(get_call_kernels(call))
    GlobalCounters.kernel_count += len(call.arg.aux.kernels) if is_hcq else n
    GlobalCounters.global_ops += n*sym_infer(estimates.ops, ctx.var_vals)
    GlobalCounters.global_mem += n*sym_infer(estimates.mem, ctx.var_vals)
    GlobalCounters.time_sum_s += sum(et for et in ets if et is not None)
  if DEBUG < 2 and not PROFILE: return

  kernels = get_call_kernels(call) # everything below is the per kernel display: exec events for the profiler and DEBUG=2 lines
  args = resolve_params(call, ctx.input_uops) if kernels and kernels[0][2] is None else []
  lanes = list(unwrap_multi(call, [args[g] for g in call.body.arg.globals] if call.body.op is Ops.PROGRAM else args)) if args else []
  for i, (device, kcall, stats) in enumerate(kernels):
    et, bufs = ets[i] if i < len(ets) else None, lanes[i][0] if i < len(lanes) else []
    display_name = get_call_name(kcall, bufs, ctx.var_vals) if stats is None else stats[0]
    if PROFILE: # backdate the event to the start of the call, the viz matches a device range with the exec event before it
      outputs, inputs = get_call_outs_ins(kcall)
      cpu_events.append(ProfilePointEvent(device, "exec", len(cpu_events), {"var_vals": ctx.var_vals,
        "bufs": [b.trace_num for b in bufs], "name": display_name, "outputs": outputs, "inputs": inputs}, ts=st))
    if DEBUG < 2 or not ctx.update_stats: continue
    if et is None and not getattr(call.arg.aux, "skip_wait", False):
      Device[device].synchronize()
      et, st = float(perf_counter_us() - st)*1e-6, perf_counter_us()
      GlobalCounters.time_sum_s += et

    estimates = estimate_uop(kcall) if stats is None else stats[1]
    op_est, mem_est, lds_est = (sym_infer(x, ctx.var_vals) for x in (estimates.ops, estimates.mem, estimates.lds))
    key = kcall.body.key if stats is None else stats[2]
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
    runtime = Device[device].runtime(ast.to_elf())
    if cache: runtime_cache[key] = runtime
  return runtime

graph_cache:weakref.WeakKeyDictionary[UOp, Any] = weakref.WeakKeyDictionary()
def get_graph_runtime(ast:UOp, input_uops:tuple[UOp, ...]|None=None):
  assert ast.op is Ops.CUSTOM_FUNCTION and ast.arg == "graph", "get_graph_runtime should only be called with a graph ast"
  if (runtime:=graph_cache.get(ast)) is None and input_uops is not None:
    graph_cache[ast] = runtime = Device[ast.device if isinstance(ast.device, str) else ast.device[0]].graph(ast, input_uops=input_uops)
  return runtime

# **************** run linear ****************

capturing: list = []  # put classes with an add_linear method in here

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
  if b.op in (Ops.MSELECT, Ops.SHRINK, Ops.BITCAST): return b.replace(src=(_resolve(b.src[0], inputs), *b.src[1:]))
  if b.op is Ops.MSTACK: return b.replace(src=tuple(_resolve(x, inputs) for x in b.src))
  return inputs[b.arg.slot] if b.op is Ops.PARAM else b
def resolve_params(call:UOp, inputs:tuple[UOp, ...]) -> list[UOp]: return [_resolve(b, inputs) for b in get_call_arg_uops(call)]

def unwrap_multi(call:UOp, resolved:list[UOp]) -> Iterator[tuple[list[Buffer], dict[str, int]]]:
  bufs = [b.buffer for b in resolved]
  if not any(isinstance(b, MultiBuffer) for b in bufs): yield cast(list[Buffer], bufs), {}
  else:
    # the DEVICE axis is bound per device at launch: it's a RANGE in the AST and the _device_num variable after codegen
    has_dnum = any((x.op is Ops.RANGE and x.arg[-1] is AxisType.DEVICE) or (x.op is Ops.PARAM and x.arg.name == '_device_num')
                   for x in call.body.toposort())
    lanes = max(len(b.bufs) for b in bufs if isinstance(b, MultiBuffer)) # a single buffer is shared by every lane
    per_lane = [b.bufs if isinstance(b, MultiBuffer) else (b,)*lanes for b in bufs]
    for j, per_dev in enumerate(zip(*per_lane)): yield list(per_dev), {"_device_num": j} if has_dnum else {}

def exec_copy(ctx:ExecContext, call:UOp, ast:UOp) -> list[float|None]:
  for bufs, device_vars in unwrap_multi(call, resolve_params(call, ctx.input_uops)):
    dest, src = bufs[0].ensure_allocated(), bufs[1].ensure_allocated()
    if hasattr(dest.allocator,'_transfer') and dest.allocator.supports_transfer and dest.device.split(":")[0] == src.device.split(":")[0]:
      dest.allocator._transfer(dest._buf, src._buf, dest.nbytes, src_dev=src.allocator.dev, dest_dev=dest.allocator.dev)
    elif src.device.startswith("DISK") and getattr(src.allocator.dev, 'fd', None) is not None \
         and hasattr(dest.allocator, 'copy_from_disk') and src.nbytes >= 4096 and dest.allocator.supports_copy_from_disk:
      dest.allocator.copy_from_disk(dest._buf, src._buf, src.nbytes)
    elif dest.get_storage().host is not None and src.get_storage().host is not None:
      for b in (dest, src): b.allocator.dev.synchronize()
      with cpu_profile(f"{src.device} -> {dest.device}", f"{src.device}:COPY"): dest.host[:] = src.host[:]
    elif dest._host_mv() is not None: src.allocator._copyout(dest.as_memoryview(allow_zero_copy=True), src._buf)
    else: dest.allocator._copyin(dest._buf, src.as_memoryview(allow_zero_copy=True))
  return []

def exec_kernel(ctx:ExecContext, call:UOp, ast:UOp, devices=None) -> list[float|None]:
  ets:list[float|None] = []
  resolved = resolve_params(call, ctx.input_uops)
  for device, (bufs, device_vars) in zip(devices or to_tuple(call.src[1].device), unwrap_multi(call, [resolved[i] for i in ast.arg.globals])):
    var_vals = {**ctx.var_vals, **device_vars}
    prg_bufs = [b.ensure_allocated() for b in bufs]
    rt = get_runtime(device, ast, cache=ctx.cache)
    global_size, local_size = ast.arg.launch_dims(var_vals)
    ets.append(rt(*[b.get_buf(device) for b in prg_bufs], global_size=global_size, local_size=local_size, vals=ast.arg.vals(var_vals),
                  wait=ctx.wait, timeout=ctx.timeout))
  return ets

def exec_validate(ctx:ExecContext, call:UOp, ast:UOp) -> list[float|None]:
  import numpy as np
  for bufs, device_vars in unwrap_multi(call, resolve_params(call, ctx.input_uops)):
    bufs, dev_bufs = bufs[:len(bufs)//2], bufs[len(bufs)//2:]
    var_vals = {**ctx.var_vals, **device_vars}
    cpu_rt = get_runtime("CPU", prg:=to_program(ast.src[0], Device["CPU"].renderer))
    global_size, local_size = prg.arg.launch_dims(var_vals)
    cpu_rt(*[bufs[i].ensure_allocated()._buf for i in prg.arg.globals], global_size=global_size, local_size=local_size, vals=prg.arg.vals(var_vals))
    for i in prg.arg.outs: np.testing.assert_allclose(dev_bufs[i].ensure_allocated().numpy(), bufs[i].numpy(), rtol=1e-3, atol=1e-3)
  return []

def exec_encdec(ctx:ExecContext, call:UOp, ast:UOp) -> list[float|None]:
  bufs = [cast(Buffer, b.buffer).ensure_allocated() for b in resolve_params(call, ctx.input_uops)]
  shape, pos_var = tuple(s.val for s in ast.src if s.op is Ops.CONST), ast.variables()[0].expr
  bufs[0].allocator._encode_decode(bufs[0]._buf, bufs[1]._buf, bufs[2]._buf, [x._buf for x in bufs[3:]], shape, ctx.var_vals[pos_var])
  return []

def exec_graph(ctx:ExecContext, call:UOp, ast:UOp) -> list[float|None]:
  return [get_graph_runtime(ast, ctx.input_uops)(ctx.input_uops, ctx.var_vals, wait=ctx.wait)]

def exec_hcq(ctx:ExecContext, call:UOp, ast:UOp) -> list[float|None]:
  if (info:=call.arg.aux).inputs:
    addrs = [cast(Buffer, _resolve(u, ctx.input_uops).buffer).get_buf(dev) + off for u, dev, off in info.inputs]
    cast(Buffer, call.src[1 + info.table].buffer).host.view(fmt='Q')[:] = array.array('Q', addrs)
  ctx = replace(ctx, wait=ctx.wait and not info.skip_wait,
                var_vals={**ctx.var_vals, **{k: v for d in info.device for k, v in cast(Any, Device[d]).var_vals.items()}})
  ets = exec_kernel(ctx, call, ast, devices=(Device[info.device[0]].host,))
  for host, dev in info.host_deps: Device[host].pending[Device[dev]] = Device[dev].timeline.host.view(fmt='Q')[1]
  if not (ctx.wait or PROFILE): return ets

  slots = {d: cast(Buffer, call.src[1 + i].buffer) for d, i in info.slots}
  for devs, name, _, prof, pkey in info.kernels:
    for d in (devs if prof else ()): cast(Any, Device[d]).prof_ents[(slots[d], prof[0])] = ProfileGraphEntry(d, name, prof[0], prof[1], pkey)
  if ctx.wait:
    for device in info.device: cast(Any, Device[device]).synchronize(timeout=ctx.timeout)
  def _prof_tm(device:str, prof:tuple[int, ...]) -> float:
    st, en = (slots[device].host.view(fmt='Q')[x] for x in prof)
    return float(en-st) / cast(Any, Device[device]).timestamp_divider / 1e6
  return ets + [_prof_tm(device, prof) if ctx.wait else None for devices, _, _, prof, _ in info.kernels if prof for device in devices]

# flatten LINEAR-in-LINEAR: any nested LINEAR child gets inlined into its parent's src
pm_flatten_linear = PatternMatcher([
  (UPat(Ops.LINEAR, custom_early_reject={Ops.LINEAR}, name="lin"),
   lambda lin: lin.replace(src=tuple(flatten(c.src if c.op is Ops.LINEAR else (c,) for c in lin.src)))),
])

def _validate(call:UOp, sink:UOp) -> UOp:
  params = get_call_arg_uops(call)
  shadows = tuple(UOp.new_buffer(("CPU",)*len(p.device) if isinstance(p.device, tuple) else "CPU", prod(p.max_shape), p.dtype) for p in params)
  copies = tuple(p.copy_to_device(s.device).call(s, p) for s, p in zip(shadows, params))
  return UOp(Ops.LINEAR, src=copies + (call, UOp(Ops.CUSTOM_FUNCTION, src=(sink,), arg="validate").call(*shadows, *params)))
pm_validate = PatternMatcher([(UPat(Ops.CALL, src=(UPat(Ops.SINK, name="sink"),), name="call", allow_any_len=True), _validate)]) + pm_flatten_linear

# ctx is beam value
pm_beam = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.SINK, name="sink"),), name="call", allow_any_len=True),
   lambda ctx,call,sink: call.replace(src=(sink.replace(arg=replace(sink.arg, beam=ctx)), *call.src[1:])) if sink.arg.beam == 0 else None),
])

# **************** parallel lowering + compilation ****************

def _compile_kernel(x:tuple[int, tuple[UOp, Renderer], dict]) -> tuple[int, UOp]:
  with Context(**x[2]): return x[0], to_program(*x[1])

def _get_call_to_compile(c:UOp) -> tuple[UOp, Renderer]|None:
  ast = c.body
  # a PROGRAM with a ProgramInfo and a BINARY is already compiled
  if ast.op is Ops.SINK or (ast.op is Ops.PROGRAM and not (isinstance(ast.arg, ProgramInfo) and ast.src[-1].op is Ops.BINARY)):
    return ast, Device[c.device if isinstance(c.device, str) else c.device[0]].renderer
  return None

def lower_and_compile(linear:UOp) -> UOp:
  # collect the kernels to lower and compile, deduped by their compile cache key
  if not len(ar:={c: a for c in linear.toposort() if c.op is Ops.CALL and (a:=_get_call_to_compile(c)) is not None}): return linear

  # lower and compile what's not cached, in parallel if there's a worker pool
  keys = {c: to_program_key(*a) for c, a in ar.items()}
  todo = list({keys[c]: a for c, a in ar.items() if keys[c] not in to_program_cache}.items())
  if len(todo):
    # kernels that beam search must compile in the parent, beam needs device access to time candidates

    pool = None if len(todo) == 1 or any(getattr(c.body.arg, "beam", 0) for c in ar) else get_worker_pool()
    ctx = {v.key: v.value for v in to_program_context}
    tasks = ((i, ast_ren, ctx) for i, (_, ast_ren) in enumerate(todo))
    try:
      with tqdm(total=len(todo), desc="compiling", disable=DEBUG<1) as pbar:
        for i, prg in (map if pool is None else pool.imap_unordered)(_compile_kernel, tasks):
          pbar.set_description(f"compiling {ansipad(prg.src[0].arg.name, 40)}")
          to_program_cache[todo[i][0]] = prg
          pbar.update(1)
    except KeyboardInterrupt:
      if pool is not None: terminate_worker_pool()
      raise

  # swap the compiled PROGRAMs into the calls
  return linear.substitute({c: c.replace(src=(c.body.substitute({a[0]: to_program_cache[keys[c]]}), *c.src[1:])) for c, a in ar.items()},
                           name="precompile kernels")

pm_exec = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.COPY, name="ast"),), name="call", allow_any_len=True), exec_copy),
  (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, name="ast"),), name="call", allow_any_len=True),
   lambda ctx, call, ast: exec_hcq(ctx, call, ast) if isinstance(call.arg.aux, HCQInfo) else exec_kernel(ctx, call, ast)),
  (UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, arg="encdec", name="ast"),), name="call", allow_any_len=True), exec_encdec),
  (UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, arg="graph", name="ast"),), name="call", allow_any_len=True), exec_graph),
  (UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, arg="validate", name="ast"),), name="call", allow_any_len=True), exec_validate),
])

from tinygrad.runtime.support.hcq2 import hcq_compile, hcq_link, HCQInfo # noqa: E402 # down here, hcq2 imports realize

def compile_linear(linear:UOp, beam:int|None=None, validate=False, input_uops:list[UOp]|None=None, profile:bool|None=None, cache=False) -> UOp:
  if validate: linear = graph_rewrite(linear, pm_validate, name="validate", walk=True)
  if (beam_val:=BEAM.value if beam is None else beam) >= 1: linear = graph_rewrite(linear, pm_beam, ctx=beam_val, walk=True)
  linear = lower_and_compile(linear)
  linear = hcq_compile(linear, input_uops, bool(PROFILE or DEBUG >= 2) if profile is None else profile, cache=cache)
  return linear

def link_linear(linear:UOp, input_uops:list[UOp]|None=None, allow_cache=True) -> UOp:
  return hcq_link(linear, input_uops=input_uops, allow_cache=allow_cache)

def run_linear(linear:UOp, var_vals:dict[str, int]|None=None, input_uops:Sequence[UOp]=(), update_stats=True, jit=False, wait=False):
  inputs = list(input_uops)
  if not jit: linear = link_linear(compile_linear(linear, validate=VALIDATE_WITH_CPU, input_uops=inputs, cache=True), input_uops=inputs)
  ctx = ExecContext(var_vals or {}, tuple(inputs), update_stats, jit, wait or DEBUG>=2)
  for call in linear.src: track_stats(ctx, call.without_after, perf_counter_us(), pm_exec.rewrite(call.without_after, ctx))

def time_call(call:UOp, var_vals:dict[str, int]|None=None, timeout:int|None=None, clear_l2:bool=False) -> Iterator[float]:
  ctx = ExecContext(var_vals or {}, update_stats=False, wait=True, timeout=timeout, cache=False)
  linear = link_linear(compile_linear(UOp(Ops.LINEAR, src=(call,)), beam=0, profile=True, cache=False), allow_cache=ctx.cache)
  while True:
    if clear_l2:
      if hasattr(dev:=Device[call.src[1].device], 'invalidate_caches'): dev.invalidate_caches()
      else:
        from tinygrad.tensor import Tensor
        with Context(DEBUG=0, BEAM=0, CAPTURING=0, TRACK_MATCH_STATS=0): Tensor.ones(1024, 1024).contiguous().realize(do_update_stats=False)
    yield max(pm_exec.rewrite(linear.src[0].without_after, ctx) or [0.0])
