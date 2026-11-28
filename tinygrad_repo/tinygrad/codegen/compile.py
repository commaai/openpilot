from typing import Sequence
from dataclasses import replace
from tinygrad.helpers import CAPTURE_PROCESS_REPLAY, colored, DEBUG, ansipad, prod, Context, to_tuple, tqdm, BEAM, ALL2ALL, size_to_str, PROFILE
from tinygrad.uop.ops import get_process_replay_loc, Ops, PatternMatcher, UOp, UPat, sym_infer, graph_rewrite, ProgramInfo, KernelInfo
from tinygrad.uop.ops import pm_flatten_linear
from tinygrad.device import Device, Buffer, DEV, HCQ_RUNTIME_DEV
from tinygrad.dtype import AddrSpace
from tinygrad.renderer import Estimates, Renderer
from tinygrad.codegen import to_program, to_program_cache, to_program_key, to_program_context, to_program_config
from tinygrad.engine.worker import get_worker_pool, terminate_worker_pool

# **** call helpers

def get_call_arg_uops(call:UOp) -> tuple[UOp, ...]: return tuple(s for s in call.src[1:] if s.addrspace is not AddrSpace.ALU)

def get_call_outs_ins(call:UOp) -> tuple[tuple[int, ...], tuple[int, ...]]:
  ast = call.body
  if call.arg.hcq is not None: return (), ()
  if ast.op is Ops.PROGRAM: return tuple(ast.arg.outs), tuple(ast.arg.ins)
  if ast.op is Ops.STORE: return (0,), (1,)
  if ast.op is Ops.NATIVE and ast.arg.code == "encdec": return (0,), tuple(range(1, len(get_call_arg_uops(call))))
  return (), ()

def get_call_name(call:UOp, bufs:Sequence[Buffer|UOp], var_vals:dict[str, int]|None=None) -> str:
  def _uop_sz_to_str(uop:UOp) -> str: return size_to_str(sym_infer(prod(uop.shape) * uop.dtype.itemsize, var_vals or {}))
  def _dev_str(buf:Buffer|UOp) -> str: return ', '.join(d[:7] for d in to_tuple(buf.device))

  ast, arg_uops = call.body, get_call_arg_uops(call)
  if ast.op is Ops.PROGRAM: return ast.src[0].arg.name
  if ast.op is Ops.STORE: return colored(f"copy {_uop_sz_to_str(arg_uops[0]):>10}, {_dev_str(bufs[0]):>7s} <- {_dev_str(bufs[1]):7s}", "yellow")
  if ast.op is Ops.NATIVE and ast.arg.code == "encdec": return colored(f"enc/dec {_uop_sz_to_str(arg_uops[0])}", "yellow")
  raise NotImplementedError("get_call_name is not implemented")

def estimate_uop(call:UOp) -> Estimates:
  call = call.without_after
  if call.arg.hcq is not None: return call.arg.hcq.estimates
  if (ast:=call.body).op is Ops.PROGRAM and (est:=ast.src[0].arg.estimates): return Estimates(*map(call.bind_symbols, (est.ops, est.lds, est.mem)))
  if ast.op is Ops.STORE or (ast.op is Ops.NATIVE and ast.arg.code == "encdec"):
    return Estimates(lds=(nbytes:=prod(call.src[1].shape) * call.src[1].dtype.itemsize), mem=nbytes)
  return Estimates()

# **** parallel lowering + compilation

def _compile_kernel(x:tuple[int, tuple[UOp, Renderer], dict]) -> tuple[int, UOp]:
  with Context(**x[2]): return x[0], to_program(*x[1])

def _get_call_to_compile(c:UOp) -> tuple[UOp, Renderer]|None:
  ast = c.body
  # a PROGRAM with a ProgramInfo and a BINARY is already compiled
  if (ast.op is Ops.SINK and isinstance(ast.arg, KernelInfo)) or \
     (ast.op is Ops.PROGRAM and not (isinstance(ast.arg, ProgramInfo) and ast.src[-1].op is Ops.BINARY)):
    return ast, Device.renderer_for(Device.canonicalize(to_tuple(c.device)[0]), HCQ_RUNTIME_DEV if c.arg.hcq is not None else DEV)
  return None

def lower_and_compile(linear:UOp, verbose=True) -> UOp:
  # collect the kernels to lower and compile, deduped by their compile cache key
  if not len(ar:={c: a for c in linear.toposort() if c.op is Ops.CALL and (a:=_get_call_to_compile(c)) is not None}): return linear

  # lower and compile what's not cached, in parallel if there's a worker pool
  keys = {c: to_program_key(*a) for c, a in ar.items()}
  todo = list({keys[c]: a for c, a in ar.items() if keys[c] not in to_program_cache}.items())
  if len(todo):
    # kernels that beam search must compile in the parent, beam needs device access to time candidates

    pool = None if len(todo) == 1 or any(getattr(c.body.arg, "beam", 0) for c in ar) else get_worker_pool()
    ctx = {v.key: v.value for v in to_program_context}
    if CAPTURE_PROCESS_REPLAY: ctx["PROCESS_REPLAY_LOC"] = get_process_replay_loc()
    tasks = ((i, ast_ren, ctx) for i, (_, ast_ren) in enumerate(todo))
    try:
      with tqdm(total=len(todo), desc="compiling", disable=DEBUG<1 or not verbose) as pbar:
        for i, prg in (map if pool is None else pool.imap_unordered)(_compile_kernel, tasks):
          pbar.set_description(f"compiling {ansipad(prg.src[0].arg.name, 40)}")
          to_program_cache[todo[i][0]] = prg
          pbar.update(1)
    except KeyboardInterrupt:
      if pool is not None: terminate_worker_pool()
      raise

  # swap the compiled PROGRAMs into the calls
  return linear.substitute({c: c.replace(src=(to_program_cache[keys[c]], *c.src[1:])) for c in ar},
                           name="precompile kernels")

# **** compile linear

def _validate(call:UOp, sink:UOp) -> UOp:
  shadows = tuple(p if p.addrspace is AddrSpace.ALU else UOp.alloc((prod(p.max_shape),), p.dtype, int.from_bytes(call.key[:8], "little")+i,
                  device=("CPU",)*len(p.device) if isinstance(p.device, tuple) else "CPU") for i,p in enumerate(call.src[1:]))
  copies = tuple(s.store_call(p) for s, p in zip(shadows, call.src[1:]) if p.addrspace is not AddrSpace.ALU)
  return UOp(Ops.LINEAR, src=copies + (call, UOp.native("validate", sink).call(*shadows, *call.src[1:])))
pm_validate = PatternMatcher([(UPat(Ops.CALL, src=(UPat(Ops.SINK, name="sink"),), name="call", allow_any_len=True), _validate)]) + pm_flatten_linear

# ctx is beam value
pm_beam = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.SINK, name="sink"),), name="call", allow_any_len=True),
   lambda ctx,call,sink: call.replace(src=(sink.replace(arg=replace(sink.arg, beam=ctx)), *call.src[1:])) if sink.arg.beam == 0 else None),
])

from tinygrad.runtime.support.hcq2 import hcq_compile # noqa: E402 # down here, hcq2 imports compile

compile_cache:dict[tuple, UOp] = {}
def compile_linear(linear:UOp, beam:int|None=None, validate=False, profile:bool|None=None, cache=False) -> UOp:
  beam, profile = BEAM.value if beam is None else beam, bool(PROFILE or DEBUG >= 2) if profile is None else profile
  key = (linear, beam, bool(validate), profile, ALL2ALL >= 1, tuple(DEV.value), tuple(HCQ_RUNTIME_DEV.value), *[x.value for x in to_program_config])
  if cache and (hit:=compile_cache.get(key)) is not None: return hit
  scalars = [e for e in linear.toposort() if e.op is Ops.EXTERN and e.is_variable and e.arg.name is None] # they bind by slot at run
  ret = linear.substitute({e: UOp.variable(e.expr, e.vmin, e.vmax, dtype=e.dtype) for e in scalars}, walk=True)
  if validate: ret = graph_rewrite(ret, pm_validate, name="validate", walk=True)
  if beam >= 1: ret = graph_rewrite(ret, pm_beam, ctx=beam, walk=True)
  ret = hcq_compile(lower_and_compile(ret), profile)
  if cache and ret is not linear: compile_cache[key] = ret
  return ret
