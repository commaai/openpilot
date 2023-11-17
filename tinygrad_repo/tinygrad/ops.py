from __future__ import annotations
import importlib, inspect, functools, pathlib
from enum import Enum, auto
from typing import TYPE_CHECKING, Union, Type, Tuple, Any, List, Optional, Dict, Callable, Mapping
from tinygrad.helpers import ansilen, prod, DEBUG, getenv, GlobalCounters, DType, colored, BEAM, NOOPT
from tinygrad.runtime.lib import RawBuffer
from tinygrad.shape.symbolic import Variable, sym_infer
from dataclasses import dataclass

# these are the llops your accelerator must implement, along with toCpu
# the Enum class doesn't work with mypy, this is static. sorry it's ugly
# NOTE: MOD, CMPLT don't have to be implemented on vectors, just scalars
# NOTE: rdna3 only has RECIP and not DIV. DIV and POW are on the chopping block
class UnaryOps(Enum): NOOP = auto(); EXP2 = auto(); LOG2 = auto(); CAST = auto(); SIN = auto(); SQRT = auto(); RECIP = auto(); NEG = auto() # noqa: E702
class BinaryOps(Enum): ADD = auto(); SUB = auto(); MUL = auto(); DIV = auto(); MAX = auto(); MOD = auto(); CMPLT = auto() # noqa: E702
class TernaryOps(Enum): MULACC = auto(); WHERE = auto() # noqa: E702
class ReduceOps(Enum): SUM = auto(); MAX = auto() # noqa: E702
class BufferOps(Enum): MEM = auto(); CONST = auto() # noqa: E702
# Ops below this line are not allowed in ASTs
class MovementOps(Enum): RESHAPE = auto(); PERMUTE = auto(); EXPAND = auto(); PAD = auto(); SHRINK = auto(); STRIDE = auto(); AS_STRIDED = auto() # noqa: E702
class LoadOps(Enum): EMPTY = auto(); RAND = auto(); CONST = auto(); FROM = auto(); CONTIGUOUS = auto(); CUSTOM = auto() # noqa: E702

Op = Union[UnaryOps, BinaryOps, ReduceOps, MovementOps, LoadOps, TernaryOps, BufferOps]
OpType = Union[Type[UnaryOps], Type[BinaryOps], Type[ReduceOps], Type[MovementOps], Type[LoadOps], Type[TernaryOps], Type[BufferOps]]

if TYPE_CHECKING:
  from tinygrad.shape.shapetracker import ShapeTracker
  from tinygrad.lazy import LazyBuffer

@dataclass(frozen=True)
class MemBuffer:
  idx: int
  dtype: DType
  st: ShapeTracker

@dataclass(frozen=True)
class ConstBuffer:
  val: Any
  dtype: DType
  st: ShapeTracker

@dataclass(frozen=True)
class ScheduleItem:
  ast: LazyOp
  out: LazyBuffer
  inputs: Tuple[LazyBuffer, ...]
  var_vals: Dict[Variable, int]

@dataclass(frozen=True)
class LazyOp:
  op: Op
  src: Tuple[Union[LazyOp, LazyBuffer], ...]
  arg: Any = None
  def __repr__(self): return f"LazyOp(op={self.op}, src={self.src}, arg={self.arg})"
  @property
  def buffers(self):
    buffers: Tuple[Union[LazyOp, LazyBuffer], ...] = ()
    try:  # NOTE: the linearizer's key function maps the buffers to ints, and LOCAL_BUFFER is used. we don't care about buffers in these cases
      for x in self.src: buffers += x.buffers
    except AttributeError: buffers = ()
    return buffers

  @property
  def key(self): return (self.op, tuple(map(lambda x: getattr(x, "key", x), self.src)), getattr(self.arg, "key", self.arg))

  def map_buffers(self, real_srcs: Mapping[Any, Union[LazyBuffer, LazyOp]]) -> LazyOp: return LazyOp(self.op, tuple([y.map_buffers(real_srcs) if y not in real_srcs else real_srcs[y] for y in self.src]), self.arg)
  def get_lazyops(self) -> List[LazyOp]: return [self] + [item for x in self.src for item in x.get_lazyops()]

  def replace_with_movement_ops(self:LazyOp, ops:List[Tuple[MovementOps, Tuple[Any, ...]]]) -> 'LazyBuffer':
    assert self.op in BinaryOps or self.op in UnaryOps or self.op in TernaryOps
    srcs = [z.replace_with_movement_ops(ops) for z in self.src]
    return srcs[0].e(self.op, *srcs[1:], arg=self.arg)   # type: ignore

  @property
  def st(self): raise NotImplementedError
  @property
  def realized(self): raise NotImplementedError
  @property
  def children(self): raise NotImplementedError

  # movement ops
  def reshape(self, _): raise NotImplementedError
  def pad(self, _): raise NotImplementedError
  def expand(self, _): raise NotImplementedError
  def permute(self, _): raise NotImplementedError
  def shrink(self, _): raise NotImplementedError
  def stride(self, _): raise NotImplementedError

# **************** Device ****************

class _Device:
  def __init__(self) -> None: self._buffers: List[str] = [x.stem[len("ops_"):].upper() for x in (pathlib.Path(__file__).parent/"runtime").iterdir() if x.stem.startswith("ops_")]
  def canonicalize(self, device:Optional[str]) -> str: return (device.split(":", 1)[0].upper() + ((":"+device.split(":", 1)[1]) if ':' in device else '')).replace(":0", "") if device is not None else self.DEFAULT
  @functools.lru_cache(maxsize=None)  # this class is a singleton, pylint: disable=method-cache-max-size-none
  def __getitem__(self, x:str) -> Union[Interpreted, Compiled]:
    x = x.split(":")[0].upper()
    return [cls for cname, cls in inspect.getmembers(importlib.import_module(f'tinygrad.runtime.ops_{x.lower()}')) if (cname.lower() == x.lower() + "buffer") and x in self._buffers][0]
  @functools.cached_property
  def DEFAULT(self) -> str:
    device_from_env: Optional[str] = functools.reduce(lambda val, ele: ele if getenv(ele) == 1 else val, self._buffers, None)
    if device_from_env: return device_from_env
    for device in ["METAL", "CUDA", "GPU"]:
      try:
        if self[device]: return device
      except Exception: pass
    return "CPU"
Device = _Device()

# **************** for Interpreted Buffers ****************

class Interpreted:
  def __init__(self, buffer, fxn_for_op: Dict[Op, Callable], from_underlying=None):
    self.buffer, self.fxn_for_op, self.from_underlying = buffer, fxn_for_op, from_underlying
    self.synchronize = lambda: None
    self.codegen = None
    self.method_cache: Dict[LazyOp, Callable] = {}

  def interpret_ast(self:Interpreted, ast:LazyOp) -> Callable:
    tglob: Dict[str, Any] = {}
    lines: List[str] = []
    f = self.fxn_for_op

    @functools.lru_cache(None)
    def gstr(x:Any, nm=None) -> str:
      ret = str(nm).replace(".", "_") if nm else f"m{len(tglob):04d}"
      tglob[ret] = x
      return ret

    @functools.lru_cache(None)
    def _interpret_ast(ast:LazyOp) -> str:
      if TernaryOps.MULACC in f and ast.op == ReduceOps.SUM and isinstance(ast.src[0], LazyOp) and ast.src[0].op == BinaryOps.MUL:
        ast = LazyOp(TernaryOps.MULACC, ast.src[0].src, ast.arg)

      if MovementOps.AS_STRIDED in f and ast.op in BufferOps:
        tmp = f"{gstr(f[ast.op], ast.op)}({gstr(ast.arg.val)}, {gstr(ast.arg.dtype)})" if ast.op == BufferOps.CONST else f"{gstr(f[ast.op], ast.op)}(inputs[{ast.arg.idx-1}])"
        for mop,arg in ast.arg.st.to_movement_ops(): tmp = f"{gstr(f[mop], mop)}({tmp}, {gstr(arg)})"
      else:
        inp = [_interpret_ast(src) for src in ast.src]
        tmp = f"{gstr(f[ast.op], ast.op)}({', '.join(inp + ([gstr(ast.arg)] if ast.arg else []))})"

      ret = f"a{len(lines)}"
      lines.append(f"  {ret} = {tmp}")
      return ret

    ret = _interpret_ast(ast)
    src = '\n'.join(['def run(inputs):'] + lines + [f"  return {gstr(self.from_underlying, 'from_underlying')}({ret})" if self.from_underlying else f"  return {ret}"])
    if DEBUG >= 4: print(functools.reduce(lambda x,y: (x.replace(y[0], str(y[1])) if y[0][0:2] == "m0" else x), tglob.items(), src))
    exec(compile(src, "<ast>", "exec"), tglob) # pylint: disable=exec-used
    return tglob['run']

  def exec_ast(self, ast:LazyOp, output=None, inputs=None, var_vals=None, **kwargs):
    if ast not in self.method_cache: self.method_cache[ast] = self.interpret_ast(ast)
    ret = self.method_cache[ast]([x.realized for x in inputs] if inputs else None)
    if output is not None and ret.dtype != output.dtype and UnaryOps.CAST in self.fxn_for_op:
      ret = self.from_underlying(self.fxn_for_op[UnaryOps.CAST](self.fxn_for_op[BufferOps.MEM](ret), (output.dtype, False))) # Do manual casting of ret if it does not match the required output dtype.
    # TODO: is this used?
    if output is not None and output.output_buffer is not None:
      assert output.output_buffer.dtype == ret.dtype
      output.output_buffer._buf = ret._buf
      return output.output_buffer
    return ret

@dataclass
class FlopCounter:
  shape: Tuple[int, ...]
  dtype: DType
  flops: int
  mem: Dict[int, int]
  @property
  def mem_estimate(self): return sum(self.mem.values()) + self.dtype.itemsize*prod(self.shape)
  def consume_flops(self):
    self.flops, ret = 0, self.flops
    return ret
InterpretedFlopCounter = Interpreted(FlopCounter, {
  BufferOps.MEM: lambda arg: FlopCounter(arg.st.shape, arg.dtype, 0, {arg.idx: arg.dtype.itemsize*arg.st.size()}), BufferOps.CONST: lambda arg: FlopCounter(arg.st.shape, arg.dtype, 0, {}),
  UnaryOps.CAST: lambda self,arg: FlopCounter(self.shape, arg[0], self.consume_flops(), self.mem),   # cast uses no flops
  **{op:lambda self: FlopCounter(self.shape, self.dtype, self.consume_flops() + prod(self.shape), self.mem) for op in UnaryOps if op != UnaryOps.CAST},
  **{op:lambda self,y: FlopCounter(self.shape, max(self.dtype, y.dtype), self.consume_flops() + y.consume_flops() + prod(self.shape), {**self.mem, **y.mem}) for op in BinaryOps},
  **{op:lambda self,new_shape: FlopCounter(new_shape, self.dtype, self.consume_flops() + prod(self.shape), self.mem) for op in ReduceOps},
  TernaryOps.WHERE: lambda self,y,z: FlopCounter(self.shape, y.dtype, self.consume_flops() + y.consume_flops() + z.consume_flops() + prod(self.shape), {**self.mem, **y.mem, **z.mem})})

@functools.lru_cache(None)
def get_lazyop_info(ast:LazyOp) -> FlopCounter: return InterpretedFlopCounter.exec_ast(ast)

# **************** for Compiled Buffers ****************

class ASTRunner:
  def __init__(self, name:str, prg:str, global_size:Optional[List[int]]=None, local_size:Optional[List[int]]=None, op_estimate=0, mem_estimate=0, display_name:Optional[str]=None, runtime_args:Optional[dict]=None):
    if DEBUG >= 4: print(prg)
    self.name, self.prg, self.global_size, self.local_size, self.op_estimate, self.mem_estimate, self.display_name, self.runtime_args = name, prg, global_size, local_size, op_estimate, mem_estimate, display_name, runtime_args if runtime_args is not None else {}

  def build(self, compiler, runtime):
    self.lib = compiler.__wrapped__(self.prg) if getenv("DISABLE_COMPILER_CACHE") else compiler(self.prg)
    self.clprg = runtime(self.name, self.lib)
    return self

  def exec(self, rawbufs, var_vals:Optional[Dict[Variable, int]]=None, force_wait=False) -> Optional[float]:
    from tinygrad.jit import CacheCollector
    CacheCollector.add(self, rawbufs, var_vals if var_vals is not None else {})
    return self(rawbufs, var_vals, force_wait=force_wait)

  def launch_dims(self, var_vals):
    global_size = ([sym_infer(sz, var_vals) for sz in self.global_size] + [1]*(3-len(self.global_size))) if self.global_size is not None else self.global_size
    local_size = ([sym_infer(sz, var_vals) for sz in self.local_size] + [1]*(3-len(self.local_size))) if self.local_size is not None else self.local_size
    return global_size, local_size

  def __call__(self, rawbufs:List[RawBuffer], var_vals:Optional[Dict[Variable, int]]=None, jit=False, force_wait=False) -> Optional[float]:
    if var_vals is None: var_vals = {}
    global_size, local_size = self.launch_dims(var_vals)
    if global_size is not None and local_size is None:
      # TODO: this is copied from get_program
      from tinygrad.features.search import optimize_local_size
      local_size = self.local_size = optimize_local_size(self.clprg, global_size, rawbufs)
      global_size = self.global_size = [g//l if g%l == 0 else g/l for g,l in zip(global_size, local_size)]
    lra = self.runtime_args.copy()
    if global_size: lra['global_size'] = global_size
    if local_size and 'local_size' not in lra: lra['local_size'] = local_size
    if et := self.clprg(*rawbufs, *var_vals.values(), **lra, wait=force_wait or DEBUG>=2): GlobalCounters.time_sum_s += et
    op_estimate = sym_infer(self.op_estimate, var_vals)
    mem_estimate = sym_infer(self.mem_estimate, var_vals)
    if DEBUG >= 2:
      print(f"{colored(f'*** {GlobalCounters.kernel_count:4d}', 'magenta' if jit else None)} {(self.display_name+' '*(37-ansilen(self.display_name))) if self.display_name is not None else self.name:33s} arg {len(rawbufs):3d} sz {str(global_size):18s} {str(local_size):12s} OPs {int(op_estimate/1e6):6d}M/{GlobalCounters.global_ops/1e9:7.2f}G  mem {GlobalCounters.mem_used/1e9:5.2f} GB " +
            (str() if et is None else f"tm {et*1e6:9.2f}us/{GlobalCounters.time_sum_s*1e3:9.2f}ms ({op_estimate/((et or 1e-20)*1e9):8.2f} GFLOPS, {mem_estimate/((et or 1e-20)*1e9):7.2f} GB/s)"))
    GlobalCounters.kernel_count += 1
    GlobalCounters.global_ops += op_estimate
    GlobalCounters.global_mem += mem_estimate
    return et

class Compiled:
  def __init__(self, buffer: Type[RawBuffer], linearizer_opts, renderer, compiler, runtime, synchronize=lambda: None):
    self.buffer, self.linearizer_opts, self.renderer, self.compiler, self.runtime, self.synchronize = buffer, linearizer_opts, renderer, compiler, runtime, synchronize
    self.method_cache: Dict[LazyOp, ASTRunner] = {}

  def to_program(self, k):
    k.linearize()
    src, runtime_args = self.renderer(k.function_name, k.uops)
    return ASTRunner(k.function_name, src, k.global_size, k.local_size,
                     op_estimate=k.info.flops, mem_estimate=k.info.mem_estimate,
                     display_name=k.display_name, runtime_args=runtime_args).build(self.compiler, self.runtime)

  def exec_ast(self, ast:LazyOp, output, inputs, var_vals, **kwargs):
    # check if we can reuse the output buffer
    # if it's aliased, don't use it
    # NOTE: this is pretty wrong actually, who knows where else this buffer is used?
    output.realized = output.output_buffer
    if output.realized:
      for i,a in enumerate(inputs):
        # TODO: if this is contiguous it's fine
        if a.realized == output.realized:
          if any(not x.arg.st.contiguous for x in ast.get_lazyops() if x.op == BufferOps.MEM and x.arg.idx == i+1):
            output.realized = None
            break

    # we don't have an output buffer, we have to create it, and create to max size if it has symbolic shape
    if not output.realized:
      output.realized = self.buffer(prod((s if isinstance(s, int) else s.max for s in output.shape)), output.dtype, **kwargs)

    # all the rawbuffers
    rawbuffers = [output.realized] + [x.realized for x in inputs]

    # extract real vars used in ast
    from tinygrad.lazy import vars_from_ast
    ast_vars = vars_from_ast(ast)
    assert all(v.val is None for v in ast_vars), f"ast contains bound Variable {ast_vars}"

    # compilation time
    def get_program():
      from tinygrad.codegen.linearizer import Linearizer
      k = Linearizer(ast, self.linearizer_opts)
      assert k.info.dtype == output.dtype, f"linearizer must match dtype. linearizer wants {k.info.dtype} but buffer is {output.dtype}"
      if not NOOPT:
        if not (used_tensor_cores:=k.apply_tensor_cores(getenv("TC", 1))): k.hand_coded_optimizations()
        if BEAM >= 1 and not vars_from_ast(ast):
          lins = [(("tc" if used_tensor_cores else "hc"), k)]
          # allocate a scratch buffer if output buffer is also input
          test_rawbuffers = [type(rawbuffers[0])(rawbuffers[0].size, rawbuffers[0].dtype), *rawbuffers[1:]] if rawbuffers[0] in rawbuffers[1:] else rawbuffers
          kb = Linearizer(ast, self.linearizer_opts)
          kb.required_optimizations()
          from tinygrad.features.search import beam_search, time_linearizer
          lins.append((f"beam{BEAM.value}", beam_search(kb, test_rawbuffers, BEAM.value, bool(getenv("BEAM_ESTIMATE", 1)))))
          if used_tensor_cores:
            lins.append(("hc", Linearizer(ast, self.linearizer_opts)))
            lins[-1][1].hand_coded_optimizations()
          timed = sorted([(nm, tk, time_linearizer(tk, test_rawbuffers, allow_test_size=False, disable_cache=True, clear_l2=True)) for nm, tk in lins], key=lambda x: x[2])
          if DEBUG >= 1: print("  <  ".join(f"{nm:6s} : {lin.colored_shape(30, dense=True)} : {tm*1e6:8.2f} us" for nm, lin, tm in timed))
          k = timed[0][1]
      else:
        k.required_optimizations()
      return self.to_program(k)

    if getenv("ENABLE_METHOD_CACHE", 1):
      if ast not in self.method_cache: self.method_cache[ast] = get_program()
      prg = self.method_cache[ast]
    else:
      prg = get_program()

    if prg.name == getenv("PRINT_PRG", ''): print(prg.prg)

    prg.exec(rawbuffers, var_vals={k:var_vals[k] for k in ast_vars})
    return output.realized
