from typing import TypeVar, Generic, Callable, List, Tuple, Union, Dict, cast, Optional, Any
import functools, collections
from tinygrad.tensor import Tensor
from tinygrad.helpers import flatten, merge_dicts, DEBUG, Context, BEAM, getenv, colored, JIT, dedup, partition, unwrap
from tinygrad.device import Buffer, Compiled, Device
from tinygrad.dtype import DType
from tinygrad.ops import UOp, ssimplify, Variable, sint, sym_infer
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.engine.realize import ExecItem, capturing, EmptyOp, ViewOp, BufferCopy, BufferXfer, CompiledRunner, Runner
from tinygrad.engine.memory import _internal_memory_planner
from tinygrad.nn.state import get_parameters
from dataclasses import dataclass
from weakref import WeakKeyDictionary

class GraphException(Exception): pass

def apply_graph_to_jit(jit_cache: List[ExecItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int], max_batch_size=0) -> List[ExecItem]:
  # Split JIT cache into batches for faster graph execution.
  # This allows the accelerator to run some batches while subsequent graphs are still being updated.
  graphed_jit_cache: List[ExecItem] = []
  current_batch: List[ExecItem] = []
  current_device: Optional[Compiled] = None

  def flush_batch():
    nonlocal current_batch, current_device, max_batch_size
    try:
      if len(current_batch) <= 1 or current_device is None: raise GraphException("only one kernel doesn't graph")
      graph_runner = current_device.graph(current_batch, input_rawbuffers, var_vals)
      # clear jit inputs to allow their memory to be freed/reused
      for (j,i) in graph_runner.input_replace.keys(): graph_runner.jit_cache[j].bufs[i] = None
      graphed_jit_cache.append(ExecItem(graph_runner, cast(List[Optional[Buffer]], input_rawbuffers)))
      max_batch_size *= 2
      if DEBUG >= 2: print(f"JIT GRAPHing batch with {len(current_batch)} kernels on device {current_device}")
    except GraphException as e:
      graphed_jit_cache.extend(current_batch)
      if DEBUG >= 2: print(f"JIT GRAPHing failed batch with {len(current_batch)} kernels on device {current_device}: {e}")
    current_batch = []
    current_device = None

  for ji in jit_cache:
    if ji.prg.__class__ in {EmptyOp, ViewOp}: continue
    ji_graph_dev: Optional[Compiled] = None # device on which the ji will be graphed. Not graphed if None.
    if isinstance(ji.prg, CompiledRunner): ji_graph_dev = ji.prg.dev
    elif isinstance(ji.prg, BufferXfer) and ji.bufs[0] and ji.bufs[0].device.split(":", 1)[0] in {"CUDA", "NV", "AMD"}:
      ji_graph_dev = Device[ji.bufs[0].device]

    graph_class = (ji_graph_dev.graph.func if isinstance(ji_graph_dev.graph, functools.partial) else ji_graph_dev.graph) if ji_graph_dev else None
    can_be_graphed = ji_graph_dev and ji_graph_dev.graph
    can_share_graph = (ji_graph_dev == current_device or (isinstance(graph_class, type) and issubclass(graph_class, MultiGraphRunner)) and
                       type(ji_graph_dev) is type(current_device))
    can_extend_graph_batch = can_be_graphed and (max_batch_size == 0 or len(current_batch) < max_batch_size) and can_share_graph
    if not can_extend_graph_batch and len(current_batch) > 0: flush_batch()

    if can_be_graphed: current_batch.append(ji)
    else: graphed_jit_cache.append(ji)

    current_device = ji_graph_dev

  if len(current_batch) > 0: flush_batch()
  return graphed_jit_cache

def get_input_replace(jit_cache: List[ExecItem], input_rawbuffers:List[Buffer]) -> Dict[Tuple[int, int], int]:
  input_replace: Dict[Tuple[int, int], int] = {}
  for j,ji in enumerate(jit_cache):
    for i,a in enumerate(ji.bufs):
      if a in input_rawbuffers:
        input_replace[(j,i)] = input_rawbuffers.index(a)
  return input_replace

class GraphRunner(Runner):
  def __init__(self, jit_cache: List[ExecItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]):
    self.jit_cache = jit_cache  # NOTE: this is not used, but you have to keep these objects alive for the Graph
    self.input_replace:Dict[Tuple[int, int], int] = get_input_replace(jit_cache, input_rawbuffers)
    self.var_vals_replace:Dict[int, List[int]] = {}
    self.launch_dims_replace:Dict[int, Tuple[Optional[int], Optional[int]]] = {}
    self.launch_dims_base:Dict[int, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {}

    op_estimate: sint = 0
    mem_estimate: sint = 0
    lds_estimate: sint = 0

    def is_sym_dim(dim) -> bool: return not all(isinstance(d, (int, float)) for d in dim)

    self.vars = sorted(var_vals.keys(), key=lambda v: v.expr)
    self.symbolic_dims = dedup([tuple(d) for ji in jit_cache if isinstance(ji.prg, CompiledRunner) and (d:=ji.prg.p.local_size) and is_sym_dim(d)] +
                               [tuple(d) for ji in jit_cache if isinstance(ji.prg, CompiledRunner) and (d:=ji.prg.p.global_size) and is_sym_dim(d)])
    def find_symbolic_dim(dim): return self.symbolic_dims.index(tuple(dim)) if dim is not None and tuple(dim) in self.symbolic_dims else None

    for j,ji in enumerate(jit_cache):
      op_estimate += ji.prg.op_estimate
      mem_estimate += ji.prg.mem_estimate
      lds_estimate += ji.prg.lds_estimate
      if isinstance(ji.prg, CompiledRunner):
        if ji.prg.p.vars: self.var_vals_replace[j] = [self.vars.index(v) for v in ji.prg.p.vars]

        global_dim_idx, local_dim_idx = find_symbolic_dim(ji.prg.p.global_size), find_symbolic_dim(ji.prg.p.local_size)
        if global_dim_idx is not None or local_dim_idx is not None:
          self.launch_dims_replace[j] = (global_dim_idx, local_dim_idx)
          assert ji.prg.p.global_size is not None and ji.prg.p.local_size is not None
          self.launch_dims_base[j] = (tuple(ji.prg.p.global_size), tuple(ji.prg.p.local_size))

    # used in MultiGraphRunner. the ints are id() of _bufs
    self.w_dependency_map: Dict[int, Any] = {}
    self.r_dependency_map: Dict[int, List[Any]] = collections.defaultdict(list)

    super().__init__(colored(f"<batched {len(jit_cache)}>", "cyan"), jit_cache[0].prg.device.split(":")[0],
                     ssimplify(op_estimate), ssimplify(mem_estimate), ssimplify(lds_estimate))

  def updated_vars(self, var_vals: Dict[Variable, int]):
    vals = [var_vals[v] for v in self.vars]
    for j, vidxs in self.var_vals_replace.items():
      for i, v in enumerate(vidxs): yield j, i, vals[v]

  def updated_launch_dims(self, var_vals: Dict[Variable, int]):
    dims = [tuple(sym_infer(s, var_vals) for s in dim) for dim in self.symbolic_dims]
    for j, (gl, lc) in self.launch_dims_replace.items():
      yield j, (dims[gl] if gl is not None else self.launch_dims_base[j][0]), (dims[lc] if lc is not None else self.launch_dims_base[j][1])

  def _access_resources(self, rawbufs:List[Buffer], write:List[int], new_dependency:Any):
    # To synchronize access to resources, we monitor the necessary prerequisites for accessing each resource,
    # whether for write or read operations. A resource can be accessed by either a single writer or multiple readers.
    wait_nodes = []

    for i,rawbuf in enumerate(rawbufs):
      if id(rawbuf.base._buf) in self.w_dependency_map: wait_nodes.append(self.w_dependency_map[id(rawbuf.base._buf)])
      if i in write:
        if id(rawbuf.base._buf) in self.r_dependency_map: wait_nodes.extend(self.r_dependency_map.pop(id(rawbuf.base._buf)))
        self.w_dependency_map[id(rawbuf.base._buf)] = new_dependency
      else: self.r_dependency_map[id(rawbuf.base._buf)].append(new_dependency)

    return list({id(x):x for x in wait_nodes}.values())

# a marker for your graph supporting multiple devices of the same type
class MultiGraphRunner(GraphRunner): pass

ReturnType = TypeVar('ReturnType')
@dataclass
class CapturedJit(Generic[ReturnType]):
  ret: Any  # includes the Tensors or any other returned object
  jit_cache: List[ExecItem]
  input_replace: Dict[Tuple[int, int], int]
  extra_view_inputs: List[Tuple[int, int, str, int, DType]]
  expected_names: List[Union[int, str]]
  expected_st_vars_dtype_device: List[Tuple[ShapeTracker, Tuple[Variable, ...], DType, str]]

  def __reduce__(self):
    return self.__class__, (self.ret, self.jit_cache, self.input_replace, self.extra_view_inputs,
                            self.expected_names, self.expected_st_vars_dtype_device)

  def __post_init__(self):
    self._jit_cache: List[ExecItem] = self.jit_cache
    self._input_replace: Dict[Tuple[int, int], int] = self.input_replace
    self._graphed = False
    self._clear_inputs()

  def _clear_inputs(self):
    for (j,i) in self._input_replace.keys(): self._jit_cache[j].bufs[i] = None

  # jit exec
  def __call__(self, input_buffers:List[Buffer], var_vals:Dict[Variable, int]) -> ReturnType:
    # assign inputs
    for idx, offset, device, size, dtype in self.extra_view_inputs:
      input_buffers.append(Buffer(device, size, dtype, base=input_buffers[idx], offset=offset).ensure_allocated())
    for (j,i),input_idx in self._input_replace.items(): self._jit_cache[j].bufs[i] = input_buffers[input_idx]

    # Condense the items into a graph executor.
    if JIT < 2 and not self._graphed:
      self._jit_cache = apply_graph_to_jit(self.jit_cache, input_buffers, var_vals, max_batch_size=getenv("JIT_BATCH_SIZE", 32))
      self._input_replace = get_input_replace(self._jit_cache, input_buffers)
      self._graphed = True

    if DEBUG >= 1 and len(self._jit_cache) >= 10: print(f"jit execs {len(self._jit_cache)} kernels")
    for ei in self._jit_cache: ei.run(var_vals, jit=True)
    self._clear_inputs()
    return self.ret

def _prepare_jit_inputs(args, kwargs):
  input_tensors: List[Tuple[int|str, Tensor]] = [(name,t) for name,t in list(enumerate(args))+sorted(kwargs.items()) if t.__class__ is Tensor]
  names, tensors = [name for name,_ in input_tensors], [t for _,t in input_tensors]
  if tensors: Tensor.realize(*tensors)
  lbs: List[UOp] = flatten([t.lazydata.lbs for t in tensors])
  input_buffers: List[Buffer] = [lb.base.realized for lb in lbs if lb.base.realized is not None]
  assert len(set(input_buffers)) == len(input_buffers), "duplicate inputs to JIT"
  st_varval_dtype_device = [(*unwrap(lb.st).unbind(), lb.dtype, lb.device) for lb in lbs]
  var_vals = merge_dicts([x[1] for x in st_varval_dtype_device] + [dict(v.unbind() for v in (args + tuple(kwargs.values())) if isinstance(v, UOp))])
  st_vars_dtype_device = [(x[0], tuple(sorted(x[1].keys(), key=lambda v: v.expr)), x[2], x[3]) for x in st_varval_dtype_device]
  return input_buffers, var_vals, names, st_vars_dtype_device

class TinyJit(Generic[ReturnType]):
  def __init__(self, fxn:Optional[Callable[..., ReturnType]], captured:Optional[CapturedJit]=None, prune=False):
    assert fxn or captured, "need either a function or a CapturedJit"
    self.fxn = fxn
    self.captured: Optional[CapturedJit] = captured
    self.cnt: int = 2 if self.fxn is None else 0
    self.prune = prune

  def add_buffer(self, b:Buffer) -> Buffer:
    if found:=self._buffer_replace.get(b, None): return found
    if b.is_allocated() or b.lb_refcount > 0: return b
    if b._base is not None:
      self._buffer_replace[b] = ret = Buffer(b.device, b.size, b.dtype, base=self.add_buffer(b._base), offset=b.offset)
    else:
      self._buffer_replace[b] = ret = Buffer(b.device, b.size, b.dtype, options=b.options)
    return ret

  def add(self, ei:ExecItem):
    self._jit_cache.append(ExecItem(ei.prg, [self.add_buffer(buf) for buf in ei.bufs if buf is not None]))

  def reset(self):
    assert self.fxn is not None, "can't reset without function"
    self.cnt = 0
    self.captured = None

  def __reduce__(self):
    assert self.captured is not None, "can't pickle an uncaptured JIT"
    return self.__class__, (None, self.captured)

  # keep legacy code working
  @property
  def jit_cache(self) -> List[ExecItem]: return self.captured._jit_cache if self.captured is not None else []
  @property
  def input_replace(self) -> Dict[Tuple[int, int], int]: return self.captured._input_replace if self.captured is not None else {}

  def __get__(self, obj, objtype): return functools.partial(self.__call__, obj) # add support for instance methods

  def __call__(self, *args, **kwargs) -> ReturnType:
    input_buffers, var_vals, names, st_vars_dtype_device = _prepare_jit_inputs(args, kwargs)
    if not JIT or self.cnt == 0:
      # jit ignore
      assert self.fxn is not None
      with Context(BEAM=0 if getenv("IGNORE_JIT_FIRST_BEAM") else BEAM.value):
        ret = self.fxn(*args, **kwargs)
        if len(params:=get_parameters(ret)): Tensor.realize(params[0], *params[1:])
    elif self.cnt == 1:
      # jit capture
      assert self.fxn is not None
      if capturing: raise RuntimeError(f"having TinyJit inside another TinyJit is not supported {len(capturing)=} {capturing=}")
      self._jit_cache: List[ExecItem] = []
      self._buffer_replace: WeakKeyDictionary[Buffer, Buffer] = WeakKeyDictionary()
      # TODO: should we always disable the memory planner here? it must be off for prune
      with Context(BEAM=getenv("JITBEAM", BEAM.value), NO_MEMORY_PLANNER=int(self.prune)):
        capturing.append(self)
        try:
          ret = self.fxn(*args, **kwargs)
          if len(params:=get_parameters(ret)): Tensor.realize(params[0], *params[1:])
        except Exception as e: raise e
        finally: capturing.clear()
      jit_cache = self._jit_cache
      del self._buffer_replace, self._jit_cache
      assert len(jit_cache), "didn't JIT anything!"
      if DEBUG >= 1: print(f"JIT captured {len(jit_cache)} kernels with {len(input_buffers)} inputs")

      # track inputs that are views of buffers
      # TODO: eventually expected_buffers should live in ExecItem
      extra_view_inputs: List[Tuple[int, int, str, int, DType]] = []
      for item in jit_cache:
        for b in item.bufs:
          if b is not None and b._base is not None and b._base in input_buffers:
            input_buffers.append(b)
            extra_view_inputs.append((input_buffers.index(b.base), b.offset, b.device, b.size, b.dtype))

      # prune independent kernels (optional)
      if self.prune:
        depends = set(input_buffers)
        for ei in jit_cache:
          if any(b in depends for b in ei.bufs):
            if isinstance(ei.prg, CompiledRunner):
              depends.update(cast(Buffer, ei.bufs[out]) for out in ei.prg.p.outs)
            if isinstance(ei.prg, (BufferCopy, BufferXfer)):
              depends.add(cast(Buffer, ei.bufs[0]))
        pruned, onetime = partition(jit_cache,
                                    lambda ei: not isinstance(ei.prg, CompiledRunner) or any(ei.bufs[out] in depends for out in ei.prg.p.outs))
        if DEBUG >= 1: print(f"pruned from {len(jit_cache)} -> {len(pruned)} kernels")
        # run the onetime kernels here
        for ei in onetime:
          for b in ei.bufs: cast(Buffer, b).ensure_allocated()
          ei.run(var_vals, jit=True)
        jit_cache = pruned

      # memory planning (optional)
      # Exclude buffers involved in transfer ops to preserve parallelism.
      noopt_buffers = {b for ji in jit_cache if isinstance(ji.prg, BufferXfer) for b in ji.bufs}
      assigned = _internal_memory_planner([cast(List[Buffer], item.bufs) for item in jit_cache], noopt_buffers, debug_prefix="JIT ")
      jit_cache = [ExecItem(item.prg, [assigned.get(b,b).ensure_allocated() for b in item.bufs if b is not None]) for item in jit_cache]

      input_replace = get_input_replace(jit_cache, input_buffers)
      if DEBUG >= 1 and len(set(input_replace.values())) != len(input_buffers): print("WARNING: some input tensors not found")

      # set this for next run
      self.captured = CapturedJit(ret, jit_cache, input_replace, extra_view_inputs, names, st_vars_dtype_device)
    elif self.cnt >= 2:
      # jit exec
      assert self.captured is not None
      assert self.captured.expected_names == names, f"args mismatch in JIT: {self.captured.expected_names=} != {names}"
      assert self.captured.expected_st_vars_dtype_device == st_vars_dtype_device, \
        f"args mismatch in JIT: {self.captured.expected_st_vars_dtype_device=} != {st_vars_dtype_device=}"
      ret = self.captured(input_buffers, var_vals)

    self.cnt += 1
    return ret
