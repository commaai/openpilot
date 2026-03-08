from typing import TypeVar, Generic, Callable, cast, Any
import functools, collections
from tinygrad.tensor import Tensor
from tinygrad.helpers import flatten, merge_dicts, DEBUG, Context, BEAM, getenv, colored, JIT, JIT_BATCH_SIZE, dedup, partition, unwrap
from tinygrad.device import Buffer, Compiled, Device, MultiBuffer
from tinygrad.dtype import DType
from tinygrad.uop.ops import UOp, Variable, sym_infer, Ops
from tinygrad.engine.realize import ExecItem, capturing, ViewOp, BufferCopy, BufferXfer, EncDec, CompiledRunner, Runner, Estimates
from tinygrad.engine.memory import _internal_memory_planner
from tinygrad.nn.state import get_parameters
from tinygrad.schedule.rangeify import mop_cleanup
from dataclasses import dataclass, replace
from weakref import WeakKeyDictionary

class GraphException(Exception): pass
class JitError(Exception): pass

def _check_no_non_tensor_return(ret):
  if ret is None or isinstance(ret, Tensor): return
  if isinstance(ret, (tuple, list, dict)):
    for item in (ret.values() if isinstance(ret, dict) else ret): _check_no_non_tensor_return(item)
    return
  raise JitError(f"JIT return contains non-Tensor value of type {type(ret).__name__}")

def graph_class(dev): return dev.graph.func if isinstance(dev.graph, functools.partial) else dev.graph

def apply_graph_to_jit(jit_cache: list[ExecItem], input_buffers: list[Buffer], var_vals: dict[str, int],
                       orig_valid_positions: dict[int, set[int]]|None = None, max_batch_size=0) -> list[ExecItem]:
  # Split JIT cache into batches for faster graph execution.
  # This allows the accelerator to run some batches while subsequent graphs are still being updated.
  graphed_jit_cache: list[ExecItem] = []
  current_batch: list[ExecItem] = []
  current_batch_devs: list[Compiled] = []

  def flush_batch():
    nonlocal current_batch, current_batch_devs, max_batch_size
    try:
      if len(current_batch_devs) == 0: raise GraphException("no device for graph")
      if len(current_batch) <= 1 and not getenv("GRAPH_ONE_KERNEL"): raise GraphException("only one kernel doesn't graph")
      graph_runner = current_batch_devs[0].graph(current_batch, input_buffers, var_vals, orig_valid_positions=orig_valid_positions)
      # clear jit inputs to allow their memory to be freed/reused
      for (j,i) in graph_runner.input_replace.keys(): graph_runner.jit_cache[j].bufs[i] = None
      graphed_jit_cache.append(ExecItem(UOp(Ops.NOOP), cast(list[Buffer|None], input_buffers), prg=graph_runner))
      max_batch_size *= 2
      if DEBUG >= 2: print(f"JIT GRAPHing batch with {len(current_batch)} kernels on device {current_batch_devs[0]}")
    except GraphException as e:
      graphed_jit_cache.extend(current_batch)
      if DEBUG >= 2: print(f"JIT GRAPHing failed batch with {len(current_batch)} kernels on device {current_batch_devs[0]}: {e}")
    current_batch = []
    current_batch_devs = []

  for ji in jit_cache:
    match ji.prg:
      case CompiledRunner(): ji_graph_dev = ji.prg.dev
      case BufferXfer(): ji_graph_dev = Device[unwrap(ji.bufs[0]).device]
      case BufferCopy(): ji_graph_dev = next((Device[unwrap(b).device] for b in ji.bufs if unwrap(b).device != "CPU"), None)
      case ViewOp(): continue # ViewOps are just ignored
      case _: ji_graph_dev = None # Everything else is not graphed and flushes existing graph if it's being constructed

    # Check if this jit item can be graphed at all, so check if a new graph supports the current item.
    can_be_graphed = ji_graph_dev is not None and ji_graph_dev.graph is not None and graph_class(ji_graph_dev).supports_exec_item([ji_graph_dev], ji)

    # Check if the current batch can be extended with this item.
    can_share_graph = can_be_graphed and len(current_batch_devs) > 0 and \
                      graph_class(current_batch_devs[0]).supports_exec_item(dedup(current_batch_devs + [ji_graph_dev]), ji)
    can_extend_graph_batch = can_share_graph and (max_batch_size == 0 or len(current_batch) < max_batch_size)

    # Flush the current batch if any, since it can't be extended or is full.
    if not can_extend_graph_batch and len(current_batch) > 0: flush_batch()
    (current_batch if can_be_graphed else graphed_jit_cache).append(ji)
    current_batch_devs = dedup(current_batch_devs + [ji_graph_dev]) if can_be_graphed else []

  if len(current_batch) > 0: flush_batch()
  return graphed_jit_cache

def get_input_replace(jit_cache: list[ExecItem], input_buffers:list[Buffer],
                      orig_valid_positions: dict[int, set[int]]|None = None) -> dict[tuple[int, int], int]:
  input_replace: dict[tuple[int, int], int] = {}
  for j,ji in enumerate(jit_cache):
    for i,a in enumerate(ji.bufs):
      if a in input_buffers:
        # filter out positions that weren't valid inputs in the original capture (prevents aliasing bugs)
        if orig_valid_positions is not None and i not in orig_valid_positions.get(id(ji), set()): continue
        input_replace[(j,i)] = input_buffers.index(a)
  return input_replace

class GraphRunner(Runner):
  def __init__(self, jit_cache: list[ExecItem], input_buffers: list[Buffer], var_vals: dict[str, int],
               orig_valid_positions: dict[int, set[int]]|None = None):
    self.jit_cache = jit_cache  # NOTE: this is not used, but you have to keep these objects alive for the Graph
    self.input_replace:dict[tuple[int, int], int] = get_input_replace(jit_cache, input_buffers, orig_valid_positions)
    self.var_vals_replace:dict[int, list[tuple[int, int]]] = {}
    self.launch_dims_replace:dict[int, tuple[int|None, int|None]] = {}
    self.launch_dims_base:dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}

    def is_sym_dim(dim) -> bool: return not all(isinstance(d, (int, float)) for d in dim)

    self.vars = sorted(var_vals.keys())
    self.symbolic_dims = dedup([tuple(d) for ji in jit_cache if isinstance(ji.prg, CompiledRunner) and (d:=ji.prg.p.local_size) and is_sym_dim(d)] +
                               [tuple(d) for ji in jit_cache if isinstance(ji.prg, CompiledRunner) and (d:=ji.prg.p.global_size) and is_sym_dim(d)])
    def find_symbolic_dim(dim): return self.symbolic_dims.index(tuple(dim)) if dim is not None and tuple(dim) in self.symbolic_dims else None

    estimates = Estimates()
    for j,ji in enumerate(jit_cache):
      assert ji.prg is not None
      estimates += ji.prg.estimates
      if isinstance(ji.prg, CompiledRunner):
        if (replace:=[(i, self.vars.index(v.expr)) for i, v in enumerate(ji.prg.p.vars) if v.expr not in ji.fixedvars | ji.prg.p.runtimevars]):
          self.var_vals_replace[j] = replace

        global_dim_idx, local_dim_idx = find_symbolic_dim(ji.prg.p.global_size), find_symbolic_dim(ji.prg.p.local_size)
        if global_dim_idx is not None or local_dim_idx is not None:
          self.launch_dims_replace[j] = (global_dim_idx, local_dim_idx)
          assert ji.prg.p.local_size is not None
          self.launch_dims_base[j] = (tuple(ji.prg.p.global_size), tuple(ji.prg.p.local_size))

    # used in MultiGraphRunner. the ints are id() of _bufs
    self.w_dependency_map: dict[int, Any] = {}
    self.r_dependency_map: dict[int, list[Any]] = collections.defaultdict(list)

    assert jit_cache[0].prg is not None
    super().__init__(colored(f"<batched {len(jit_cache)}>", "cyan"), jit_cache[0].prg.device.split(":")[0], estimates.simplify())

  def updated_vars(self, var_vals: dict[str, int]):
    vals = [var_vals[v] for v in self.vars]
    for j, vidxs in self.var_vals_replace.items():
      for i, v in vidxs: yield j, i, vals[v]

  def updated_launch_dims(self, var_vals: dict[str, int]):
    dims = [tuple(sym_infer(s, var_vals) for s in dim) for dim in self.symbolic_dims]
    for j, (gl, lc) in self.launch_dims_replace.items():
      yield j, (dims[gl] if gl is not None else self.launch_dims_base[j][0]), (dims[lc] if lc is not None else self.launch_dims_base[j][1])

  def _access_resources(self, bufs:list[Buffer], write:list[int], new_dependency:Any):
    # To synchronize access to resources, we monitor the necessary prerequisites for accessing each resource,
    # whether for write or read operations. A resource can be accessed by either a single writer or multiple readers.
    wait_nodes = []

    for i,buf in enumerate(bufs):
      if id(buf.base._buf) in self.w_dependency_map: wait_nodes.append(self.w_dependency_map[id(buf.base._buf)])
      if i in write:
        if id(buf.base._buf) in self.r_dependency_map: wait_nodes.extend(self.r_dependency_map.pop(id(buf.base._buf)))

    for i,buf in enumerate(bufs):
      if i in write: self.w_dependency_map[id(buf.base._buf)] = new_dependency
      else: self.r_dependency_map[id(buf.base._buf)].append(new_dependency)

    return list({id(x):x for x in wait_nodes}.values())

  @staticmethod
  def supports_exec_item(devs:list[Compiled], ei:ExecItem) -> bool: return isinstance(ei.prg, CompiledRunner) and len(dedup(devs)) == 1

# a marker for your graph supporting multiple devices of the same type
class MultiGraphRunner(GraphRunner):
  @staticmethod
  def supports_exec_item(devs:list[Compiled], ei:ExecItem) -> bool:
    # Devices must be the same type
    return isinstance(ei.prg, (CompiledRunner, BufferXfer)) and len(dedup([type(Device[b.device]) for b in ei.bufs if b]+[type(d) for d in devs]))==1

def get_out_buffers_for_ei(ei:ExecItem) -> list[Buffer]:
  if isinstance(ei.prg, CompiledRunner): return [cast(Buffer, ei.bufs[out]) for out in ei.prg.p.outs if out not in ei.prg.p.ins]
  if isinstance(ei.prg, (BufferCopy, BufferXfer, EncDec)): return [cast(Buffer, ei.bufs[0])]
  return []

def update_depends(depends:set[Buffer|None], jit_cache:list[ExecItem]):
  for ei in jit_cache:
    if any(b in depends for b in ei.bufs): depends.update(get_out_buffers_for_ei(ei))

ReturnType = TypeVar('ReturnType')
@dataclass
class CapturedJit(Generic[ReturnType]):
  ret: Any  # includes the Tensors or any other returned object
  jit_cache: list[ExecItem]
  input_replace: dict[tuple[int, int], int]
  extra_view_inputs: list[tuple[int, int, str, int, DType]]
  expected_names: list[int|str]
  expected_input_info: list[tuple[UOp, tuple[Variable, ...], DType, str]]  # (view, variables, dtype, device) per input

  def __reduce__(self):
    # TODO: free_intermediates here? replan_buffers_memory_layout here?
    return self.__class__, (self.ret, self.jit_cache, self.input_replace, self.extra_view_inputs, self.expected_names, self.expected_input_info)

  def __post_init__(self):
    self._jit_cache: list[ExecItem] = self.jit_cache
    self._input_replace: dict[tuple[int, int], int] = self.input_replace
    self._first_run = True
    # precompute read-after-write hazard detection
    self._output_to_writer = {b: j for j, ei in enumerate(self.jit_cache) for b in get_out_buffers_for_ei(ei)}
    self._input_to_max_reader: dict[int, int] = {}
    for (j, _), idx in self.input_replace.items(): self._input_to_max_reader[idx] = max(self._input_to_max_reader.get(idx, -1), j)
    self._clear_inputs()

  def _clear_inputs(self):
    for (j,i) in self._input_replace.keys(): self._jit_cache[j].bufs[i] = None

  def free_intermediates(self):
    depends: set[Buffer|None] = set([None])
    update_depends(depends, self.jit_cache)
    for b in depends:
      if b is not None:
        if b.is_allocated(): b.deallocate()
        if b._base is not None and b._base.allocated_views == 0 and b._base.is_allocated(): b._base.deallocate()
    self.__post_init__()   # reset the graph state

  def replan_buffers_memory_layout(self):
    blacklist = [t.uop.buffer for t in get_parameters(self.ret)]
    asgn = _internal_memory_planner([[b for item in self.jit_cache for b in item.bufs if b is not None and b not in blacklist]], ignore_checks=True)
    self.jit_cache = [replace(item, bufs=[asgn.get(b,b) if b is not None else None for b in item.bufs]) for item in self.jit_cache]
    for old, new in asgn.items():
      if old.is_allocated(): new.ensure_allocated().copyin(old.as_buffer())
    self.__post_init__()

  # jit exec
  def __call__(self, input_buffers:list[Buffer], var_vals:dict[str, int]) -> ReturnType:
    # assign inputs
    for idx, offset, device, size, dtype in self.extra_view_inputs:
      input_buffers.append(Buffer(device, size, dtype, base=input_buffers[idx], offset=offset).ensure_allocated())

    # copy aliased inputs to prevent read-after-write hazard
    for i, ib in enumerate(input_buffers):
      if (writer := self._output_to_writer.get(ib)) is not None and self._input_to_max_reader.get(i, -1) > writer:
        input_buffers[i] = Buffer(ib.device, ib.size, ib.dtype).ensure_allocated().copyin(ib.as_buffer())

    for (j,i),input_idx in self._input_replace.items(): self._jit_cache[j].bufs[i] = input_buffers[input_idx]

    # Condense the items into a graph executor.
    if self._first_run:
      # allocate intermediates if freed
      for ji in self.jit_cache:
        for b in ji.bufs:
          if b is not None: b.ensure_allocated()
      # create graph if needed
      if JIT < 2:
        # build a map from ExecItem object to the buffer positions that are valid inputs (from original input_replace)
        orig_valid_positions: dict[int, set[int]] = {}  # id(ExecItem) -> set of valid buffer indices
        for (j, i) in self.input_replace: orig_valid_positions.setdefault(id(self.jit_cache[j]), set()).add(i)
        self._jit_cache = apply_graph_to_jit(self.jit_cache, input_buffers, var_vals, orig_valid_positions, max_batch_size=JIT_BATCH_SIZE.value)
        # recompute input_replace: GraphRunner items have all positions valid, non-GraphRunner items use orig_valid_positions
        valid_positions = {id(ji): set(range(len(ji.bufs))) if isinstance(ji.prg, GraphRunner) else orig_valid_positions.get(id(ji), set())
                          for ji in self._jit_cache}
        self._input_replace = get_input_replace(self._jit_cache, input_buffers, valid_positions)
      self._first_run = False

    if DEBUG >= 1 and len(self._jit_cache) >= 10: print(f"jit execs {len(self._jit_cache)} kernels")
    for ei in self._jit_cache: ei.run(var_vals, jit=True)
    self._clear_inputs()
    return self.ret

def _prepare_jit_inputs(args, kwargs):
  input_tensors: list[tuple[int|str, Tensor]] = [(name,t) for name,t in list(enumerate(args))+sorted(kwargs.items()) if t.__class__ is Tensor]
  names, tensors = [name for name,_ in input_tensors], [t for _,t in input_tensors]
  # extract tensors from containers (shallow, not recursive to avoid grabbing model weights)
  for x in args + tuple(kwargs.values()):
    it = x if isinstance(x, (tuple,list)) else x.values() if isinstance(x, dict) else []
    tensors += [t for t in it if t.__class__ is Tensor and not any(t is y for y in tensors)]
  if len(unrealized_tensors := [x for x in tensors if not x.uop.is_realized]): Tensor.realize(*unrealized_tensors)
  input_uops: list[UOp] = flatten([t.uop.src if t.uop.op is Ops.MULTI else [t.uop] for t in tensors])
  if any(u.base.op is Ops.CONST for u in input_uops):
    raise JitError("JIT inputs cannot be const, create a buffer with .contiguous()")
  input_buffers: list[Buffer] = flatten([b.bufs if isinstance(b:=u.base.realized, MultiBuffer) else [b]
                                         for u in input_uops if u.base.realized is not None])
  if len(set(input_buffers)) != len(input_buffers): raise JitError("duplicate inputs to JIT")
  inputs = [(*(u.substitute({u.base:UOp(Ops.NOOP)}, extra_pm=mop_cleanup).unbind_all()), u.dtype, u.device) for u in input_uops]
  _var_vals = merge_dicts([x[1] for x in inputs] + [dict(v.unbind() for v in (args + tuple(kwargs.values())) if isinstance(v, UOp))])
  var_vals = {k.expr:v for k,v in _var_vals.items()}
  expected_input_info = [(x[0], tuple(sorted(x[1].keys(), key=lambda v: v.expr)), x[2], x[3]) for x in inputs]
  return input_buffers, var_vals, names, expected_input_info

class TinyJit(Generic[ReturnType]):
  def __init__(self, fxn:Callable[..., ReturnType]|None, captured:CapturedJit|None=None, prune=False, optimize=False):
    assert fxn or captured, "need either a function or a CapturedJit"
    self.fxn = fxn
    self.captured: CapturedJit|None = captured
    self.cnt: int = 2 if self.fxn is None else 0
    self.prune = prune
    self.optimize = optimize

  def add_buffer(self, b:Buffer) -> Buffer:
    if found:=self._buffer_replace.get(b, None): return found
    if b.is_allocated() or b.uop_refcount > 0: return b
    if b._base is not None:
      self._buffer_replace[b] = ret = Buffer(b.device, b.size, b.dtype, base=self.add_buffer(b._base), offset=b.offset)
    else:
      self._buffer_replace[b] = ret = Buffer(b.device, b.size, b.dtype, options=b.options)
    return ret

  def add(self, ei:ExecItem):
    self._jit_cache.append(ExecItem(ei.ast, [self.add_buffer(buf) for buf in ei.bufs if buf is not None], ei.metadata, ei.fixedvars, ei.prg))

  def reset(self):
    assert self.fxn is not None, "can't reset without function"
    self.cnt = 0
    self.captured = None

  def __reduce__(self):
    assert self.captured is not None, "can't pickle an uncaptured JIT"
    return self.__class__, (None, self.captured)

  # keep legacy code working
  @property
  def jit_cache(self) -> list[ExecItem]: return self.captured._jit_cache if self.captured is not None else []
  @property
  def input_replace(self) -> dict[tuple[int, int], int]: return self.captured._input_replace if self.captured is not None else {}

  def __get__(self, obj, objtype): return functools.partial(self.__call__, obj) # add support for instance methods

  def __call__(self, *args, **kwargs) -> ReturnType:
    input_buffers, var_vals, names, expected_input_info = _prepare_jit_inputs(args, kwargs)
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
      self._jit_cache: list[ExecItem] = []
      self._buffer_replace: WeakKeyDictionary[Buffer, Buffer] = WeakKeyDictionary()
      # TODO: should we always disable the memory planner here? it must be off for prune
      with Context(BEAM=getenv("JITBEAM", BEAM.value), NO_MEMORY_PLANNER=int(self.prune)):
        capturing.append(self)
        try:
          ret = self.fxn(*args, **kwargs)
          if len(params:=get_parameters(ret)): Tensor.realize(params[0], *params[1:])
        finally: capturing.clear()
      jit_cache = self._jit_cache
      del self._buffer_replace, self._jit_cache
      if not len(jit_cache): raise JitError("didn't JIT anything!")
      _check_no_non_tensor_return(ret)
      if DEBUG >= 1: print(f"JIT captured {len(jit_cache)} kernels with {len(input_buffers)} inputs")

      # track inputs that are views of buffers
      # TODO: eventually expected_buffers should live in ExecItem
      extra_view_inputs: list[tuple[int, int, str, int, DType]] = []
      for item in jit_cache:
        for b in item.bufs:
          if b is not None and b._base is not None and b._base in input_buffers:
            input_buffers.append(b)
            extra_view_inputs.append((input_buffers.index(b.base), b.offset, b.device, b.size, b.dtype))

      # prune independent kernels (optional)
      if self.prune:
        depends = set(input_buffers)
        update_depends(depends, jit_cache)
        pruned, onetime = partition(jit_cache, lambda ei: any(b in depends for b in get_out_buffers_for_ei(ei)))
        if DEBUG >= 1: print(f"pruned from {len(jit_cache)} -> {len(pruned)} kernels")
        # run the onetime kernels here
        for ei in onetime:
          for b in ei.bufs: cast(Buffer, b).ensure_allocated()
          ei.run(var_vals, jit=True)
        jit_cache = pruned

      # memory planning (optional)
      # Exclude buffers involved in transfer ops to preserve parallelism.
      noopt_buffers = {b for ji in jit_cache if isinstance(ji.prg, (BufferXfer, BufferCopy, EncDec)) for b in ji.bufs}
      assigned = _internal_memory_planner([cast(list[Buffer], item.bufs) for item in jit_cache], noopt_buffers, debug_prefix="JIT ")
      jit_cache = [replace(item, bufs=[assigned.get(b,b).ensure_allocated() for b in item.bufs if b is not None]) for item in jit_cache]

      input_replace = get_input_replace(jit_cache, input_buffers)
      if DEBUG >= 1 and len(set(input_replace.values())) != len(input_buffers): print("WARNING: some input tensors not found")

      # set this for next run
      self.captured = CapturedJit(ret, jit_cache, input_replace, extra_view_inputs, names, expected_input_info)
      if self.optimize: self.captured.replan_buffers_memory_layout()
    elif self.cnt >= 2:
      # jit exec
      assert self.captured is not None
      if self.captured.expected_names != names: raise JitError(f"args mismatch in JIT: {self.captured.expected_names=} != {names}")
      if self.captured.expected_input_info != expected_input_info:
        raise JitError(f"args mismatch in JIT: {self.captured.expected_input_info=} != {expected_input_info=}")
      ret = self.captured(input_buffers, var_vals)

    self.cnt += 1
    return ret
