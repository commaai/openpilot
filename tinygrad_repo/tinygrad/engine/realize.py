from typing import List, Dict, Optional, cast, Generator, Tuple
import time, pprint
from dataclasses import dataclass, replace
from tinygrad.helpers import colored, getenv, DEBUG, GlobalCounters, ansilen, BEAM, NOOPT, all_int, CAPTURING, Metadata, TRACEMETA
from tinygrad.ops import Ops, UOp, Variable, sym_infer, sint
from tinygrad.device import Device, Buffer
from tinygrad.renderer import Renderer, ProgramSpec
from tinygrad.codegen.kernel import Kernel
from tinygrad.engine.schedule import ScheduleItem

# **************** Program Creation ****************

logkerns, logkerns_level = open(getenv("LOGKERNS", ""), "a") if getenv("LOGKERNS", "") else None, getenv("LOGKERNS_LEVEL", 1)
def get_kernel(renderer:Renderer, ast:UOp) -> Kernel:
  if DEBUG >= 5: print(ast)
  k = Kernel(ast, opts=renderer).required_optimizations()
  if not NOOPT:
    if not k.apply_tensor_cores(getenv("TC", 1)): k.hand_coded_optimizations()
    if BEAM >= 1:
      from tinygrad.engine.search import beam_search, bufs_from_lin
      kb = Kernel(ast, opts=renderer).required_optimizations()
      rawbufs = bufs_from_lin(kb, allocate=False)
      k = beam_search(kb, rawbufs, BEAM.value, bool(getenv("BEAM_ESTIMATE", 1)))
  if logkerns is not None: logkerns.writelines([f"{(k.ast, k.applied_opts)}\n"])
  if DEBUG >= 5: print((k.ast, k.applied_opts)) # print here to show final applied_opts for all kernels instead of just in beam_search
  return k

# **************** Runners ****************

class Runner:
  def __init__(self, display_name:str, device:str, op_estimate:sint=0, mem_estimate:sint=0, lds_estimate:Optional[sint]=None):
    self.first_run, self.display_name, self.device, self.op_estimate, self.mem_estimate, self.lds_estimate = \
      True, display_name, device, op_estimate, mem_estimate, mem_estimate if lds_estimate is None else lds_estimate
  @property
  def dev(self): return Device[self.device]
  def exec(self, rawbufs:List[Buffer], var_vals:Optional[Dict[Variable, int]]=None) -> Optional[float]:
    return self(rawbufs, {} if var_vals is None else var_vals)
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False) -> Optional[float]:
    raise NotImplementedError("override this")

class CompiledRunner(Runner):
  def __init__(self, p:ProgramSpec, precompiled:Optional[bytes]=None):
    if DEBUG >= 4: print(p.src)
    self.p:ProgramSpec = p
    self.lib:bytes = precompiled if precompiled is not None else Device[p.device].compiler.compile_cached(p.src)
    if DEBUG >= 6: Device[p.device].compiler.disassemble(self.lib)
    self._prg = Device[p.device].runtime(p.function_name, self.lib)
    super().__init__(p.name, p.device, p.op_estimate, p.mem_estimate, p.lds_estimate)

  def __reduce__(self): return self.__class__, (self.p, self.lib)

  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False) -> Optional[float]:
    global_size, local_size = self.p.launch_dims(var_vals)
    if global_size is not None and local_size is None and all_int(self.p.global_size): # type: ignore[arg-type]
      # TODO: this is copied from get_program
      from tinygrad.engine.search import optimize_local_size
      local_size = optimize_local_size(self._prg, global_size, rawbufs)
      global_size = [g//l if g%l == 0 else g/l for g,l in zip(global_size, local_size)]
      self.p = replace(self.p, global_size=global_size, local_size=local_size)
    lra = {}
    if global_size:
      lra['global_size'] = tuple(global_size)
      assert len(global_size) == 3, "global size must have len 3"
    if local_size:
      lra['local_size'] = tuple(local_size)
      assert len(local_size) == 3, "local size must have len 3"
    return self._prg(*[x._buf for x in rawbufs], **lra, vals=tuple(var_vals[k] for k in self.p.vars), wait=wait)

class EmptyOp(Runner):
  def __init__(self, buf:Buffer): super().__init__(colored(f"empty {buf.size:10d} {buf.dtype}", "yellow"), buf.device)
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False): pass

class ViewOp(Runner):
  def __init__(self, buf:Buffer): super().__init__(colored(f"view {buf.nbytes:8d} @ {buf.offset:<10d}", "yellow"), buf.device)
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False):
    assert rawbufs[0]._base is not None and rawbufs[0]._base == rawbufs[1].base, f"must be base {rawbufs}"

class BufferCopy(Runner):
  def __init__(self, total_sz, dest_device, src_device):
    if total_sz >= 1e6: name = f"{type(self).__name__[6:].lower()} {total_sz/1e6:7.2f}M, {dest_device[:7]:>7s} <- {src_device[:7]:7s}"
    else: name = f"{type(self).__name__[6:].lower()} {total_sz:8d}, {dest_device[:7]:>7s} <- {src_device[:7]:7s}"
    super().__init__(colored(name, "yellow"), dest_device, 0, total_sz)
  def copy(self, dest, src):
    disk_supports_fast_copyout = src.device.startswith("DISK") and hasattr(src.allocator.dev, 'io_uring') and \
      getattr(src.allocator.dev, 'fd', None) is not None
    if src.device.startswith("DISK") and hasattr(dest.allocator, 'copy_from_disk') and disk_supports_fast_copyout and src.nbytes >= 4096:
      dest.allocator.copy_from_disk(dest._buf, src._buf, src.nbytes)
    elif src.device.startswith("DISK") and hasattr(dest.allocator, '_as_buffer'):
      # fast(ish) path, uses readinto in diskbuffers
      src.allocator._copyout(dest.allocator._as_buffer(dest._buf), src._buf)
    else:
      dest.copyin(src.as_buffer(allow_zero_copy=True))  # may allocate a CPU buffer depending on allow_zero_copy
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False):
    dest, src = rawbufs[0:2]
    assert dest.size == src.size and dest.dtype == src.dtype, f"buffer copy mismatch, {dest.size} != {src.size}, {dest.dtype} != {src.dtype}"
    st = time.perf_counter()
    self.copy(dest, src)
    if wait:
      Device[dest.device].synchronize()
      return time.perf_counter() - st

class BufferXfer(BufferCopy):
  def copy(self, dest, src): dest.allocator._transfer(dest._buf, src._buf, dest.nbytes, src_dev=src.allocator.dev, dest_dev=dest.allocator.dev)

# **************** method cache ****************

method_cache: Dict[Tuple[str, bytes, int, int, bool], CompiledRunner] = {}
def get_runner(device:str, ast:UOp) -> CompiledRunner:
  ckey = (device, ast.key, BEAM.value, NOOPT.value, False)
  if cret:=method_cache.get(ckey): return cret
  bkey = (device.split(":")[0], ast.key, BEAM.value, NOOPT.value, True)
  if bret:=method_cache.get(bkey):
    method_cache[ckey] = ret = CompiledRunner(replace(bret.p, device=device), bret.lib)
  else:
    prg: ProgramSpec = get_kernel(Device[device].renderer, ast).to_program()
    method_cache[ckey] = method_cache[bkey] = ret = CompiledRunner(replace(prg, device=device))
  return ret

# **************** lowering functions ****************

@dataclass(frozen=True)
class ExecItem:
  prg: Runner
  bufs: List[Optional[Buffer]]
  metadata: Optional[Tuple[Metadata, ...]] = None
  def run(self, _var_vals:Optional[Dict[Variable, int]]=None, wait=False, jit=False, do_update_stats=True) -> Optional[float]:
    var_vals = {} if _var_vals is None else _var_vals
    bufs = [cast(Buffer, x) for x in self.bufs] if jit else [cast(Buffer, x).ensure_allocated() for x in self.bufs]
    et = self.prg(bufs, var_vals, wait=wait or DEBUG >= 2)
    if do_update_stats:
      GlobalCounters.kernel_count += 1
      GlobalCounters.global_ops += (op_est:=sym_infer(self.prg.op_estimate, var_vals))
      GlobalCounters.global_mem += (mem_est:=sym_infer(self.prg.mem_estimate, var_vals))
      if et is not None: GlobalCounters.time_sum_s += et
      if DEBUG >= 2:
        lds_est = sym_infer(self.prg.lds_estimate, var_vals)
        mem_est = min(mem_est, lds_est)   # there can't be more memory accessed than loads/stores. remove this when symbolic is fixed
        ptm = (colored(f"{et*1e3:9.2f}ms", "yellow") if et > 0.01 else f"{et*1e6:9.2f}us") if et is not None else ""
        print(f"{colored(f'*** {self.prg.device[:7]:7s} {GlobalCounters.kernel_count:4d}', 'magenta' if jit else ('green' if self.prg.first_run else None))} {self.prg.display_name+' '*(41-ansilen(self.prg.display_name))} arg {len(bufs):2d} mem {GlobalCounters.mem_used/1e9:5.2f} GB " +  # noqa: E501
              (str() if et is None else f"tm {ptm}/{GlobalCounters.time_sum_s*1e3:9.2f}ms ({op_est/((et or 1e-20)*1e9):9.2f} GFLOPS {mem_est/((et or 1e-20)*1e9):6.1f}|{lds_est/((et or 1e-20)*1e9):<7.1f} GB/s)" +  # noqa: E501
               f" {[repr(m) if TRACEMETA >= 2 else str(m) for m in self.metadata] if self.metadata else ''}"))
      self.prg.first_run = False
    return et

def lower_schedule_item(si:ScheduleItem) -> ExecItem:
  assert len(set(x.device for x in si.bufs)) == 1 or si.ast.op is Ops.COPY
  if si.ast.op is Ops.SINK:
    runner = get_runner(si.outputs[0].device, si.ast)
    return ExecItem(runner, [si.bufs[x] for x in runner.p.globals], si.metadata)
  out = si.outputs[0]
  if si.ast.op is Ops.COPY:
    kernel_type = BufferCopy
    if hasattr(Device[out.device].allocator, '_transfer') and out.device.split(":")[0] == si.inputs[0].device.split(":")[0]:
      kernel_type = BufferXfer
    return ExecItem(kernel_type(out.nbytes, out.device, si.inputs[0].device), list(si.bufs))
  if si.ast.op is Ops.EMPTY: return ExecItem(EmptyOp(out), list(si.bufs))
  if si.ast.op is Ops.BUFFER_VIEW: return ExecItem(ViewOp(out), list(si.bufs))
  raise RuntimeError(f"don't know how to lower {si.ast}")

def lower_schedule(schedule:List[ScheduleItem]) -> Generator[ExecItem, None, None]:
  while len(schedule):
    si = schedule.pop(0)
    try: yield lower_schedule_item(si)
    except Exception as e:
      if DEBUG >= 2:
        print(f"error lowering {si.ast.op}")
        print("tensor operations:")
        pprint.pprint(si.metadata, indent=2)
      raise e

# **************** main run function ****************

capturing: List = []  # put classes with an add method in here

def run_schedule(schedule:List[ScheduleItem], var_vals:Optional[Dict[Variable, int]]=None, do_update_stats=True):
  for ei in lower_schedule(schedule):
    if len(capturing) and CAPTURING: capturing[0].add(ei)
    ei.run(var_vals, do_update_stats=do_update_stats)
