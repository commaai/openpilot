from __future__ import annotations
import time
from typing import cast
from tinygrad.device import Buffer, BufferSpec, Compiled, Device, MultiBuffer
from tinygrad.dtype import dtypes
from tinygrad.engine.jit import GraphRunner
from tinygrad.engine.realize import get_call_outs_ins, get_runtime
from tinygrad.helpers import round_up, ceildiv
from tinygrad.runtime.support.memory import BumpAllocator
from tinygrad.uop.ops import Ops, PatternMatcher, UOp, UPat, graph_rewrite
from extra.hcq2.hcq2 import HCQ2Compiled, HCQ2DeviceCtx, HCQ2LowerCtx, pm_prep_runtime, pm_lower_ops
from extra.hcq2.hcq2 import pm_split_into_queues, pm_add_barriers, pm_add_signals
from extra.hcq2.hcq2 import pm_bufferize, pm_lift_patches_to_cmdbuf, pm_resolve_patches, pm_parametrize_host_buffers
from extra.hcq2.hcq2 import pm_add_timeline_inc, pm_callify, pm_calc_kernargs_sizes

# **************** insert deps ****************

def insert_deps(ctx:HCQ2Graph, linear:UOp) -> UOp:
  src = []
  for j, call in enumerate(linear.src):
    call = call.replace(tag=j)
    _, _, bufs, _ = ctx.calls[j]
    outs, ins = get_call_outs_ins(call)
    deps = ctx._access_resources([bufs[i] for i in outs + ins], list(range(len(outs))), call)
    src.append(UOp(Ops.AFTER, call.dtype, (call, *deps), tag=call.tag))
  return linear.replace(src=tuple(src))
pm_insert_deps = PatternMatcher([(UPat(Ops.LINEAR, name="linear"), insert_deps)])

pm_replace_params = PatternMatcher([
  (UPat(Ops.PARAM, name="p"), lambda ctx, p: ctx.input_addrs_uop.index(UOp.const(dtypes.int, p.arg))),
  (UPat(Ops.BUFFER_VIEW, src=(UPat(Ops.INDEX, name="addr"),), name="bv"),
    lambda ctx, bv, addr: addr.cast(dtypes.uint64) + UOp.const(dtypes.uint64, bv.arg[1] * bv.dtype.itemsize)),
])

# **************** graph-only passes ****************

def alloc_queue_sig(ctx:HCQ2Graph, q:UOp) -> None:
  if q.arg in ctx.queue_sigs: return None
  dev = q.arg[0][0]  # TODO: multi device
  buf = Buffer(dev, 0x100, dtypes.uint8, options=BufferSpec(host=True, uncached=True, cpu_access=True), preallocate=True)
  ctx.queue_sig_bufs.append(buf)
  ctx.queue_sigs[q.arg] = UOp.from_buffer(buf, dev)
  return None
pm_alloc_queue_sigs = PatternMatcher([(UPat(Ops.LINEAR, src=UPat({Ops.PROGRAM, Ops.COPY}), name="q"), alloc_queue_sig)])

def lower_queue_deps(ctx:HCQ2Graph, after:UOp) -> UOp:
  wrapper, deps, call_idx = after.src[0], after.src[1:], after.tag
  def store(q_arg, v): return ctx.queue_sigs[q_arg].store(UOp.const(dtypes.uint32, v))
  waits = tuple(UOp(Ops.WAIT, dtypes.void, (ctx.queue_sigs[dep.src[0].arg], UOp.const(dtypes.uint32, dep.tag),
                                            store(dep.src[0].arg, dep.tag))) for dep in deps)
  return wrapper.replace(src=tuple(q.replace(src=(*waits, *q.src, store(q.arg, call_idx))) for q in wrapper.src))
pm_lower_queue_deps = PatternMatcher([(UPat(Ops.AFTER, src=UPat(Ops.LINEAR), name="after"), lower_queue_deps)])

def optimize_queue_deps(ctx:HCQ2Graph, queue:UOp) -> UOp|None:
  src, seen, pending, queue_sig = [], {}, {}, ctx.queue_sigs[queue.arg]
  for x in queue.src:
    if x.op is Ops.WAIT:
      sig, val = x.src[0], x.src[1]
      if sig is queue_sig or seen.get(sig, -1) >= val.arg: continue
      if (old:=pending.get(sig)) is None or old.src[1].arg < val.arg: pending[sig] = x
      continue
    for wait in pending.values():
      src.append(wait)
      seen[wait.src[0]] = wait.src[1].arg
    pending.clear()
    src.append(x)
  src += pending.values()
  return queue.replace(src=tuple(src)) if tuple(src) != queue.src else None
pm_optimize_queue_deps = PatternMatcher([
  (UPat(Ops.LINEAR, src=UPat({Ops.BARRIER, Ops.WAIT, Ops.STORE, Ops.PROGRAM, Ops.COPY}), name="queue"), optimize_queue_deps),
])

def drop_dead_stores(ctx:HCQ2Graph, outer:UOp) -> UOp:
  live = {u.src[2] for u in outer.toposort() if u.op is Ops.WAIT}
  return outer.replace(src=tuple(q.replace(src=tuple(x for x in q.src if x.op is not Ops.STORE or x in live)) for q in outer.src))
pm_drop_dead_stores = PatternMatcher([(UPat(Ops.LINEAR, src=UPat(Ops.LINEAR), name="outer"), drop_dead_stores)])

def add_queue_sig_resets(ctx:HCQ2Graph, x:UOp, cmdbuf:UOp) -> UOp|None:
  if not ctx.queue_sig_bufs or cmdbuf.tag not in ("compute", "copy"): return None
  resets = tuple((b:=UOp.from_buffer(sig)).index(UOp.const(dtypes.int, 0), dtype=b.dtype.ptr())
                 .cast(dtypes.uint64.ptr()).store(UOp.const(dtypes.uint64, 0)) for sig in ctx.queue_sig_bufs)
  return x.replace(src=x.src + resets)
pm_add_queue_sig_resets = PatternMatcher([(UPat(Ops.AFTER, src=(UPat(Ops.BUFFER, name="cmdbuf"),), allow_any_len=True, name="x"),
                                           add_queue_sig_resets)])

# **************** Graph ****************

class HCQ2Graph(GraphRunner):
  def __init__(self, linear:UOp, input_uops:tuple[UOp, ...]=()):
    super().__init__(linear, input_uops)
    self.dev = cast(HCQ2Compiled, Device[self.device])
    self.hcq_ctx = HCQ2LowerCtx(name="hcq_graph")

    self.input_addrs = Buffer("CPU", max(len(input_uops), 1), dtypes.uint64, preallocate=True)
    self.input_addrs_uop = UOp.from_buffer(self.input_addrs, "CPU")

    self.linear = graph_rewrite(self.linear, pm_insert_deps, ctx=self, name="hcq: insert deps", walk=True)
    self.linear = graph_rewrite(self.linear, pm_replace_params, ctx=self, name="hcq: replace params", walk=True)
    self.linear = graph_rewrite(self.linear, pm_prep_runtime, ctx=self.hcq_ctx, name="hcq: prepare runtime")
    self.linear = graph_rewrite(self.linear, pm_lower_ops, ctx=self.hcq_ctx, name="hcq: lower ops")

    # per-queue signal state — populated as a side-effect by pm_alloc_queue_sigs walking the lowered linear.
    self.queue_sig_bufs:list[Buffer] = []
    self.queue_sigs:dict[tuple[str, str], UOp] = {}
    graph_rewrite(self.linear, pm_alloc_queue_sigs, ctx=self, name="hcq: alloc queue sigs", walk=True)

    self.linear = graph_rewrite(self.linear, pm_lower_queue_deps, ctx=self, name="hcq: lower queue deps")
    self.linear = graph_rewrite(self.linear, pm_split_into_queues, ctx=self.hcq_ctx, name="hcq: split into queues")
    self.linear = graph_rewrite(self.linear, pm_add_barriers, ctx=self.hcq_ctx, name="hcq: add barriers", walk=True)
    self.linear = graph_rewrite(self.linear, pm_optimize_queue_deps, ctx=self, name="hcq: optimize queue deps", walk=True)
    self.linear = graph_rewrite(self.linear, pm_drop_dead_stores, ctx=self, name="hcq: drop dead stores")
    self.linear = graph_rewrite(self.linear, pm_add_signals, ctx=self.hcq_ctx, name="hcq: add signals", walk=True)
    self.linear = graph_rewrite(self.linear, pm_add_timeline_inc, ctx=self.hcq_ctx, name="hcq: add submit", walk=True)
    self.linear = graph_rewrite(self.linear, self.dev.pm_lower, ctx=self.hcq_ctx, name=f"hcq: encode cmdbuf {self.dev.device}", walk=True)

    graph_rewrite(self.linear, pm_calc_kernargs_sizes, ctx=(sizes:={}), name=None)
    for dev_name, sz in sizes.items():
      buf = Buffer(dev_name, sz, dtypes.uint8, options=BufferSpec(cpu_access=True), preallocate=True)
      self.hcq_ctx.dev_ctx[dev_name] = HCQ2DeviceCtx(dev_name, UOp.from_buffer(buf, dev_name), UOp.const(dtypes.uint64, buf._buf.va_addr))

    self.linear = graph_rewrite(self.linear, pm_bufferize, ctx=self.hcq_ctx, bottom_up=True, name="realize binaries")
    self.linear = graph_rewrite(self.linear, pm_lift_patches_to_cmdbuf, ctx=self.hcq_ctx, bottom_up=False, name="lift patches to cmdbuf")
    self.linear = graph_rewrite(self.linear, pm_resolve_patches, ctx=self.hcq_ctx, bottom_up=False, name="simplify patches")
    self.linear = graph_rewrite(self.linear, pm_add_queue_sig_resets, ctx=self, name="hcq: add queue sig resets", walk=True)
    self.linear = graph_rewrite(self.linear, pm_parametrize_host_buffers, ctx=self.hcq_ctx, bottom_up=True, name="parametrize host buffers")
    self.host_call = graph_rewrite(self.linear, pm_callify, ctx=self.hcq_ctx, name="hcq: callify")

    self.host_rt, self.host_globals = get_runtime("CPU", self.host_call.src[0]), self.host_call.src[0].arg.globals

  def __call__(self, input_uops:tuple[UOp, ...], var_vals:dict[str, int], wait=False) -> float|None:
    addrs = self.input_addrs.as_memoryview(force_zero_copy=True).cast('Q')
    for i, u in enumerate(input_uops):
      buf = next(b for b in u.buffer.bufs if b.device == self.dev.device) if isinstance(u.buffer, MultiBuffer) else u.buffer
      addrs[i] = buf._buf.va_addr
    self.host_rt(*[self.hcq_ctx.inputs[i].get_buf("CPU") for i in self.host_globals], vals=self.host_call.src[0].arg.vals(var_vals), wait=True)
    if wait:
      st = time.perf_counter()
      self.dev.synchronize()
      return time.perf_counter() - st
    return None

  @staticmethod
  def supports_uop(batch_devs:list[Compiled], new_call:UOp) -> bool:
    all_devs = GraphRunner._all_devs(batch_devs, new_call)
    return new_call.src[0].op in (Ops.PROGRAM, Ops.COPY) and len(all_devs) == 1 and isinstance(all_devs[0], HCQ2Compiled)
