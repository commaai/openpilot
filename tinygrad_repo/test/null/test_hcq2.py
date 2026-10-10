import unittest
from typing import cast
from types import SimpleNamespace
from collections import defaultdict
from unittest.mock import patch
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.device import Buffer, Compiled, ProfileGraphEvent
from tinygrad.helpers import Context, unwrap, to_tuple
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher, KernelInfo
from tinygrad.engine.realize import compile_linear, link_linear, get_call_arg_uops, lower_and_compile, run_linear
import tinygrad.runtime.support.hcq2 as hcq2
from tinygrad.runtime.support.hcq2 import HCQInfo

def lower_hcq(*body:UOp) -> UOp: # body through lower_call, as a linear
  return UOp(Ops.LINEAR, src=(unwrap(hcq2.lower_call(UOp.sink(*body, arg=KernelInfo("test"), tag=1).call(aux=HCQInfo(("CPU",))))),))

def chain(x:Tensor, n:int) -> Tensor:
  for _ in range(n): x = (x + 1).contiguous()
  return x

def chain_input(value:int=2, device="NULL") -> Tensor: return Tensor.full((4,), value, dtype=dtypes.int32, device=device).contiguous().realize()

def compiled_chain(n:int, jit=False, device="NULL") -> tuple[Tensor, UOp, list[UOp]]:
  x, inputs = chain_input(device=device), []
  if jit:
    f = TinyJit(lambda a: chain(a, n).realize())
    f(x)
    return f(x), f.captured._linear, [x.uop.base]
  out = chain(x, n)
  return out, compile_linear(out.schedule_linear(), input_uops=inputs, cache=True), inputs

# NULL never runs a batch, so the scheduler is tested on the commands it hands each queue, run by a small executor with symbolic
# signals and timelines. the fence and the ffi are tested by running them on CPU, see test/runtime/test_hcq2.py.

def scheduled(*ts:Tensor, **kwargs) -> list[UOp]:
  batches, orig = list[UOp](), hcq2.sched_batches
  def track(l, profile):
    lin = orig(l, profile)
    batches.extend(c for c in lin.src if c.op is Ops.CALL and isinstance(c.arg.aux, HCQInfo))
    return lin
  with patch.object(hcq2, "sched_batches", track): compile_linear(ts[0].schedule_linear(*ts[1:]), **kwargs)
  return batches

def queues(batch:UOp) -> dict[tuple[str, str], list[UOp]]:
  return {(lin.arg[0][0], lin.arg[1]): list(lin.src) for lin in (s.src[1].without_after for s in batch.body.src)}
def calls(batch:UOp) -> list[UOp]: return [c for cmds in queues(batch).values() for c in cmds if c.op is Ops.CALL]
def devices_of(call:UOp) -> set[str]: return {to_tuple(a.device)[0] for a in get_call_arg_uops(call)}

def word(u:UOp) -> tuple[UOp, int]:
  if u.op is Ops.INDEX: return (v:=hcq2.unwrap_view(u.src[0]))[0], v[1] + u.src[1].val * u.dtype.itemsize
  return hcq2.unwrap_view(u)

T = 5 # the timeline value the last submitted batch of every device signals
def run(batch:UOp, done:dict[str, int]|None=None, prio:list|None=None) -> tuple[list, dict, dict]:
  qs, cs, mem = queues(batch), calls(batch), defaultdict(int)
  for d in batch.arg.aux.device:
    mem[word(hcq2.timeline((d,)))] = (done or {}).get(d, T)
    mem[word(hcq2.timeline((d,)).index(1))] = T
  def val(u:UOp) -> int:
    if u.op is Ops.CONST: return u.val
    if u.op is Ops.LOAD: return mem[word(u.src[0])]
    return sum(val(s) for s in u.src)
  log:list[int|str] = [] # the calls that ran, and a device each time its timeline bumps
  while True:
    for q in (list(qs) if prio is None else prio):
      if not (cmds:=qs[q]): continue
      c = cmds[0]
      if c.op is Ops.INS and c.arg[0].startswith("wait"):
        sig, target = mem[word(c.src[0])], val(c.src[1])
        if sig != target if c.arg[0] == "wait_eq" else sig < target: continue
      elif c.op is Ops.INS and c.arg[0] == "store":
        mem[w:=word(c.src[0])] = val(c.src[1])
        if w[0].tag == "timeline": log.append(w[0].device)
      elif c.op is Ops.CALL: log.append(cs.index(c))
      cmds.pop(0)
      break
    else: return log, {q: len(cmds) for q, cmds in qs.items()}, {d: mem[word(hcq2.timeline((d,)))] for d in batch.arg.aux.device}

def rotations(batch:UOp) -> list[list]: return [(qs:=list(queues(batch)))[i:] + qs[:i] for i in range(len(queues(batch)))]
def orders(batch:UOp) -> set[tuple[int, ...]]: return {tuple(x for x in run(batch, prio=p)[0] if isinstance(x, int)) for p in rotations(batch)}

class TestHCQ2Deps(unittest.TestCase):
  def test_buffer_views(self):
    b = Buffer("NULL", 16)
    for write in ([], [0]):
      tracker = hcq2.DepsTracker()
      tracker.access_resources([b], write, 0)
      self.assertEqual(tracker.access_resources([b.view(8, 4)], [0], 1), [0])
      self.assertEqual(tracker.access_resources([b.view(4, 0)], [0], 2), [0])
      self.assertEqual(tracker.access_resources([b.view(4, 12)], [0], 3), [0])
      self.assertEqual(tracker.access_resources([b.view(8, 4)], [], 4), [1])

  def test_dependencies_through_selected_slices(self):
    b = UOp.param(0, dtypes.float32, 64, device=("NULL", "NULL:1"))
    for view in [b.mselect(0).shrink(((8, 16),)), b.shrink(((8, 16),)).mselect(0), b.shrink(((4, 32),)).mselect(0).shrink(((4, 12),))]:
      tracker = hcq2.DepsTracker()
      tracker.access_resources([view], [0], 0)
      self.assertEqual(tracker.access_resources([b.mselect(1)], [], 1), [])
      self.assertEqual(tracker.access_resources([b.mselect(0).shrink(((16, 24),))], [], 2), [])
      self.assertEqual(tracker.access_resources([b.mselect(0).shrink(((12, 20),))], [], 3), [0])

  def test_disjoint_write_preserves_dependencies(self):
    b = UOp.param(0, dtypes.uint8, 16, device="NULL")
    for write in ([], [0]):
      tracker = hcq2.DepsTracker()
      tracker.access_resources([b.shrink(((0, 4),))], write, 0)
      self.assertEqual(tracker.access_resources([b.shrink(((4, 8),))], [0], 1), [])
      self.assertEqual(tracker.access_resources([b.shrink(((0, 4),))], [0], 2), [0])

  def test_partial_write_preserves_dependencies(self):
    b = UOp.param(0, dtypes.uint8, 16, device="NULL")
    for write in ([], [0]):
      tracker = hcq2.DepsTracker()
      tracker.access_resources([b], write, 0)
      self.assertEqual(tracker.access_resources([b.shrink(((4, 12),))], [0], 1), [0])
      self.assertEqual(tracker.access_resources([b.shrink(((0, 4),))], [0], 2), [0])
      self.assertEqual(tracker.access_resources([b.shrink(((12, 16),))], [0], 3), [0])
      self.assertEqual(tracker.access_resources([b.shrink(((4, 12),))], [], 4), [1])

  def test_write_waits_for_all_readers(self):
    b = UOp.param(0, dtypes.uint8, 16, device="NULL")
    tracker = hcq2.DepsTracker()
    self.assertEqual(tracker.access_resources([b], [], 0), [])
    self.assertEqual(tracker.access_resources([b], [], 1), [])
    self.assertEqual(tracker.access_resources([b], [0], 2), [0, 1])
    self.assertEqual(tracker.access_resources([b], [], 3), [2])

  def test_aliases_do_not_wait_on_themselves(self):
    b = UOp.param(0, dtypes.uint8, 16, device="NULL")
    for write in ([0], [1], [0, 1]):
      tracker = hcq2.DepsTracker()
      tracker.access_resources([b], [0], 0)
      self.assertEqual(tracker.access_resources([b, b.shrink(((4, 12),))], write, 1), [0])
      self.assertEqual(tracker.access_resources([b.shrink(((4, 12),))], [0], 2), [1])

class TestHCQ2Schedule(unittest.TestCase):
  def setUp(self):
    self.enterContext(Context(DEV="NULL"))
    self.x = Tensor.ones(4).contiguous().realize()

  def check(self, batch:UOp) -> UOp: # what every batch promises
    cs = calls(batch)
    for prio in rotations(batch):
      log, left, timelines = run(batch, prio=prio)
      self.assertFalse(any(left.values()), f"deadlock, left {left}")
      for d in batch.arg.aux.device:
        bumps = [i for i, x in enumerate(log) if x == d]
        self.assertEqual((len(bumps), timelines[d]), (1, T + 1), f"{d} must bump its timeline once")
        self.assertTrue(all(i < bumps[0] for i, x in enumerate(log) if isinstance(x, int) and d in devices_of(cs[x])), f"{d} bumps too early")
    for d in batch.arg.aux.device:
      log = run(batch, done={d: T - 1})[0]
      self.assertFalse([x for x in log if isinstance(x, int) and d in devices_of(cs[x])], f"{d} runs before its previous batch is done")
    return batch

  def scheduled(self, *ts:Tensor) -> list[UOp]: return [self.check(b) for b in scheduled(*ts)]
  def batch(self, *ts:Tensor) -> UOp: return unwrap(self.scheduled(*ts)[0])

  def test_kernels_run_in_order(self): self.assertEqual(orders(self.batch(chain(self.x, 3))), {(0, 1, 2)})

  def test_a_peer_kernel_runs_after_the_copy_that_feeds_it(self):
    self.assertEqual(orders(self.batch((self.x.to("NULL:1") + 1).contiguous())), {(0, 1)})

  def test_lanes_of_a_sharded_kernel_do_not_wait_for_each_other(self):
    s = Tensor.ones(8).contiguous().realize().shard(("NULL", "NULL:1"), axis=0).contiguous().realize()
    b = self.batch((s + 1).contiguous())
    self.assertEqual(orders(b), {(0, 1), (1, 0)})
    self.assertEqual(len([x for x in run(b, done={"NULL": T - 1})[0] if isinstance(x, int)]), 1)

  def test_a_device_without_a_copy_queue_copies_with_a_kernel(self):
    with patch.object(type(Device["NULL"]), "has_copy_queue", property(lambda _: False)):
      b = self.batch((self.x.to("NULL:1") + 1).contiguous())
    self.assertTrue(all(c.body.op is Ops.PROGRAM for c in calls(b)))
    self.assertEqual(orders(b), {(0, 1)})

  def test_a_host_kernel_splits_the_batch(self):
    self.assertEqual(len(self.scheduled(((self.x + 1).contiguous().to("PYTHON") + 2).contiguous().to("NULL") + 3)), 2)

  def test_batches_of_real_workloads_are_well_formed(self):
    t = Tensor.ones(6).contiguous().realize().shard(("NULL", "NULL:1", "NULL:2"), axis=0)
    self.scheduled((t + 1).sum(0).contiguous())
    self.scheduled((self.x.to("NULL:1") + 1).to("NULL:2").contiguous().to("NULL") + 1)

class TestHCQ2Profile(unittest.TestCase):
  def setUp(self): self.enterContext(Context(DEV="NULL"))

  def test_profiling_reports_a_range_per_kernel(self, n=2):
    x = Tensor.ones(4).contiguous().realize()
    with Context(PROFILE=1):
      seen = len(Compiled.profile_events)
      chain(x, n).realize()
      Device["NULL"].synchronize()
    (ev,) = [e for e in Compiled.profile_events[seen:] if isinstance(e, ProfileGraphEvent)]
    ranges = [(ev.sigs[e.st_id], ev.sigs[e.en_id]) for e in ev.ents]
    self.assertEqual([e.device for e in ev.ents], ["NULL"] * n)
    self.assertEqual([en - st for st, en in ranges], [1] * n, "NULL emulates 1us per kernel")
    self.assertEqual(ranges, sorted(ranges))

  def test_slots_addressed_by_the_device(self):
    pm = PatternMatcher([(UPat((Ops.LOAD, Ops.STORE), src=(UPat(Ops.INDEX, src=(UPat.var("buf"), UPat())),), allow_any_len=True),
                          lambda buf: buf.getaddr(Device["NULL"].host) if hcq2.unwrap_view(buf)[0].tag == "slots" else None)])
    with patch.object(Device["NULL"], "pm_lower", pm): self.test_profiling_reports_a_range_per_kernel(n=3)

class TestHCQ2Link(unittest.TestCase):
  def setUp(self): self.enterContext(Context(DEV="NULL"))

  def test_links_serve_any_input(self):
    a, inputs = chain_input(), list[UOp]()
    linear = compile_linear(chain(a, 2).schedule_linear(), input_uops=inputs, cache=True)
    linked = link_linear(linear, input_uops=inputs)
    self.assertIs(link_linear(linear, input_uops=[chain_input(3).uop.base, *inputs[1:]]), linked)
    bufs = [cast(Buffer, u.buffer) for u in linked.toposort() if u.op is Ops.BUFFER and u.addrspace is AddrSpace.GLOBAL]
    self.assertNotIn(a.uop.base.buffer, bufs)
    words = [w for b in bufs if b.options.external_ptr and b.nbytes % 8 == 0 for w in b.host.view(fmt='Q')[:]]
    self.assertNotIn(cast(Buffer, a.uop.base.buffer)._buf, words)

  def test_eager_templates_compile_once(self): self.assertIs(compiled_chain(3)[1], compiled_chain(3)[1])

  def test_repeated_word_loops(self): # for (i..10) cmdbuf[off[i]] = var
    var, offs = UOp.alloc((1,), dtypes.uint32, device="CPU").rtag("var"), [4 * i * i for i in range(10)]
    var = hcq2.patch(var, [], bytes(4)) # initialize at link: the allocator may return a reused buffer
    hq = SimpleNamespace(blob=bytearray(offs[-1] + 4), patches=[(o, var.index(0).load()) for o in offs], devs=("CPU",), queue="COPY:0", deps=())
    lowered = lower_hcq(var.after(hcq2.encode_cmdbuf(hq, device="CPU")).index(0).store(var.index(0).load() + 1))
    self.assertEqual(len([u for u in lowered.src[0].without_after.src[0].toposort() if u.op is Ops.RANGE]), 1)
    linked = hcq2.hcq_link(lower_and_compile(lowered), allow_cache=False)
    args = zip(lowered.src[0].without_after.src[1:], linked.src[0].without_after.src[1:])
    cmdbuf = next(b.buffer for p, b in args if p.without_after.tag == "cmdbuf_copy_0")
    for step in range(3):
      run_linear(linked, jit=True)
      self.assertEqual([cmdbuf.host.view(fmt="I")[o // 4] for o in offs], [step] * 10)

if __name__ == "__main__":
  unittest.main()
