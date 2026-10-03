import unittest, contextlib, ctypes, gc, struct, numpy as np
from unittest.mock import patch
from tinygrad import Device, Tensor, TinyJit, Variable, dtypes, GlobalCounters
from tinygrad.device import Buffer, Compiled
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import Context, dedup, partition, unwrap
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher, KernelInfo
from tinygrad.engine.realize import compile_linear, link_linear, lower_and_compile, run_linear
from tinygrad.codegen import do_to_program
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.runtime.autogen import libc
from tinygrad.runtime.support.c import init_c_struct_t
import tinygrad.runtime.support.hcq2 as hcq2
from tinygrad.runtime.support.hcq2 import HCQ_DEVS, all_devices_in, hcq_compile_cache, link_linear_cache
from test.helpers import call_is_hcq

@contextlib.contextmanager
def rt_buffers():
  calls, orig = [], Compiled.rt_buffer
  def track(dev, *args, **kwargs):
    calls.append(dev)
    return orig(dev, *args, **kwargs)
  with patch.object(Compiled, "rt_buffer", track): yield calls

def chain(x:Tensor, n:int) -> Tensor:
  for _ in range(n): x = (x + 1).contiguous()
  return x

@contextlib.contextmanager
def encoded_batches():
  batches, orig = [], hcq2.lower_and_compile
  def track(l, *args, **kwargs):
    batches.extend(c.without_after for c in l.src if call_is_hcq(c))
    return orig(l, *args, **kwargs)
  with patch.object(hcq2, "lower_and_compile", track): yield batches

def eager_chain(x:Tensor, n:int=64) -> Tensor: # at hcq_compile's use_rt bound: an eager linear this big bakes its inputs and borrows ring slots
  for _ in range(n): x = (x + 1).contiguous()
  return x.realize()

def patch_words(batch:UOp) -> list[UOp]:
  return [w for s in batch.src[0].toposort() if s.op is Ops.STORE and s.src[0].op is Ops.INDEX and s.src[0].src[1].op is Ops.STACK
          and s.src[1].op is Ops.STACK for w in s.src[1].src]

def rt_params(batch:UOp) -> list[str]:
  return dedup([u.arg.name for w in patch_words(batch) for u in w.toposort() if u.op is Ops.PARAM and u.arg.addrspace is AddrSpace.GLOBAL])

def cpu_buf(size:int=1, dtype=dtypes.uint8, **kwargs) -> UOp: return UOp.placeholder((size,), dtype, device="CPU", **kwargs)

def lower_hcq(body:UOp) -> UOp:
  return unwrap(hcq2.lower_call(UOp.sink(body, arg=KernelInfo("test")).call(aux=hcq2.HCQInfo(("CPU",)))))

class TestHCQ2Deps(unittest.TestCase):
  def test_copy_only_batch_with_multiple_queues(self):
    from types import SimpleNamespace
    bufs = [UOp.param(i, dtypes.uint8, 16, device="AMD") for i in range(4)]
    calls = [(src.copy_to_device("AMD").call(dst, src), ("AMD",), f"COPY:{i}") for i, (dst, src) in enumerate(zip(bufs[:2], bufs[2:]))]
    with patch.object(type(Device), "__getitem__", return_value=SimpleNamespace(pm_batch=None)):
      batch = hcq2._finalize_batch(hcq2.BatchCtx(calls, False))
    streams = [s.without_after.src[0] for s in batch.src[0].src]
    self.assertEqual([s.arg[1] for s in streams], ["COPY:0", "COPY:1", "COMPUTE:0"])
    self.assertEqual([u.arg[0] for u in streams[-1].src], ["wait", "wait", "store"])

  def test_peer_access_syncs_both_ways(self):
    from types import SimpleNamespace
    dst, src = UOp.param(0, dtypes.uint8, 16, device="AMD:1"), UOp.param(1, dtypes.uint8, 16, device="AMD")
    with patch.object(type(Device), "__getitem__", return_value=SimpleNamespace(pm_batch=None)):
      batch = hcq2._finalize_batch(hcq2.BatchCtx([(src.copy_to_device("AMD:1").call(dst, src), ("AMD",), "COPY:0")], False))
    streams = {s.without_after.src[0].arg[1]: [u.arg[0] for u in s.without_after.src[0].src if u.op is Ops.INS] for s in batch.src[0].src}
    # the copy queue waits for its device and for the peer, then signals and bumps. the peer waits for the signal before its bump
    self.assertEqual(streams, {"COPY:0": ["barrier", "wait", "wait", "store", "store"], "COMPUTE:0": ["barrier", "wait", "wait", "store"]})

  def test_dependencies_through_selected_slices(self):
    b = UOp.param(0, dtypes.float32, 64, device=("AMD", "AMD:1"))
    for view in [b.mselect(0).shrink(((8, 16),)), b.shrink(((8, 16),)).mselect(0), b.shrink(((4, 32),)).mselect(0).shrink(((4, 12),))]:
      tracker = hcq2.HCQDepsTracker()
      tracker.access_resources([view], [0], 0)
      self.assertEqual(tracker.access_resources([b.mselect(1)], [], 1), [])
      self.assertEqual(tracker.access_resources([b.mselect(0).shrink(((16, 24),))], [], 2), [])
      self.assertEqual(tracker.access_resources([b.mselect(0).shrink(((12, 20),))], [], 3), [0])

  def test_disjoint_write_preserves_dependencies(self):
    b = UOp.param(0, dtypes.uint8, 16, device="CPU")
    for write in ([], [0]):
      tracker = hcq2.HCQDepsTracker()
      tracker.access_resources([b.shrink(((0, 4),))], write, 0)
      self.assertEqual(tracker.access_resources([b.shrink(((4, 8),))], [0], 1), [])
      self.assertEqual(tracker.access_resources([b.shrink(((0, 4),))], [0], 2), [0])

  def test_partial_write_preserves_dependencies(self):
    b = UOp.param(0, dtypes.uint8, 16, device="CPU")
    for write in ([], [0]):
      tracker = hcq2.HCQDepsTracker()
      tracker.access_resources([b], write, 0)
      self.assertEqual(tracker.access_resources([b.shrink(((4, 12),))], [0], 1), [0])
      self.assertEqual(tracker.access_resources([b.shrink(((0, 4),))], [0], 2), [0])
      self.assertEqual(tracker.access_resources([b.shrink(((12, 16),))], [0], 3), [0])
      self.assertEqual(tracker.access_resources([b.shrink(((4, 12),))], [], 4), [1])

@unittest.skipUnless(all_devices_in(Device.DEFAULT, HCQ_DEVS), "hcq2 device required")
class TestHCQ2Schedule(unittest.TestCase):
  @staticmethod
  def input(value:int=2) -> Tensor: return Tensor.full((4,), value, dtype=dtypes.int32).contiguous().realize()

  def compiled(self, n:int, jit=False):
    x, inputs = self.input(), []
    if jit:
      f = TinyJit(lambda a: chain(a, n).realize())
      f(x)
      return f(x), f.captured._linear, [x.uop.base]
    out = chain(x, n)
    return out, compile_linear(out.schedule_linear(), input_uops=inputs, cache=True), inputs

  def test_jit_has_no_rt_buffers(self):
    dev = Device[Device.DEFAULT]
    rings = [dev.rt_buffer(True, host) for host in (False, True)]
    ranges = [(b._buf, b._buf + b.nbytes) for b in rings]
    for n in (1, 65):
      with self.subTest(kernels=n):
        x, f = self.input(), TinyJit(lambda a: chain(a, n).realize())
        for _ in range(2): f(x)
        for u in f.captured.linear.toposort():
          if u.op is Ops.BUFFER and (buf:=u.buffer).device == dev.device:
            addr = buf._buf
            self.assertFalse(any(addr < end and start < addr + buf.nbytes for start, end in ranges))

  def test_amd_cmdbuf_uncached(self):
    dev = Device[Device.DEFAULT]
    if not dev.device.startswith("AMD") or not dev.is_am(): self.skipTest("AMD PCI interface required")
    for name, uncached in (("cmdbuf", True), ("kernargs", False)):
      b = UOp.placeholder((256,), dtypes.uint8, device=(dev.device,), tag=hcq2.to_name(name, "COMPUTE:0"))
      buf = unwrap(hcq2.bufferize_buf(hcq2.LinkCtx({}, use_rt=False), b)).buffer
      self.assertEqual(buf.base.options.uncached, uncached)
      self.assertEqual(buf.base.meta.mapping.uncached, uncached)

  def test_small_eager_cached(self):
    _, compiled, inputs = self.compiled(1)
    linked = link_linear(compiled, input_uops=inputs)
    self.assertIs(link_linear(compiled, input_uops=inputs), linked)

  def test_profile_slots_survive_indirect_access(self):
    pm = PatternMatcher([(UPat((Ops.LOAD, Ops.STORE), src=(UPat(Ops.INDEX, src=(UPat.var("buf"), UPat())),), allow_any_len=True),
                          lambda buf: hcq2.rt_addr(buf, "CPU") if hcq2.unwrap_view(buf)[0].tag == "slots" else None)])
    with patch.object(Device[Device.DEFAULT], "pm_lower", pm):
      compiled = compile_linear(Tensor.ones(4).contiguous().schedule_linear(), profile=True)
    self.assertFalse(any(param.op is Ops.PARAM and (param.arg.name or "").startswith("slots_")
                         for param in compiled.src[0].without_after.src[0].toposort()))
    call = link_linear(compiled).src[0].without_after
    ((device, index),) = call.arg.aux.slots
    self.assertEqual(device, Device.DEFAULT)
    self.assertEqual(call.src[1 + index].buffer.dtype, dtypes.uint64)

  def test_large_eager_not_cached(self):
    _, compiled, inputs = self.compiled(65)
    linked = link_linear(compiled, input_uops=inputs)
    self.assertIsNot(link_linear(compiled, input_uops=inputs), linked)
    self.assertNotIn(compiled, link_linear_cache)

  def test_double_compile(self):
    for n in (1, 65):
      for jit in (False, True):
        with self.subTest(kernels=n, jit=jit):
          out, compiled, inputs = self.compiled(n, jit=jit)
          linked = link_linear(compiled, input_uops=inputs, allow_cache=not jit)
          before = tuple(inputs)
          with rt_buffers() as borrowed:
            for linear in (compiled, linked):
              self.assertIs(compile_linear(linear, input_uops=inputs, cache=not jit), linear)
          self.assertEqual(tuple(inputs), before)
          self.assertFalse(borrowed)
          run_linear(linked, input_uops=inputs, jit=True, wait=True)
          self.assertEqual(out.tolist(), [2 + n] * 4)

  def test_double_link(self):
    for n in (1, 65):
      for jit in (False, True):
        with self.subTest(kernels=n, jit=jit):
          out, compiled, inputs = self.compiled(n, jit=jit)
          linked = link_linear(compiled, input_uops=inputs, allow_cache=not jit)
          with rt_buffers() as borrowed:
            again = link_linear(linked, input_uops=inputs, allow_cache=not jit)
          self.assertIs(again, linked)
          self.assertFalse(borrowed)
          run_linear(again, input_uops=inputs, jit=True, wait=True)
          self.assertEqual(out.tolist(), [2 + n] * 4)

  def test_jit_new_inputs_each_call(self):
    @TinyJit
    def f(a, b): return (a * b + a).contiguous().realize()
    ins = [(Tensor.full((23,), float(i)).contiguous().realize(), Tensor.full((23,), 2.0).contiguous().realize()) for i in range(6)]
    for a, b in ins[:3]: f(a, b).tolist() # warm the jit and the copyout

    before = len(hcq_compile_cache)
    self.assertEqual([f(a, b).tolist() for a, b in ins[3:]], [[i * 3.0] * 23 for i in range(3, 6)])
    self.assertEqual(len(hcq_compile_cache), before)

  def test_jit_symbolic(self):
    @TinyJit
    def f(a): return (a + 1).sum().contiguous().realize()
    a = Tensor.rand(3, 10).contiguous().realize()
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      np.testing.assert_allclose(f(a[:, :vi]).item(), (a[:, :i] + 1).sum().item(), atol=1e-5, rtol=1e-5)

  def test_map_cpu_buffer_preserves_contents(self):
    src = Buffer("CPU", 16, dtypes.uint8, preallocate=True)
    data = bytes(range(16))
    src.host[:] = data
    src.get_buf(Device.DEFAULT)
    self.assertEqual(bytes(src.as_memoryview()), data)

  def test_rt_patches_are_inputs_and_vars_only(self):
    x = Tensor.rand(17, 33).contiguous().realize()
    with encoded_batches() as batches:
      @TinyJit
      def f(a): return (a.sin() * 3).contiguous().realize()
      for _ in range(3): f(x)
      eager_chain(x)

    jit, eager = partition(batches, lambda c: c.arg.aux.table >= 0)
    self.assertTrue(jit and eager, f"want both kinds of batch, got {len(jit)} jit and {len(eager)} eager")
    for c in batches:
      self.assertTrue(all(n.startswith(("inputs_", "timeline_")) for n in rt_params(c)), f"runtime patch reads {rt_params(c)}")
      self.assertFalse([u for w in patch_words(c) for u in w.toposort() if u.op is Ops.GETADDR], "addresses bake at link time")
    self.assertTrue(any(n.startswith("inputs_") for c in jit for n in rt_params(c)), "the jit patches its input addresses in")
    self.assertFalse(any(n.startswith("inputs_") for c in eager for n in rt_params(c)), "eager bakes its input addresses")

  def test_programs_are_not_call_args(self):
    # a program is a link-time patch a cmdbuf word addresses: it rides inside that word, no arg or param of its own
    def nargs(n):
      x = Tensor.ones(16).contiguous().realize()
      with encoded_batches() as batches:
        @TinyJit
        def f(a):
          for i in range(n): a = (a * (i + 1.5)).contiguous()
          return a.realize()
        for _ in range(3): f(x)
      return max(c.arg.aux.nargs for c in batches)
    self.assertEqual(nargs(2), nargs(12))

  def test_caches_hold_no_buffers(self):
    # an eager template caches without its buffers and the jit's linear compiles once uncached: freeing the tensors frees the device memory
    def step(i):
      buf = Buffer("NPY", 1024, dtypes.float32, initial_value=struct.pack("f", i) * 1024)
      x = Tensor(UOp.from_buffer(buf)).to(Device.DEFAULT).realize()
      @TinyJit
      def f(a): return (a * 2 + 1).contiguous().realize()
      for _ in range(3): out = f(x)
      self.assertEqual(out.to("CPU").tolist(), [2.0 * i + 1] * 1024)
    step(1) # warms the programs, templates and rings
    gc.collect()
    used = GlobalCounters.mem_used
    for i in range(2, 5): step(i)
    gc.collect()
    self.assertEqual(GlobalCounters.mem_used, used)

  def test_device_state_survives_as_link_refs(self):
    # a buffer the commands only address, never a param of the body, is kept by the linked call as a ref of what its getaddr resolved into
    dev = Device[Device.DEFAULT]
    names = {"AMD": () if getattr(dev, "is_aql", False) else ("scratch",), # the aql descriptor holds the scratch, nothing addresses it
             "NV": ("timeline",), "QCOM": ("_stack", "dummy"), "CUDA": ("timeline",)}[Device.DEFAULT.split(":")[0]]
    @TinyJit
    def f(a): return (a * 2 + 1).contiguous().realize()
    x = Tensor.ones(16).contiguous().realize()
    for _ in range(3): f(x)
    call = f.captured.linear.src[0]
    self.assertIs(call.op, Ops.AFTER, "the linked call sits after its refs")
    refs = [u.buffer for u in call.src[1:] if u.op is Ops.BUFFER]
    for n in names: self.assertTrue(any(r is getattr(dev, n) for r in refs), f"{n} is not a ref of the call")

  def test_usb_renumbering(self):
    programs = []
    with Context(HCQ_RUNTIME_DEV="CPU"), patch("tinygrad.codegen.do_to_program", wraps=do_to_program) as build:
      for ids in ((0, 1, 2, 3), (2, 0, 3, 1), (1, 0, 2, 3), (0, 1, 3, 2), (100, 101, 102, 103)):
        with self.subTest(ids=ids):
          regs = [UOp.placeholder((1,), dtypes.uint32, slot=i, addrspace=AddrSpace.REG) for i in ids[:2]]
          a, b = [r.after(r.index(0).store(v)) for r, v in zip(regs, (3, 5))]
          i, j = [UOp.range(UOp(Ops.NOOP), n, dtype=dtypes.void, src=(a, b)) for n in ids[2:]]
          out = cpu_buf(dtype=dtypes.uint32, tag="out")
          body = out.index(0).store(a.after(i, j).index(0).load()*10 + b.index(0).load()).end(j, UOp.const(False)).end(i, UOp.const(False))
          compiled = lower_and_compile(UOp(Ops.LINEAR, src=(lower_hcq(body),)))
          programs.append(compiled.src[0].without_after.src[0])
          self.assertIs(programs[-1], programs[0])
          linear = hcq2.hcq_link(compiled, allow_cache=False)
          run_linear(linear, jit=True)
          self.assertEqual(linear.src[0].without_after.src[1].buffer.host.view(fmt='I')[0], 35)
      self.assertLessEqual(build.call_count, 1)

  def test_patched_view(self):
    with Context(HCQ_RUNTIME_DEV="CPU"):
      ctx = hcq2.EncodeCtx(("CPU",))
      inner = hcq2.patch(cpu_buf(8, tag="inner"), [(4, UOp.const(42, dtypes.uint32))], bytes(8))
      inner = unwrap(hcq2.hoist_links(ctx, inner))
      outer = hcq2.patch(cpu_buf(8, tag="outer"), [(0, inner[4:8].getaddr("CPU"))])
      with patch.object(hcq2, "EncodeCtx", return_value=ctx): call = lower_hcq(outer.bitcast(dtypes.uint64).index(0).load())
      self.assertEqual(call.without_after.arg.aux.nargs, 1)
      self.assertTrue(all(s.op is Ops.STORE for s in call.src[1:]))
      linked = hcq2.hcq_link(UOp(Ops.LINEAR, src=(call,)), allow_cache=False).src[0]
      inner_buf, outer_buf = linked.src[1].buffer, linked.without_after.src[1].buffer
      self.assertEqual(inner_buf.host.view(fmt='I')[1], 42)
      self.assertEqual(outer_buf.host.view(fmt='Q')[0], inner_buf._buf + 4)

@unittest.skipUnless(isinstance(Device["CPU"].renderer, CStyleLanguage), "CALL is rendered in C style only")
class TestHCQ2FFI(unittest.TestCase):
  @staticmethod
  def _run(body:UOp) -> list[Buffer]:
    linear = hcq2.hcq_link(lower_and_compile(UOp(Ops.LINEAR, src=(lower_hcq(body),))), allow_cache=False)
    run_linear(linear, jit=True)
    return [u.buffer for u in linear.src[0].without_after.src[1:] if u.op is Ops.BUFFER]

  def test_ffi_ccall(self):
    with Context(HCQ_RUNTIME_DEV="CPU"):
      out = cpu_buf(dtype=dtypes.int32, slot=1, volatile=True, tag="ffi_result")
      bufs = self._run(out.index(0).store(hcq2.ccall(libc.dll.ffs, 0x10)))
    self.assertEqual(next(b for b in bufs if b.dtype is dtypes.int).host.view(fmt='i')[0], 5)

  def test_ffi_cstruct(self):
    struct_t = init_c_struct_t(16, (("u8", ctypes.c_uint8, 0), ("u16", ctypes.c_uint16, 2),
                                  ("u32", ctypes.c_uint32, 4), ("u64", ctypes.c_uint64, 8)))
    cpu_buf() # reserve slot zero for device-owned placeholders
    with Context(HCQ_RUNTIME_DEV="CPU"):
      s = hcq2.cstruct(struct_t, u8=0x12, u16=UOp.const(0x3456, dtypes.uint16), u32=0x789ABCDE, u64=0xFEDCBA9876543210)
      bufs = self._run(s.index(0).load())
    got = struct_t.from_buffer_copy(bytes(next(b for b in bufs if b.nbytes == ctypes.sizeof(struct_t)).host.view(fmt='B')))
    self.assertEqual((got.u8, got.u16, got.u32, got.u64), (0x12, 0x3456, 0x789ABCDE, 0xFEDCBA9876543210))

  def test_nested_cstruct_patches(self):
    with Context(HCQ_RUNTIME_DEV="CPU"):
      inner = hcq2.cstruct(init_c_struct_t(4, (("value", ctypes.c_uint32, 0),)), value=42)
      outer = hcq2.cstruct(init_c_struct_t(8, (("ptr", ctypes.c_uint64, 0),)), ptr=inner.getaddr("CPU"))
      out = cpu_buf(dtype=dtypes.uint32, tag="result")
      copied = hcq2.ccall(libc.memcpy, out.index(0), outer.bitcast(dtypes.uint64).index(0).load(), 4)
      bufs = self._run(out.after(copied).index(0).load())
    self.assertEqual(next(b for b in bufs if b.dtype is dtypes.uint32).host.view(fmt='I')[0], 42)

if __name__ == "__main__":
  unittest.main()
