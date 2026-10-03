import unittest, threading, functools
from tinygrad import Tensor, UOp, Context
from tinygrad.device import Device, Buffer, BufferSpec
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.engine.realize import run_linear
from tinygrad.uop.ops import Ops, KernelInfo
from tinygrad.renderer.isa.x86 import X86Renderer

def wait_loop_kernel(C:UOp, N=10) -> UOp:
  # a RANGE with no src is a bound-less loop header: a jump target with no induction variable.
  # the compare and conditional backedge are expanded by the renderers from the loop RANGE/END
  l = UOp.loop(0)

  i = UOp.placeholder((1,), dtypes.int, 0, addrspace=AddrSpace.REG)

  # i = 0
  i = i.after(i[0].store(0))

  # i + 1, read loop-carried through after(l)
  inc = i.after(l)[0].load() + 1

  # i = inc; END(store, l, cond): conditional backedge, loop again while inc < N (do-while)
  # NOTE: the cond uses the computed value, not a reload of the register
  st = i[0].store(inc)
  i = i.after(st.end(l, inc < N))

  return C[0].store(i[0].load()).sink(arg=KernelInfo(name="wait_loop"))

def nested_loop_kernel(C:UOp) -> UOp:
  r = UOp.range(4, 0)
  l = UOp.loop(1)

  i = UOp.placeholder((1,), dtypes.int, 0, addrspace=AddrSpace.REG)
  i = i.after(i[0].store(0))

  inc = i.after(l, r)[0].load() + 1
  st = i[0].store(inc)

  lend = st.end(l, inc < (r.cast(dtypes.int)+1)*3)
  i = i.after(lend.end(r))

  return C[0].store(i[0].load()).sink(arg=KernelInfo(name="nested_loop", opts_to_apply=()))

def pressure_loop_kernel(C:UOp, n=13) -> UOp:
  vs = [C[j+1].load() for j in range(n)]
  l = UOp.loop(0)

  i = UOp.placeholder((1,), dtypes.int, 0, addrspace=AddrSpace.REG)
  i = i.after(i[0].store(0))

  inc = i.after(l)[0].load() + 1
  st = i[0].store(inc)
  i = i.after(st.end(l, inc < sum(v & inc for v in vs)))

  return C[0].store(i[0].load()).sink(arg=KernelInfo(name="pressure_loop", opts_to_apply=()))

def wait_ext_kernel() -> UOp:
  sig = UOp.param(0, dtypes.int, 1, volatile=True)
  l = UOp.loop(0)
  v = sig.after(l)[0].load()
  e = v.end(l, v < 1)
  return e.sink(arg=KernelInfo(name="wait_ext"))

def two_loops_kernel(C:UOp) -> UOp:
  # two sequential loops on the same counter: ++ until 10, then ++ until 25
  l1, l2 = UOp.loop(0), UOp.loop(1)

  i = UOp.placeholder((1,), dtypes.int, 0, addrspace=AddrSpace.REG)
  i = i.after(i[0].store(0))

  inc1 = i.after(l1)[0].load() + 1
  i = i.after(i[0].store(inc1).end(l1, inc1 < 10))

  inc2 = i.after(l2)[0].load() + 1
  i = i.after(i[0].store(inc2).end(l2, inc2 < 25))

  return C[0].store(i[0].load()).sink(arg=KernelInfo(name="two_loops", opts_to_apply=()))

def loop_in_loop_kernel(C:UOp) -> UOp:
  # outer loop while i < 12, inner loop increments until i % 4 == 0 -> 12
  l1, l2 = UOp.loop(0), UOp.loop(1)

  i = UOp.placeholder((1,), dtypes.int, 0, addrspace=AddrSpace.REG)
  i = i.after(i[0].store(0))

  inc = i.after(l1, l2)[0].load() + 1
  st = i[0].store(inc)

  # the outer END closes the inner END, and its cond reloads the register after the inner loop (in scope at the outer level)
  e2 = st.end(l2, inc % 4 != 0)
  oc = i.after(e2)[0].load()
  i = i.after(e2.end(l1, oc < 12))

  return C[0].store(i[0].load()).sink(arg=KernelInfo(name="loop_in_loop", opts_to_apply=()))

class TestWaitLoop(unittest.TestCase):
  def test_wait_loop(self):
    c = Tensor.empty(1, dtype=dtypes.int)
    c = Tensor.custom_kernel(c, fxn=wait_loop_kernel)[0]
    c.realize()
    self.assertEqual(c.item(), 10)

  def test_nested_loop_in_range(self):
    c = Tensor.empty(1, dtype=dtypes.int)
    c = Tensor.custom_kernel(c, fxn=nested_loop_kernel)[0]
    c.realize()
    self.assertEqual(c.item(), 12)

  def test_two_sequential_loops(self):
    c = Tensor.empty(1, dtype=dtypes.int)
    c = Tensor.custom_kernel(c, fxn=two_loops_kernel)[0]
    c.realize()
    self.assertEqual(c.item(), 25)

  # TODO: x86's lower_loop builds an Ops.IF node after regalloc, which fails spec_full
  @(unittest.expectedFailure if isinstance(Device[Device.DEFAULT].renderer, X86Renderer) else lambda f: f)
  def test_wait_loop_spec(self):
    c = Tensor.custom_kernel(Tensor.empty(1, dtype=dtypes.int), fxn=functools.partial(wait_loop_kernel, N=7))[0]
    with Context(SPEC=2): c.realize()
    self.assertEqual(c.item(), 7)

  @unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, X86Renderer), "TODO: do-while loop under register pressure segfaults on x86")
  def test_loop_carried_registers(self):
    # more loads live across the backedge than any register file (x86 15 gprs, arm64 31, sass 255, rdna3 256 vgprs)
    c = Tensor.custom_kernel(Tensor.ones(301, dtype=dtypes.int), fxn=functools.partial(pressure_loop_kernel, n=300))[0]
    self.assertEqual(c[0].item(), 2)

  def test_register_pressure_loop(self):
    c = Tensor.zeros(16, dtype=dtypes.int).contiguous()
    c = Tensor.custom_kernel(c, fxn=pressure_loop_kernel)[0]
    c.realize()
    self.assertEqual(c[0].item(), 1)

  def test_loop_in_loop(self):
    c = Tensor.empty(1, dtype=dtypes.int)
    c = Tensor.custom_kernel(c, fxn=loop_in_loop_kernel)[0]
    c.realize()
    self.assertEqual(c.item(), 12)

@unittest.skipUnless(Device.DEFAULT in ("CPU", "AMD", "NV"), "need proper uncached=True handling")
class TestVolatileLoops(unittest.TestCase):
  def test_async_wait_ext(self):
    sig_buf = Buffer(Device.DEFAULT, 1, dtypes.int, options=BufferSpec(host=True, uncached=True, cpu_access=True), preallocate=True)
    try: sig_view = sig_buf.host.view(fmt='i')
    except (AssertionError, NotImplementedError): self.skipTest(f"{Device.DEFAULT} does not support host-visible buffers")
    sig_view[0] = 0

    def set_signal():
      threading.Event().wait(0.3)
      sig_view[0] = 1

    sync = threading.Thread(target=set_signal, daemon=True)
    sync.start()
    run_linear(UOp(Ops.LINEAR, src=(wait_ext_kernel().call(UOp.from_buffer(sig_buf)),)), wait=True)
    sync.join(timeout=3)


if __name__ == "__main__": unittest.main()
