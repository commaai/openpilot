import unittest, gc, struct, ctypes, threading, numpy as np
from tinygrad import Device, Tensor, TinyJit, Variable, dtypes, GlobalCounters
from tinygrad.device import Buffer, BufferSpec
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import Context
from tinygrad.uop.ops import Ops, UOp, uopfunc
from tinygrad.engine.realize import compile_linear, link_linear, lower_and_compile, run_linear
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.renderer.llvmir import LLVMRenderer
from tinygrad.runtime.autogen import libc
from tinygrad.runtime.support.c import init_c_struct_t
import tinygrad.runtime.support.hcq2 as hcq2
from tinygrad.runtime.support.hcq2 import HCQ_DEVS, all_devices_in, hcq_compile_cache
from test.null.test_hcq2 import chain, chain_input, compiled_chain, lower_hcq

def cpu_buf(size:int=1, dtype=dtypes.uint8, tag=None, **kwargs) -> UOp: return UOp.alloc((size,), dtype, device="CPU", **kwargs).rtag(tag)

@unittest.skipUnless(all_devices_in(Device.DEFAULT, HCQ_DEVS) and not Device.DEFAULT.startswith("NULL"), "hcq2 device required")
class TestHCQ2Schedule(unittest.TestCase):
  def test_compile_and_link_are_idempotent(self):
    for jit in (False, True):
      with self.subTest(jit=jit):
        out, compiled, inputs = compiled_chain(2, jit=jit, device=Device.DEFAULT)
        linked = link_linear(compiled, input_uops=inputs, allow_cache=not jit)
        before = tuple(inputs)
        for linear in (compiled, linked): self.assertIs(compile_linear(linear, input_uops=inputs, cache=not jit), linear)
        self.assertIs(link_linear(linked, input_uops=inputs, allow_cache=not jit), linked)
        self.assertEqual(tuple(inputs), before)
        run_linear(linked, input_uops=inputs, jit=True, wait=True)
        self.assertEqual(out.tolist(), [4] * 4)

  def test_jit_new_inputs_each_call(self):
    @TinyJit
    def f(a, b): return (a * b + a).contiguous().realize()
    ins = [(Tensor.full((23,), float(i)).contiguous().realize(), Tensor.full((23,), 2.0).contiguous().realize()) for i in range(6)]
    for a, b in ins[:3]: f(a, b).tolist() # warm the jit and the copyout

    before = len(hcq_compile_cache)
    for i, (a, b) in enumerate(ins[3:], 3): self.assertEqual(f(a, b).tolist(), [i * 3.0] * 23)
    self.assertEqual(len(hcq_compile_cache), before)

  def test_jit_symbolic(self):
    @TinyJit
    def f(a): return (a + 1).sum().contiguous().realize()
    a = Tensor.rand(3, 10).contiguous().realize()
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      np.testing.assert_allclose(f(a[:, :vi]).item(), (a[:, :i] + 1).sum().item(), atol=1e-5, rtol=1e-5)

  def test_repeated_copy(self):
    vram, host, new = [Buffer(d, 4096, preallocate=True) for d in (Device.DEFAULT, "CPU", "CPU")]
    new.host[:] = bytes(range(256)) * 16
    hu, vu, nu = [UOp.from_buffer(b, dtypes.uint8) for b in (host, vram, new)]
    copyout, copyin = hu.store_call(vu), vu.store_call(nu)
    run_linear(UOp(Ops.LINEAR, src=(copyout, copyin, copyout)), wait=True)
    self.assertEqual(bytes(host.host[:]), bytes(new.host[:]))

  @unittest.skipIf(Device.DEFAULT == "METAL", "unified memory: METAL copies on the host and maps nothing")
  def test_map_cpu_buffer_preserves_contents(self):
    src = Buffer("CPU", 16, preallocate=True)
    data = bytes(range(16))
    src.host[:] = data
    src.get_buf(Device.DEFAULT)
    self.assertEqual(bytes(src.as_memoryview()), data)

  def test_caches_hold_no_buffers(self):
    # an eager template caches without its buffers and the jit's linear compiles once uncached: freeing the tensors frees the device memory
    def step(i):
      buf = Buffer("NPY", 4096, initial_value=struct.pack("f", i) * 1024)
      x = Tensor(UOp.from_buffer(buf, dtypes.float32)).to(Device.DEFAULT).realize()
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

  def test_jit_has_no_rt_buffers(self):
    # a one shot link borrows ring slots, a jit's link owns its buffers: nothing it keeps may come from the ring
    dev = Device[Device.DEFAULT]
    specs = (BufferSpec(cpu_access=True), BufferSpec(host=True, uncached=True, cpu_access=True)) # device data, signals
    ranges = [((b:=dev.rt_buffer(spec))._buf, b._buf + b.nbytes) for spec in specs]
    x, f = chain_input(device=dev.device), TinyJit(lambda a: chain(a, 2).realize())
    for _ in range(2): f(x)
    for u in f.captured.linear.toposort(enter_calls=True):
      if u.op is Ops.BUFFER and u.addrspace is AddrSpace.GLOBAL and (buf:=u.buffer).device == dev.device:
        self.assertFalse(any(buf._buf < end and start < buf._buf + buf.nbytes for start, end in ranges))

# the fence and the ffi run on the CPU runtime, which is always available
@unittest.skipUnless(isinstance(Device["CPU"].renderer, (CStyleLanguage, LLVMRenderer)), "CALL is rendered in C style and LLVM only")
class TestHCQ2Fence(unittest.TestCase):
  def setUp(self):
    self.enterContext(Context(HCQ_RUNTIME_DEV="CPU"))
    self.tl = Device["CPU"].timeline.host.view(fmt='Q')
    self.addCleanup(lambda: self.tl.__setitem__(0, self.tl[1]))

  def test_a_schedule_waits_for_its_previous_run(self):
    slots = UOp.alloc((4,), dtypes.uint64, device="CPU").rtag("slots")
    program = lower_and_compile(lower_hcq(UOp.custom_function("hcq_fence").call(slots[0:2], slots[2:4])))
    linked = hcq2.hcq_link(program, allow_cache=False)
    (i,) = [i for i, p in enumerate(program.src[0].without_after.src[1:]) if p.without_after.tag == "slots"]
    slots_mv = linked.src[0].without_after.src[1 + i].buffer.host.view(fmt='Q')
    slots_mv[2], base = 7, self.tl[1]

    run_linear(linked, jit=True)
    self.assertEqual((self.tl[1], slots_mv[0], slots_mv[2]), (base + 1, base + 1, 0), "the run is announced, recorded, the signal re-armed")

    t = threading.Thread(target=run_linear, args=(linked,), kwargs={"jit": True}, daemon=True)
    t.start()
    t.join(0.2)
    self.assertTrue(t.is_alive(), "the second run must wait for the first to finish")
    self.tl[0] = base + 1
    t.join(5)
    self.assertFalse(t.is_alive())
    self.assertEqual(self.tl[1], base + 2)

@unittest.skipUnless(isinstance(Device["CPU"].renderer, (CStyleLanguage, LLVMRenderer)), "CALL is rendered in C style and LLVM only")
class TestHCQ2FFI(unittest.TestCase):
  @staticmethod
  def _run(body:UOp) -> list[UOp]:
    linear = hcq2.hcq_link(lower_and_compile(lower_hcq(body)), allow_cache=False)
    assert hcq2.hcq_link(linear, allow_cache=False) is linear, "a linked linear links to itself, with the refs its call keeps"
    run_linear(linear, jit=True)
    return [u for u in linear.src[0].without_after.src[1:] if u.op is Ops.BUFFER]

  def test_ffi_ccall(self):
    with Context(HCQ_RUNTIME_DEV="CPU"):
      out = cpu_buf(dtype=dtypes.int32, slot=1, tag="ffi_result")
      bufs = self._run(out.index(0).store(hcq2.ccall(libc.dll.ffs, 0x10)))
    self.assertEqual(next(b for b in bufs if b.dtype is dtypes.int).buffer.host.view(fmt='i')[0], 5)

  def test_nested_ffi_call(self, host="CPU"): # a function calls a C function: no pointer to pass, the symbol links
    @uopfunc
    def copy(dst:UOp, src:UOp): return hcq2.ccall(libc.memcpy, dst.index(0), src.index(0), 4).sink()
    @uopfunc
    def copy_pair(dst:UOp, src:UOp): return copy(dst.after(copy(dst, src)).index(1), src).sink()

    with Context(HCQ_RUNTIME_DEV=host):
      src = hcq2.cstruct(init_c_struct_t(4, (("value", ctypes.c_uint32, 0),)), value=42)
      out = cpu_buf(2, dtypes.uint32, tag="ffi_result")
      bufs = self._run(copy_pair(out, src.bitcast(dtypes.uint32)))
    self.assertEqual(list(next(b for b in bufs if b.dtype is dtypes.uint32).buffer.host.view(fmt='I')), [42, 42])
  def test_nested_ffi_call_python(self): self.test_nested_ffi_call("PYTHON")

  def test_ffi_cstruct(self):
    struct_t = init_c_struct_t(16, (("u8", ctypes.c_uint8, 0), ("u16", ctypes.c_uint16, 2),
                                  ("u32", ctypes.c_uint32, 4), ("u64", ctypes.c_uint64, 8)))
    cpu_buf() # reserve slot zero for device-owned placeholders
    with Context(HCQ_RUNTIME_DEV="CPU"):
      s = hcq2.cstruct(struct_t, u8=0x12, u16=UOp.const(0x3456, dtypes.uint16), u32=0x789ABCDE, u64=0xFEDCBA9876543210)
      bufs = self._run(s.index(0).load())
    got = struct_t.from_buffer_copy(bytes(next(b for b in bufs if b.nbytes() == ctypes.sizeof(struct_t)).buffer.host.view(fmt='B')))
    self.assertEqual((got.u8, got.u16, got.u32, got.u64), (0x12, 0x3456, 0x789ABCDE, 0xFEDCBA9876543210))

  def test_nested_cstruct_patches(self):
    with Context(HCQ_RUNTIME_DEV="CPU"):
      inner = hcq2.cstruct(init_c_struct_t(8, (("pad", ctypes.c_uint32, 0), ("value", ctypes.c_uint32, 4))), value=42)
      outer = hcq2.cstruct(init_c_struct_t(8, (("ptr", ctypes.c_uint64, 0),)), ptr=inner[4:8].getaddr("CPU"))
      out = cpu_buf(dtype=dtypes.uint32, tag="result")
      copied = hcq2.ccall(libc.memcpy, out.index(0), outer.bitcast(dtypes.uint64).index(0).load(), 4)
      bufs = self._run(out.after(copied).index(0).load())
    self.assertEqual(next(b for b in bufs if b.dtype is dtypes.uint32).buffer.host.view(fmt='I')[0], 42)

@uopfunc
def addr_of(o:UOp, b:UOp): return o.index(0).store(b.getaddr("CPU")).sink() # o[0] = &b

# host functions in a batch
@unittest.skipUnless(isinstance(Device["CPU"].renderer, (CStyleLanguage, LLVMRenderer)), "CALL is rendered in C style and LLVM only")
class TestHostCalls(unittest.TestCase):
  def setUp(self): self.enterContext(Context(HCQ_RUNTIME_DEV="CPU"))

  @staticmethod
  def _buf(n:int, dtype=dtypes.uint64) -> UOp:
    return UOp.from_buffer(Buffer("CPU", n * dtype.itemsize, initial_value=bytes(n * dtype.itemsize)), dtype)
  @staticmethod
  def _run(fxn, *bufs:UOp, **var_vals:int) -> list: # first buffer is the output
    run_linear(hcq2.hcq_link(lower_and_compile(lower_hcq(fxn(*bufs))), allow_cache=False), var_vals, jit=True)
    return bufs[0].buffer.host.view(fmt=bufs[0].dtype.fmt)[:]

  def test_no_addrs_no_placeholders(self):
    @uopfunc
    def inc(o:UOp, a:UOp): return o.index(0).store(a.index(0).load() + 1).sink()
    a = UOp.from_buffer(Buffer("CPU", 8, initial_value=struct.pack("Q", 41)), dtypes.uint64)
    self.assertEqual(self._run(inc, self._buf(1), a), [42])

  def test_addr_of_arg(self):
    b = self._buf(16, dtypes.uint8)
    self.assertEqual(self._run(addr_of, self._buf(1), b), [b.buffer.get_buf("CPU")])

  def test_addr_of_view(self):
    @uopfunc
    def addr(o:UOp, b:UOp): return o.index(0).store(b[4:8].getaddr("CPU")).sink()
    b = self._buf(16, dtypes.uint8)
    self.assertEqual(self._run(addr, self._buf(1), b), [b.buffer.get_buf("CPU") + 4])

  def test_sink_calls_function(self): # a bare sink with args calls a function that writes one arg and takes the address of the other
    def sink(o:UOp, b:UOp): return addr_of(UOp.param(0, dtypes.uint64, 1), UOp.param(1, dtypes.uint8, 16)).sink().call(o, b, name="sink")
    outs, b = [self._buf(1) for _ in range(2)], self._buf(16, dtypes.uint8)
    for out in outs: self.assertEqual(self._run(sink, out, b), [b.buffer.get_buf("CPU")]) # the second run reuses the program

  def test_nested_functions(self): # the address is taken two calls deep
    @uopfunc
    def mid(o:UOp, b:UOp): return addr_of(o, b).sink()
    @uopfunc
    def top(o:UOp, b:UOp): return mid(o, b).sink()
    b = self._buf(16, dtypes.uint8)
    self.assertEqual(self._run(top, self._buf(1), b), [b.buffer.get_buf("CPU")])

  def test_call_sites(self): # one function, called on different args
    @uopfunc
    def both(o:UOp, a:UOp, b:UOp): return UOp.sink(addr_of(o[0:1], a), addr_of(o[1:2], b))
    a, b = self._buf(16, dtypes.uint8), self._buf(16, dtypes.uint8)
    self.assertEqual(self._run(both, self._buf(2), a, b), [a.buffer.get_buf("CPU"), b.buffer.get_buf("CPU")])

  def test_nested_placeholders(self): # storage a function keeps for itself
    @uopfunc
    def keep(o:UOp):
      tmps = [cpu_buf(dtype=dtypes.uint64, tag="tmp") for _ in range(2)]
      return UOp.sink(*[o.index(i).store(t.after(t.index(0).store(7 + i)).index(0).load()) for i, t in enumerate(tmps)])
    @uopfunc
    def top(o:UOp): return keep(o).sink()
    self.assertEqual(self._run(top, self._buf(2)), [7, 8])

  def test_addr_of_placeholder(self): # the address is the one of the buffer the placeholder links to
    @uopfunc
    def keep(o:UOp):
      tmp = cpu_buf(dtype=dtypes.uint64, tag="tmp")
      return UOp.sink(tmp.index(0).store(7), o.index(0).store(tmp.getaddr("CPU")))
    self.assertEqual(ctypes.c_uint64.from_address(self._run(keep, self._buf(1))[0]).value, 7)

  def test_cstruct(self): # a patched placeholder inside a function
    @uopfunc
    def read(o:UOp):
      s = hcq2.cstruct(init_c_struct_t(8, (("pad", ctypes.c_uint32, 0), ("value", ctypes.c_uint32, 4))), value=42)
      return o.index(0).store(s.bitcast(dtypes.uint32).index(1).load().cast(dtypes.uint64)).sink()
    self.assertEqual(self._run(read, self._buf(1)), [42])

  def test_variable_in_function(self): # a variable is passed to a function, the program binds it by name
    @uopfunc
    def scale(o:UOp, a:UOp, k:UOp): return o.index(0).store(a.index(0).load() * k).sink()
    @uopfunc
    def top(o:UOp, a:UOp): return scale(o, a, UOp.variable("k", 0, 10, dtypes.uint64)).sink()
    a = UOp.from_buffer(Buffer("CPU", 8, initial_value=struct.pack("Q", 7)), dtypes.uint64)
    self.assertEqual(self._run(top, self._buf(1), a, k=6), [42])

  def test_weak_variable_in_function(self): # a weak variable reached in a function
    @uopfunc
    def put(o:UOp): return o.index(0).store(UOp.variable("n", 1, 10).cast(dtypes.uint64)).sink()
    self.assertEqual(self._run(put, self._buf(1), n=7), [7])

  def test_weak_variable_as_arg(self): # a weak arg commits its dtype, like lift does
    @uopfunc
    def put(o:UOp, v:UOp): return o.index(0).store(v.cast(dtypes.uint64)).sink()
    @uopfunc
    def top(o:UOp): return put(o, UOp.variable("n", 1, 10)).sink()
    self.assertEqual(self._run(top, self._buf(1), n=7), [7])

  def test_addr_of_arg_after_a_write(self): # the arg contains the param it replaces
    @uopfunc
    def top(o:UOp, b:UOp): return addr_of(o, b.after(b.index(0).store(2))).sink()
    b = self._buf(1)
    self.assertEqual(self._run(top, self._buf(1), b), [b.buffer.get_buf("CPU")])

  def test_inputs_out_of_slot_order(self): # inputs reached out of slot order
    p = [UOp.param(i, dtypes.uint64, 1, "CPU") for i in range(2)]
    self.assertEqual(lower_hcq(p[1].index(0).load(), p[0].index(0).load()).src[0].without_after.src[1:], (p[1], p[0]))

  def test_one_function_for_any_placeholder(self): # a function names its params: what it is called on does not make another function
    @uopfunc
    def put(out:UOp, v:UOp): return out.index(0).store(v).sink()
    a, b = [cpu_buf(dtype=dtypes.uint64, tag=t) for t in ("cb", "enc")]
    lowered = lower_hcq(put(a, UOp.const(1, dtypes.uint64)), put(b, UOp.const(2, dtypes.uint64)))
    self.assertEqual(len({c.body for c in lowered.toposort(enter_calls=True) if c.op is Ops.CALL and c.arg.name == "put"}), 1)

  def test_one_function_with_registers(self): # a body numbers its own registers and loops: two traces are one function
    @uopfunc
    def put(out:UOp, v:UOp):
      r = UOp.placeholder((1,), dtypes.uint64, addrspace=AddrSpace.REG)
      return out.index(0).store(r.after(r.index(0).store(v)).index(0).load()).sink()
    calls = [put(cpu_buf(dtype=dtypes.uint64), UOp.const(1, dtypes.uint64)) for _ in range(2)]
    self.assertIs(calls[0].body, calls[1].body)

if __name__ == "__main__":
  unittest.main()
