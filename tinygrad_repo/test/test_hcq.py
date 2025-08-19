import unittest, ctypes, struct, os, random, numpy as np
from tinygrad import Device, Tensor, dtypes
from tinygrad.helpers import getenv, CI, mv_address, DEBUG
from tinygrad.device import Buffer, BufferSpec
from tinygrad.runtime.support.hcq import HCQCompiled, HCQBuffer
from tinygrad.runtime.autogen import libc
from tinygrad.runtime.support.system import PCIIfaceBase
from tinygrad.engine.realize import get_runner, CompiledRunner, get_program
from tinygrad.codegen.opt.kernel import Opt, OptOps
from tinygrad import Variable

MOCKGPU = getenv("MOCKGPU")

@unittest.skipUnless(issubclass(type(Device[Device.DEFAULT]), HCQCompiled), "HCQ device required to run")
class TestHCQ(unittest.TestCase):
  @classmethod
  def setUpClass(self):
    TestHCQ.d0 = Device[Device.DEFAULT]
    TestHCQ.a = Tensor([0.,1.], device=Device.DEFAULT).realize()
    TestHCQ.b = self.a + 1
    si = self.b.schedule()[-1]

    TestHCQ.runner = get_runner(TestHCQ.d0.device, si.ast)
    TestHCQ.b.uop.buffer.allocate()

    TestHCQ.kernargs_ba_ptr = TestHCQ.runner._prg.fill_kernargs([TestHCQ.b.uop.buffer._buf, TestHCQ.a.uop.buffer._buf])
    TestHCQ.kernargs_ab_ptr = TestHCQ.runner._prg.fill_kernargs([TestHCQ.a.uop.buffer._buf, TestHCQ.b.uop.buffer._buf])

  def setUp(self):
    TestHCQ.d0.synchronize()
    TestHCQ.a.uop.buffer.copyin(memoryview(bytearray(struct.pack("ff", 0, 1))))
    TestHCQ.b.uop.buffer.copyin(memoryview(bytearray(struct.pack("ff", 0, 0))))
    TestHCQ.d0.synchronize() # wait for copyins to complete

  # Test signals
  def test_signal(self):
    for queue_type in [TestHCQ.d0.hw_compute_queue_t, TestHCQ.d0.hw_copy_queue_t]:
      if queue_type is None: continue

      with self.subTest(name=str(queue_type)):
        queue_type().signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
        TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
        TestHCQ.d0.timeline_value += 1

  def test_signal_update(self):
    for queue_type in [TestHCQ.d0.hw_compute_queue_t, TestHCQ.d0.hw_copy_queue_t]:
      if queue_type is None: continue

      virt_val = Variable("sig_val", 0, 0xffffffff, dtypes.uint32)
      virt_signal = TestHCQ.d0.signal_t(base_buf=HCQBuffer(Variable("sig_addr", 0, 0xffffffffffffffff, dtypes.uint64), 16))

      with self.subTest(name=str(queue_type)):
        q = queue_type().signal(virt_signal, virt_val)

        var_vals = {virt_signal.base_buf.va_addr: TestHCQ.d0.timeline_signal.base_buf.va_addr, virt_val: TestHCQ.d0.timeline_value}
        q.submit(TestHCQ.d0, var_vals)
        TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
        TestHCQ.d0.timeline_value += 1

        var_vals = {virt_signal.base_buf.va_addr: TestHCQ.d0.timeline_signal.base_buf.va_addr, virt_val: TestHCQ.d0.timeline_value}
        q.submit(TestHCQ.d0, var_vals)
        TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
        TestHCQ.d0.timeline_value += 1

  # Test wait
  def test_wait(self):
    for queue_type in [TestHCQ.d0.hw_compute_queue_t, TestHCQ.d0.hw_copy_queue_t]:
      if queue_type is None: continue

      with self.subTest(name=str(queue_type)):
        fake_signal = TestHCQ.d0.new_signal()
        fake_signal.value = 1
        queue_type().wait(fake_signal, 1) \
                    .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
        TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
        TestHCQ.d0.timeline_value += 1

  @unittest.skipIf(MOCKGPU or Device.DEFAULT in {"CPU", "LLVM"}, "Can't handle async update on MOCKGPU for now")
  def test_wait_late_set(self):
    for queue_type in [TestHCQ.d0.hw_compute_queue_t, TestHCQ.d0.hw_copy_queue_t]:
      if queue_type is None: continue

      with self.subTest(name=str(queue_type)):
        fake_signal = TestHCQ.d0.new_signal()
        queue_type().wait(fake_signal, 1) \
                    .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)

        with self.assertRaises(RuntimeError):
          TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value, timeout=500)

        fake_signal.value = 1
        TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)

        TestHCQ.d0.timeline_value += 1

  def test_wait_update(self):
    for queue_type in [TestHCQ.d0.hw_compute_queue_t, TestHCQ.d0.hw_copy_queue_t]:
      if queue_type is None: continue

      with self.subTest(name=str(queue_type)):
        virt_val = Variable("sig_val", 0, 0xffffffff, dtypes.uint32)
        virt_signal = TestHCQ.d0.signal_t(base_buf=HCQBuffer(Variable("sig_addr", 0, 0xffffffffffffffff, dtypes.uint64), 16))

        fake_signal = TestHCQ.d0.new_signal()
        q = queue_type().wait(virt_signal, virt_val).signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)

        fake_signal.value = 0x30

        q.submit(TestHCQ.d0, {virt_signal.base_buf.va_addr: fake_signal.base_buf.va_addr, virt_val: fake_signal.value})
        TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
        TestHCQ.d0.timeline_value += 1

  # Test exec
  def test_exec_one_kernel(self):
    TestHCQ.d0.hw_compute_queue_t().exec(TestHCQ.runner._prg, TestHCQ.kernargs_ba_ptr, TestHCQ.runner.p.global_size, TestHCQ.runner.p.local_size) \
                                   .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)

    TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1

    val = TestHCQ.b.uop.buffer.as_buffer().cast("f")[0]
    assert val == 1.0, f"got val {val}"

  def test_exec_2_kernels_100_times(self):
    virt_val = Variable("sig_val", 0, 0xffffffff, dtypes.uint32)

    q = TestHCQ.d0.hw_compute_queue_t()
    q.wait(TestHCQ.d0.timeline_signal, virt_val - 1) \
     .exec(TestHCQ.runner._prg, TestHCQ.kernargs_ba_ptr, TestHCQ.runner.p.global_size, TestHCQ.runner.p.local_size) \
     .exec(TestHCQ.runner._prg, TestHCQ.kernargs_ab_ptr, TestHCQ.runner.p.global_size, TestHCQ.runner.p.local_size) \
     .signal(TestHCQ.d0.timeline_signal, virt_val)

    for _ in range(100):
      q.submit(TestHCQ.d0, {virt_val: TestHCQ.d0.timeline_value})
      TestHCQ.d0.timeline_value += 1

    val = TestHCQ.a.uop.buffer.as_buffer().cast("f")[0]
    assert val == 200.0, f"got val {val}"

  @unittest.skipIf(Device.DEFAULT in {"CPU", "LLVM"}, "No globals/locals on LLVM/CPU")
  def test_exec_update(self):
    sint_global = (Variable("sint_global", 0, 0xffffffff, dtypes.uint32),) + tuple(TestHCQ.runner.p.global_size[1:])
    sint_local = (Variable("sint_local", 0, 0xffffffff, dtypes.uint32),) + tuple(TestHCQ.runner.p.local_size[1:])

    q = TestHCQ.d0.hw_compute_queue_t()
    q.exec(TestHCQ.runner._prg, TestHCQ.kernargs_ba_ptr, sint_global, sint_local) \
     .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)

    q.submit(TestHCQ.d0, {sint_global[0]: 1, sint_local[0]: 1})
    TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1

    val = TestHCQ.b.uop.buffer.as_buffer().cast("f")[0]
    assert val == 1.0, f"got val {val}"
    val = TestHCQ.b.uop.buffer.as_buffer().cast("f")[1]
    assert val == 0.0, f"got val {val}, should not be updated"

  @unittest.skipIf(Device.DEFAULT in {"CPU", "LLVM"}, "No globals/locals on LLVM/CPU")
  def test_exec_update_fuzz(self):
    virt_val = Variable("sig_val", 0, 0xffffffff, dtypes.uint32)
    virt_local = [Variable(f"local_{i}", 0, 0xffffffff, dtypes.uint32) for i in range(3)]

    a = Tensor.randint((3, 3, 3), dtype=dtypes.int, device=Device.DEFAULT).realize()
    b = a + 1
    si = b.schedule()[-1]

    runner = CompiledRunner(get_program(si.ast, TestHCQ.d0.renderer, opts=[Opt(op=OptOps.LOCAL, axis=0, arg=3) for _ in range(3)]))

    zb = Buffer(Device.DEFAULT, 3 * 3 * 3, dtypes.int, options=BufferSpec(cpu_access=True, nolru=True)).ensure_allocated()
    zt = Buffer(Device.DEFAULT, 3 * 3 * 3, dtypes.int, options=BufferSpec(cpu_access=True, nolru=True)).ensure_allocated()
    ctypes.memset(zb._buf.va_addr, 0, zb.nbytes)
    kernargs = runner._prg.fill_kernargs([zt._buf, zb._buf])

    q = TestHCQ.d0.hw_compute_queue_t()
    q.memory_barrier() \
     .exec(runner._prg, kernargs, (1,1,1), virt_local) \
     .signal(TestHCQ.d0.timeline_signal, virt_val)

    for x in range(1, 4):
      for y in range(1, 4):
        for z in range(1, 4):
          ctypes.memset(zt._buf.va_addr, 0, zb.nbytes)

          q.submit(TestHCQ.d0, {virt_val: TestHCQ.d0.timeline_value, virt_local[0]: x, virt_local[1]: y, virt_local[2]: z})
          TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
          TestHCQ.d0.timeline_value += 1

          res_sum = sum(x for x in zt.as_buffer().cast("I"))
          assert x * y * z == res_sum, f"want {x * y * z}, got {res_sum}"

  # Test copy
  def test_copy(self):
    if TestHCQ.d0.hw_copy_queue_t is None: self.skipTest("device does not support copy queue")

    TestHCQ.d0.hw_copy_queue_t().wait(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1) \
                                .copy(TestHCQ.b.uop.buffer._buf.va_addr, TestHCQ.a.uop.buffer._buf.va_addr, 8) \
                                .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)

    TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1

    val = TestHCQ.b.uop.buffer.as_buffer().cast("f")[1]
    assert val == 1.0, f"got val {val}"

  def test_copy_long(self):
    if TestHCQ.d0.hw_copy_queue_t is None: self.skipTest("device does not support copy queue")

    sz = 64 << 20
    buf1 = Buffer(Device.DEFAULT, sz, dtypes.int8, options=BufferSpec(nolru=True)).ensure_allocated()
    buf2 = Buffer(Device.DEFAULT, sz, dtypes.int8, options=BufferSpec(host=True, nolru=True)).ensure_allocated()
    ctypes.memset(buf2._buf.va_addr, 1, sz)

    TestHCQ.d0.hw_copy_queue_t().wait(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1) \
                                .copy(buf1._buf.va_addr, buf2._buf.va_addr, sz) \
                                .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)

    TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1

    mv_buf1 = buf1.as_buffer().cast('Q')
    assert libc.memcmp(mv_address(mv_buf1), buf2._buf.va_addr, sz) == 0

  @unittest.skipIf(CI, "skip in CI")
  def test_copy_64bit(self):
    if TestHCQ.d0.hw_copy_queue_t is None: self.skipTest("device does not support copy queue")

    for sz in [(1 << 32) - 1, (1 << 32), (1 << 32) + 1, (5 << 30), (6 << 30) - 0x4642ee1]:
      buf1 = Buffer(Device.DEFAULT, sz, dtypes.int8, options=BufferSpec(nolru=True)).ensure_allocated()
      buf2 = Buffer(Device.DEFAULT, sz, dtypes.int8, options=BufferSpec(host=True, nolru=True)).ensure_allocated()

      ctypes.memset(buf2._buf.va_addr, 0x3e, sz)
      buf2_q_view = buf2._buf.cpu_view().view(fmt='Q')
      for i in range(0, sz//8, 0x1000):
        for j in range(32): buf2_q_view[min(max(i + j - 16, 0), (sz // 8) - 1)] = random.randint(0, 0xffffffffffffffff)

      TestHCQ.d0.hw_copy_queue_t().wait(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1) \
                                  .copy(buf1._buf.va_addr, buf2._buf.va_addr, sz) \
                                  .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)

      TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
      TestHCQ.d0.timeline_value += 1

      mv_buf1 = buf1.as_buffer()
      assert libc.memcmp(mv_address(mv_buf1), buf2._buf.va_addr, sz) == 0

  def test_update_copy(self):
    if TestHCQ.d0.hw_copy_queue_t is None: self.skipTest("device does not support copy queue")

    virt_src_addr = Variable("virt_src_addr", 0, 0xffffffffffffffff, dtypes.uint64)
    virt_dest_addr = Variable("virt_dest_addr", 0, 0xffffffffffffffff, dtypes.uint64)

    q = TestHCQ.d0.hw_copy_queue_t().wait(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1) \
                                    .copy(virt_dest_addr, virt_src_addr, 8) \
                                    .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)

    q.submit(TestHCQ.d0, {virt_src_addr: TestHCQ.a.uop.buffer._buf.va_addr, virt_dest_addr: TestHCQ.b.uop.buffer._buf.va_addr})

    TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1

    val = TestHCQ.b.uop.buffer.as_buffer().cast("f")[1]
    assert val == 1.0, f"got val {val}"

  def test_update_copy_long(self):
    if TestHCQ.d0.hw_copy_queue_t is None: self.skipTest("device does not support copy queue")

    virt_src_addr = Variable("virt_src_addr", 0, 0xffffffffffffffff, dtypes.uint64)
    virt_dest_addr = Variable("virt_dest_addr", 0, 0xffffffffffffffff, dtypes.uint64)

    sz = 64 << 20
    buf1 = Buffer(Device.DEFAULT, sz, dtypes.int8, options=BufferSpec(nolru=True)).ensure_allocated()
    buf2 = Buffer(Device.DEFAULT, sz, dtypes.int8, options=BufferSpec(host=True, nolru=True)).ensure_allocated()
    ctypes.memset(buf2._buf.va_addr, 1, sz)

    q = TestHCQ.d0.hw_copy_queue_t().wait(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1) \
                                    .copy(virt_dest_addr, virt_src_addr, sz) \
                                    .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)

    q.submit(TestHCQ.d0, {virt_src_addr: buf2._buf.va_addr, virt_dest_addr: buf1._buf.va_addr})

    TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1

    mv_buf1 = buf1.as_buffer().cast('Q')
    for i in range(sz//8): assert mv_buf1[i] == 0x0101010101010101, f"offset {i*8} differs, not all copied, got {hex(mv_buf1[i])}"

  # Test bind api
  def test_bind(self):
    for queue_type in [TestHCQ.d0.hw_compute_queue_t, TestHCQ.d0.hw_copy_queue_t]:
      if queue_type is None: continue

      virt_val = Variable("sig_val", 0, 0xffffffff, dtypes.uint32)
      virt_signal = TestHCQ.d0.signal_t(base_buf=HCQBuffer(Variable("sig_addr", 0, 0xffffffffffffffff, dtypes.uint64), 16))

      with self.subTest(name=str(queue_type)):
        fake_signal = TestHCQ.d0.new_signal()
        q = queue_type().wait(virt_signal, virt_val).signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
        q.bind(TestHCQ.d0)

        fake_signal.value = 0x30

        q.submit(TestHCQ.d0, {virt_signal.base_buf.va_addr: fake_signal.base_buf.va_addr, virt_val: fake_signal.value})
        TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
        TestHCQ.d0.timeline_value += 1

  # Test multidevice
  def test_multidevice_signal_wait(self):
    if TestHCQ.d0.hw_copy_queue_t is None: self.skipTest("device does not support copy queue")

    try: d1 = Device[f"{Device.DEFAULT}:1"]
    except Exception: self.skipTest("no multidevice, test skipped")

    TestHCQ.d0.hw_copy_queue_t().signal(sig:=TestHCQ.d0.new_signal(value=0), value=0xfff) \
                                .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)

    d1.hw_copy_queue_t().wait(sig, value=0xfff) \
                        .signal(d1.timeline_signal, d1.timeline_value).submit(d1)

    TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1

    d1.timeline_signal.wait(d1.timeline_value)
    d1.timeline_value += 1

  # Test profile api
  def test_speed_exec_time(self):
    sig_st, sig_en = TestHCQ.d0.new_signal(), TestHCQ.d0.new_signal()
    TestHCQ.d0.hw_compute_queue_t().timestamp(sig_st) \
                                   .exec(TestHCQ.runner._prg, TestHCQ.kernargs_ba_ptr, TestHCQ.runner.p.global_size, TestHCQ.runner.p.local_size) \
                                   .timestamp(sig_en) \
                                   .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)

    TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1

    et = float(sig_en.timestamp - sig_st.timestamp)

    print(f"exec kernel time: {et:.2f} us")
    assert 0.1 <= et <= (100000 if MOCKGPU or Device.DEFAULT in {"CPU", "LLVM"} else 100)

  def test_speed_copy_bandwidth(self):
    if TestHCQ.d0.hw_copy_queue_t is None: self.skipTest("device does not support copy queue")

    # THEORY: the bandwidth is low here because it's only using one SDMA queue. I suspect it's more stable like this at least.
    SZ = 200_000_000
    a = Buffer(Device.DEFAULT, SZ, dtypes.uint8, options=BufferSpec(nolru=True)).allocate()
    b = Buffer(Device.DEFAULT, SZ, dtypes.uint8, options=BufferSpec(nolru=True)).allocate()

    sig_st, sig_en = TestHCQ.d0.new_signal(), TestHCQ.d0.new_signal()
    TestHCQ.d0.hw_copy_queue_t().timestamp(sig_st) \
                                .copy(a._buf.va_addr, b._buf.va_addr, SZ) \
                                .timestamp(sig_en) \
                                .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)

    TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1

    et = float(sig_en.timestamp - sig_st.timestamp)
    et_ms = et / 1e3

    gb_s = ((SZ / 1e9) / et_ms) * 1e3
    print(f"same device copy:  {et_ms:.2f} ms, {gb_s:.2f} GB/s")
    assert (0.2 if MOCKGPU else 10) <= gb_s <= 1000

  def test_speed_cross_device_copy_bandwidth(self):
    if TestHCQ.d0.hw_copy_queue_t is None: self.skipTest("device does not support copy queue")

    try: _ = Device[f"{Device.DEFAULT}:1"]
    except Exception: self.skipTest("no multidevice, test skipped")

    SZ = 200_000_000
    b = Buffer(f"{Device.DEFAULT}:1", SZ, dtypes.uint8, options=BufferSpec(nolru=True)).allocate()
    a = Buffer(Device.DEFAULT, SZ, dtypes.uint8, options=BufferSpec(nolru=True)).allocate()
    TestHCQ.d0.allocator.map(b._buf)

    sig_st, sig_en = TestHCQ.d0.new_signal(), TestHCQ.d0.new_signal()
    TestHCQ.d0.hw_copy_queue_t().timestamp(sig_st) \
                                .copy(a._buf.va_addr, b._buf.va_addr, SZ) \
                                .timestamp(sig_en) \
                                .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)

    TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1

    et = float(sig_en.timestamp - sig_st.timestamp)
    et_ms = et / 1e3

    gb_s = ((SZ / 1e9) / et_ms) * 1e3
    print(f"cross device copy: {et_ms:.2f} ms, {gb_s:.2f} GB/s")
    assert (0.2 if MOCKGPU else 2) <= gb_s <= 100

  def test_timeline_signal_rollover(self):
    for queue_type in [TestHCQ.d0.hw_compute_queue_t, TestHCQ.d0.hw_copy_queue_t]:
      if queue_type is None: continue

      with self.subTest(name=str(queue_type)):
        TestHCQ.d0.timeline_value = (1 << 32) - 20 # close value to reset
        queue_type().signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1).submit(TestHCQ.d0)
        TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value - 1)

        for _ in range(40):
          queue_type().wait(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1) \
                      .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
          TestHCQ.d0.timeline_value += 1
          TestHCQ.d0.synchronize()

  def test_small_copies_from_host_buf(self):
    if TestHCQ.d0.hw_copy_queue_t is None: self.skipTest("device does not support copy queue")

    buf1 = Buffer(Device.DEFAULT, 1, dtypes.int8, options=BufferSpec(nolru=True)).ensure_allocated()
    buf2 = Buffer(Device.DEFAULT, 1, dtypes.int8, options=BufferSpec(host=True, nolru=True)).ensure_allocated()

    for i in range(256):
      ctypes.memset(buf2._buf.va_addr, i, 1)

      TestHCQ.d0.hw_copy_queue_t().wait(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1) \
                                  .copy(buf1._buf.va_addr, buf2._buf.va_addr, 1) \
                                  .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
      TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
      TestHCQ.d0.timeline_value += 1

      assert buf1.as_buffer()[0] == i

  def test_small_copies_from_host_buf_intercopy(self):
    if TestHCQ.d0.hw_copy_queue_t is None: self.skipTest("device does not support copy queue")

    buf1 = Buffer(Device.DEFAULT, 1, dtypes.int8, options=BufferSpec(nolru=True)).ensure_allocated()
    buf2 = Buffer(Device.DEFAULT, 1, dtypes.int8, options=BufferSpec(nolru=True)).ensure_allocated()
    buf3 = Buffer(Device.DEFAULT, 1, dtypes.int8, options=BufferSpec(host=True, nolru=True)).ensure_allocated()

    for i in range(256):
      ctypes.memset(buf3._buf.va_addr, i, 1)

      TestHCQ.d0.hw_copy_queue_t().wait(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1) \
                                  .copy(buf1._buf.va_addr, buf3._buf.va_addr, 1) \
                                  .copy(buf2._buf.va_addr, buf1._buf.va_addr, 1) \
                                  .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
      TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
      TestHCQ.d0.timeline_value += 1

      assert buf2.as_buffer()[0] == i

  def test_small_copies_from_host_buf_transfer(self):
    if TestHCQ.d0.hw_copy_queue_t is None: self.skipTest("device does not support copy queue")

    try: _ = Device[f"{Device.DEFAULT}:1"]
    except Exception: self.skipTest("no multidevice, test skipped")

    buf1 = Buffer(Device.DEFAULT, 1, dtypes.int8, options=BufferSpec(nolru=True)).ensure_allocated()
    buf2 = Buffer(f"{Device.DEFAULT}:1", 1, dtypes.int8, options=BufferSpec(nolru=True)).ensure_allocated()
    buf3 = Buffer(Device.DEFAULT, 1, dtypes.int8, options=BufferSpec(host=True, nolru=True)).ensure_allocated()
    TestHCQ.d0.allocator.map(buf2._buf)

    for i in range(256):
      ctypes.memset(buf3._buf.va_addr, i, 1)

      TestHCQ.d0.hw_copy_queue_t().wait(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1) \
                                  .copy(buf1._buf.va_addr, buf3._buf.va_addr, 1) \
                                  .copy(buf2._buf.va_addr, buf1._buf.va_addr, 1) \
                                  .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
      TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
      TestHCQ.d0.timeline_value += 1

      assert buf2.as_buffer()[0] == i

  def test_memory_barrier(self):
    a = Tensor([0, 1], device=Device.DEFAULT, dtype=dtypes.int8).realize()
    b = a + 1
    runner = get_runner(TestHCQ.d0.device, b.schedule()[-1].ast)

    buf1 = Buffer(Device.DEFAULT, 2, dtypes.int8, options=BufferSpec(nolru=True)).ensure_allocated()
    buf2 = Buffer(Device.DEFAULT, 2, dtypes.int8, options=BufferSpec(cpu_access=True, nolru=True)).ensure_allocated()

    kernargs_ptr = runner._prg.fill_kernargs([buf1._buf, buf2._buf])

    for i in range(255):
      ctypes.memset(buf2._buf.va_addr, i, 2)

      # Need memory_barrier after direct write to vram
      TestHCQ.d0.hw_compute_queue_t().wait(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1) \
                                     .memory_barrier() \
                                     .exec(runner._prg, kernargs_ptr, runner.p.global_size, runner.p.local_size) \
                                     .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
      TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
      TestHCQ.d0.timeline_value += 1

      assert buf1.as_buffer()[0] == (i + 1), f"has {buf1.as_buffer()[0]}, need {i + 1}"

  def test_memory_barrier_before_copy(self):
    if TestHCQ.d0.hw_copy_queue_t is None: self.skipTest("device does not support copy queue")

    buf1 = Buffer(Device.DEFAULT, 1, dtypes.int8, options=BufferSpec(nolru=True)).ensure_allocated()
    buf2 = Buffer(Device.DEFAULT, 1, dtypes.int8, options=BufferSpec(nolru=True)).ensure_allocated()
    buf3 = Buffer(Device.DEFAULT, 1, dtypes.int8, options=BufferSpec(cpu_access=True, nolru=True)).ensure_allocated()

    for i in range(256):
      ctypes.memset(buf3._buf.va_addr, i, 1)

      # Need memory_barrier after direct write to vram
      TestHCQ.d0.hw_compute_queue_t().wait(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1) \
                                     .memory_barrier() \
                                     .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
      TestHCQ.d0.timeline_value += 1

      TestHCQ.d0.hw_copy_queue_t().wait(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1) \
                                  .copy(buf1._buf.va_addr, buf3._buf.va_addr, 1) \
                                  .copy(buf2._buf.va_addr, buf1._buf.va_addr, 1) \
                                  .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
      TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
      TestHCQ.d0.timeline_value += 1

      assert buf2.as_buffer()[0] == i

  def test_map_cpu_buffer_to_device(self):
    if Device[Device.DEFAULT].hw_copy_queue_t is None: self.skipTest("skip device without copy queue")

    sz = 0x2000
    cpu_buffer = Buffer("CPU", sz, dtypes.uint8, options=BufferSpec(cpu_access=True)).ensure_allocated()
    cpu_buffer._buf.cpu_view().view(fmt='B')[:] = bytes([x & 0xff for x in range(sz)])

    for devid in range(6):
      if DEBUG >= 2: print(f"Testing map to device {Device.DEFAULT}:{devid}")

      try: d = Device[f"{Device.DEFAULT}:{devid}"]
      except Exception: break

      local_buf = Buffer(f"{Device.DEFAULT}:{devid}", sz, dtypes.uint8, options=BufferSpec(cpu_access=True)).ensure_allocated()

      d.allocator.map(cpu_buffer._buf)

      d.hw_copy_queue_t().wait(d.timeline_signal, d.timeline_value - 1) \
                         .copy(local_buf._buf.va_addr, cpu_buffer._buf.va_addr, sz) \
                         .signal(d.timeline_signal, d.timeline_value).submit(d)
      d.timeline_signal.wait(d.timeline_value)
      d.timeline_value += 1

      np.testing.assert_equal(cpu_buffer.numpy(), local_buf.numpy(), "failed")

  @unittest.skipUnless(MOCKGPU, "Emulate this on MOCKGPU to check the path in CI")
  def test_on_device_hang(self):
    if not hasattr(self.d0, 'on_device_hang'): self.skipTest("device does not have on_device_hang")

    os.environ["MOCKGPU_EMU_FAULTADDR"] = "0xDEADBEE1"

    # Check api calls
    with self.assertRaises(RuntimeError) as ctx:
      self.d0.on_device_hang()

    assert "0xDEADBEE1" in str(ctx.exception)
    os.environ.pop("MOCKGPU_EMU_FAULTADDR")

  def test_multidevice(self):
    try: amd_dev = Device["AMD"]
    except Exception: self.skipTest("no AMD device, test skipped")

    try: nv_dev = Device["NV"]
    except Exception: self.skipTest("no NV device, test skipped")

    x = amd_dev.new_signal()
    y = nv_dev.new_signal()
    assert type(x) is amd_dev.signal_t
    assert type(y) is nv_dev.signal_t

  def test_multidevice_p2p(self):
    try:
      amd_dev = Device["AMD"]
      if not issubclass(type(amd_dev.iface), PCIIfaceBase): self.skipTest("Not a pci dev")
    except Exception: self.skipTest("no AMD device, test skipped")

    try:
      nv_dev = Device["NV"]
      if not issubclass(type(nv_dev.iface), PCIIfaceBase): self.skipTest("Not a pci dev")
    except Exception: self.skipTest("no NV device, test skipped")

    def _check_copy(dev1, dev2):
      buf1 = Tensor.randn(10, 10, device=dev1).realize()
      buf2 = buf1.to(dev2).realize()
      np.testing.assert_equal(buf1.numpy(), buf2.numpy(), "p2p failed")
    _check_copy("AMD", "NV")
    _check_copy("NV", "AMD")

if __name__ == "__main__":
  unittest.main()
