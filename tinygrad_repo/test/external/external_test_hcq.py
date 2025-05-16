import unittest, ctypes, struct, time, array
from tinygrad import Device, Tensor, dtypes
from tinygrad.helpers import to_mv, CI
from tinygrad.device import Buffer, BufferSpec
from tinygrad.engine.realize import get_runner

def _time_queue(q, d):
  st = time.perf_counter()
  q.signal(d.timeline_signal, d.timeline_value)
  q.submit(d)
  d._wait_signal(d.timeline_signal, d.timeline_value)
  d.timeline_value += 1
  return time.perf_counter() - st

@unittest.skipUnless(Device.DEFAULT in ["NV", "AMD"], "Runs only on NV or AMD")
class TestHCQ(unittest.TestCase):
  @classmethod
  def setUpClass(self):
    TestHCQ.d0 = Device[Device.DEFAULT]
    #TestHCQ.d1: AMDDevice = Device["AMD:1"]
    TestHCQ.a = Tensor([0.,1.], device=Device.DEFAULT).realize()
    TestHCQ.b = self.a + 1
    si = self.b.schedule()[-1]
    TestHCQ.runner = get_runner(TestHCQ.d0.device, si.ast)
    TestHCQ.b.lazydata.buffer.allocate()
    # wow that's a lot of abstraction layers
    TestHCQ.addr = struct.pack("QQ", TestHCQ.b.lazydata.buffer._buf.va_addr, TestHCQ.a.lazydata.buffer._buf.va_addr)
    TestHCQ.addr2 = struct.pack("QQ", TestHCQ.a.lazydata.buffer._buf.va_addr, TestHCQ.b.lazydata.buffer._buf.va_addr)
    TestHCQ.kernargs_off = TestHCQ.runner._prg.kernargs_offset
    TestHCQ.kernargs_size = TestHCQ.runner._prg.kernargs_alloc_size
    ctypes.memmove(TestHCQ.d0.kernargs_ptr+TestHCQ.kernargs_off, TestHCQ.addr, len(TestHCQ.addr))
    ctypes.memmove(TestHCQ.d0.kernargs_ptr+TestHCQ.kernargs_size+TestHCQ.kernargs_off, TestHCQ.addr2, len(TestHCQ.addr2))

    if Device.DEFAULT == "AMD":
      from tinygrad.runtime.ops_amd import HWQueue, HWPM4Queue
      TestHCQ.compute_queue = HWPM4Queue
      TestHCQ.copy_queue = HWQueue
    elif Device.DEFAULT == "NV":
      from tinygrad.runtime.ops_nv import HWQueue, HWQueue
      # nv need to copy constbuffer there as well
      to_mv(TestHCQ.d0.kernargs_ptr, 0x160).cast('I')[:] = array.array('I', TestHCQ.runner._prg.constbuffer_0)
      to_mv(TestHCQ.d0.kernargs_ptr+TestHCQ.kernargs_size, 0x160).cast('I')[:] = array.array('I', TestHCQ.runner._prg.constbuffer_0)
      TestHCQ.compute_queue = HWQueue
      TestHCQ.copy_queue = HWQueue

  def setUp(self):
    TestHCQ.d0.synchronize()
    TestHCQ.a.lazydata.buffer.copyin(memoryview(bytearray(struct.pack("ff", 0, 1))))
    TestHCQ.b.lazydata.buffer.copyin(memoryview(bytearray(struct.pack("ff", 0, 0))))
    TestHCQ.d0.synchronize() # wait for copyins to complete

  def test_run_1000_times_one_submit(self):
    temp_signal, temp_value = TestHCQ.d0._alloc_signal(value=0), 0
    q = TestHCQ.compute_queue()
    for _ in range(1000):
      q.exec(TestHCQ.runner._prg, TestHCQ.d0.kernargs_ptr, TestHCQ.runner.p.global_size, TestHCQ.runner.p.local_size)
      q.signal(temp_signal, temp_value + 1).wait(temp_signal, temp_value + 1)
      temp_value += 1

      q.exec(TestHCQ.runner._prg, TestHCQ.d0.kernargs_ptr+TestHCQ.kernargs_size, TestHCQ.runner.p.global_size, TestHCQ.runner.p.local_size)
      q.signal(temp_signal, temp_value + 1).wait(temp_signal, temp_value + 1)
      temp_value += 1

    q.signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
    q.submit(TestHCQ.d0)
    TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1
    val = TestHCQ.a.lazydata.buffer.as_buffer().cast("f")[0]
    assert val == 2000.0, f"got val {val}"

  def test_run_1000_times(self):
    temp_signal = TestHCQ.d0._alloc_signal(value=0)
    q = TestHCQ.compute_queue()
    q.exec(TestHCQ.runner._prg, TestHCQ.d0.kernargs_ptr, TestHCQ.runner.p.global_size, TestHCQ.runner.p.local_size)
    q.signal(temp_signal, 2).wait(temp_signal, 2)
    q.exec(TestHCQ.runner._prg, TestHCQ.d0.kernargs_ptr+TestHCQ.kernargs_size, TestHCQ.runner.p.global_size,
           TestHCQ.runner.p.local_size)
    for _ in range(1000):
      TestHCQ.d0._set_signal(temp_signal, 1)
      q.submit(TestHCQ.d0)
      TestHCQ.compute_queue().signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
      TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
      TestHCQ.d0.timeline_value += 1
    val = TestHCQ.a.lazydata.buffer.as_buffer().cast("f")[0]
    assert val == 2000.0, f"got val {val}"

  def test_run_to_3(self):
    temp_signal = TestHCQ.d0._alloc_signal(value=0)
    q = TestHCQ.compute_queue()
    q.exec(TestHCQ.runner._prg, TestHCQ.d0.kernargs_ptr, TestHCQ.runner.p.global_size, TestHCQ.runner.p.local_size)
    q.signal(temp_signal, 1).wait(temp_signal, 1)
    q.exec(TestHCQ.runner._prg, TestHCQ.d0.kernargs_ptr+TestHCQ.kernargs_size, TestHCQ.runner.p.global_size, TestHCQ.runner.p.local_size)
    q.signal(temp_signal, 2).wait(temp_signal, 2)
    q.exec(TestHCQ.runner._prg, TestHCQ.d0.kernargs_ptr, TestHCQ.runner.p.global_size, TestHCQ.runner.p.local_size)
    q.signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
    TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1
    val = TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[0]
    assert val == 3.0, f"got val {val}"

  def test_update_exec(self):
    q = TestHCQ.compute_queue()
    exec_cmd_idx = len(q)
    q.exec(TestHCQ.runner._prg, TestHCQ.d0.kernargs_ptr, TestHCQ.runner.p.global_size, TestHCQ.runner.p.local_size)
    q.update_exec(exec_cmd_idx, (1,1,1), (1,1,1))
    q.signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
    TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1
    val = TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[0]
    assert val == 1.0, f"got val {val}"
    val = TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[1]
    assert val == 0.0, f"got val {val}, should not be updated"

  @unittest.skipUnless(Device.DEFAULT == "NV", "Only NV supports bind")
  def test_bind_run(self):
    temp_signal = TestHCQ.d0._alloc_signal(value=0)
    q = TestHCQ.compute_queue()
    q.exec(TestHCQ.runner._prg, TestHCQ.d0.kernargs_ptr, TestHCQ.runner.p.global_size, TestHCQ.runner.p.local_size)
    q.signal(temp_signal, 2).wait(temp_signal, 2)
    q.exec(TestHCQ.runner._prg, TestHCQ.d0.kernargs_ptr+TestHCQ.kernargs_size, TestHCQ.runner.p.global_size,
           TestHCQ.runner.p.local_size)
    q.bind(TestHCQ.d0)
    for _ in range(1000):
      TestHCQ.d0._set_signal(temp_signal, 1)
      q.submit(TestHCQ.d0)
      TestHCQ.compute_queue().signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
      TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
      TestHCQ.d0.timeline_value += 1
    val = TestHCQ.a.lazydata.buffer.as_buffer().cast("f")[0]
    assert val == 2000.0, f"got val {val}"

  @unittest.skipUnless(Device.DEFAULT == "NV", "Only NV supports bind")
  def test_update_exec_binded(self):
    q = TestHCQ.compute_queue()
    exec_ptr = q.ptr()
    q.exec(TestHCQ.runner._prg, TestHCQ.d0.kernargs_ptr, TestHCQ.runner.p.global_size, TestHCQ.runner.p.local_size)
    q.signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
    q.bind(TestHCQ.d0)

    q.update_exec(exec_ptr, (1,1,1), (1,1,1))
    q.submit(TestHCQ.d0)
    TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1
    val = TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[0]
    assert val == 1.0, f"got val {val}"
    val = TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[1]
    assert val == 0.0, f"got val {val}, should not be updated"

  @unittest.skipIf(CI, "Can't handle async update on CPU")
  def test_wait_signal(self):
    temp_signal = TestHCQ.d0._alloc_signal(value=0)
    TestHCQ.compute_queue().wait(temp_signal, value=1).signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
    with self.assertRaises(RuntimeError):
      TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value, timeout=50)
    # clean up
    TestHCQ.d0._set_signal(temp_signal, 1)
    TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value, timeout=100)
    TestHCQ.d0.timeline_value += 1

  @unittest.skipIf(CI, "Can't handle async update on CPU")
  def test_wait_copy_signal(self):
    temp_signal = TestHCQ.d0._alloc_signal(value=0)
    TestHCQ.copy_queue().wait(temp_signal, value=1).signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
    with self.assertRaises(RuntimeError):
      TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value, timeout=50)
    # clean up
    TestHCQ.d0._set_signal(temp_signal, 1)
    TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value, timeout=100)
    TestHCQ.d0.timeline_value += 1

  def test_run_normal(self):
    q = TestHCQ.compute_queue()
    q.exec(TestHCQ.runner._prg, TestHCQ.d0.kernargs_ptr, TestHCQ.runner.p.global_size, TestHCQ.runner.p.local_size)
    q.signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
    TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1
    val = TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[0]
    assert val == 1.0, f"got val {val}"

  def test_submit_empty_queues(self):
    TestHCQ.compute_queue().submit(TestHCQ.d0)
    TestHCQ.copy_queue().submit(TestHCQ.d0)

  def test_signal_timeout(self):
    with self.assertRaises(RuntimeError):
      TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value, timeout=50)
      TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value + 122, timeout=50)
    TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1, timeout=50)

  def test_signal(self):
    new_timeline_value = TestHCQ.d0.timeline_value + 0xff
    TestHCQ.compute_queue().signal(TestHCQ.d0.timeline_signal, new_timeline_value).submit(TestHCQ.d0)
    TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, new_timeline_value)
    TestHCQ.d0.timeline_value = new_timeline_value + 1 # update to not break runtime

  def test_copy_signal(self):
    new_timeline_value = TestHCQ.d0.timeline_value + 0xff
    TestHCQ.copy_queue().signal(TestHCQ.d0.timeline_signal, new_timeline_value).submit(TestHCQ.d0)
    TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, new_timeline_value)
    TestHCQ.d0.timeline_value = new_timeline_value + 1 # update to not break runtime

  def test_run_signal(self):
    q = TestHCQ.compute_queue()
    q.exec(TestHCQ.runner._prg, TestHCQ.d0.kernargs_ptr, TestHCQ.runner.p.global_size, TestHCQ.runner.p.local_size)
    q.signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
    q.submit(TestHCQ.d0)
    TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1
    val = TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[0]
    assert val == 1.0, f"got val {val}"

  def test_copy_1000_times(self):
    q = TestHCQ.copy_queue()
    q.copy(TestHCQ.a.lazydata.buffer._buf.va_addr, TestHCQ.b.lazydata.buffer._buf.va_addr, 8)
    q.copy(TestHCQ.b.lazydata.buffer._buf.va_addr, TestHCQ.a.lazydata.buffer._buf.va_addr, 8)
    for _ in range(1000):
      q.submit(TestHCQ.d0)
      TestHCQ.copy_queue().signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
      TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
      TestHCQ.d0.timeline_value += 1
    # confirm the signal didn't exceed the put value
    with self.assertRaises(RuntimeError):
      TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value + 1, timeout=50)
    val = TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[1]
    assert val == 0.0, f"got val {val}"

  def test_copy(self):
    q = TestHCQ.copy_queue()
    q.copy(TestHCQ.b.lazydata.buffer._buf.va_addr, TestHCQ.a.lazydata.buffer._buf.va_addr, 8)
    q.signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
    q.submit(TestHCQ.d0)
    TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1
    val = TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[1]
    assert val == 1.0, f"got val {val}"

  @unittest.skipUnless(Device.DEFAULT == "NV", "Only NV supports bind")
  def test_bind_copy(self):
    q = TestHCQ.copy_queue()
    q.copy(TestHCQ.a.lazydata.buffer._buf.va_addr, TestHCQ.b.lazydata.buffer._buf.va_addr, 8)
    q.copy(TestHCQ.b.lazydata.buffer._buf.va_addr, TestHCQ.a.lazydata.buffer._buf.va_addr, 8)
    q.bind(TestHCQ.d0)
    for _ in range(1000):
      q.submit(TestHCQ.d0)
      TestHCQ.copy_queue().signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
      TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
      TestHCQ.d0.timeline_value += 1
    # confirm the signal didn't exceed the put value
    with self.assertRaises(RuntimeError):
      TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value + 1, timeout=50)
    val = TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[1]
    assert val == 0.0, f"got val {val}"

  def test_copy_bandwidth(self):
    # THEORY: the bandwidth is low here because it's only using one SDMA queue. I suspect it's more stable like this at least.
    SZ = 2_000_000_000
    a = Buffer(Device.DEFAULT, SZ, dtypes.uint8, options=BufferSpec(nolru=True)).allocate()
    b = Buffer(Device.DEFAULT, SZ, dtypes.uint8, options=BufferSpec(nolru=True)).allocate()
    q = TestHCQ.copy_queue()
    q.copy(a._buf.va_addr, b._buf.va_addr, SZ)
    et = _time_queue(q, TestHCQ.d0)
    gb_s = (SZ/1e9)/et
    print(f"same device copy:  {et*1e3:.2f} ms, {gb_s:.2f} GB/s")
    assert (0.3 if CI else 10) <= gb_s <= 1000

  def test_cross_device_copy_bandwidth(self):
    SZ = 2_000_000_000
    b = Buffer(f"{Device.DEFAULT}:1", SZ, dtypes.uint8, options=BufferSpec(nolru=True)).allocate()
    a = Buffer(Device.DEFAULT, SZ, dtypes.uint8, options=BufferSpec(nolru=True)).allocate()
    TestHCQ.d0._gpu_map(b._buf)
    q = TestHCQ.copy_queue()
    q.copy(a._buf.va_addr, b._buf.va_addr, SZ)
    et = _time_queue(q, TestHCQ.d0)
    gb_s = (SZ/1e9)/et
    print(f"cross device copy: {et*1e3:.2f} ms, {gb_s:.2f} GB/s")
    assert (0.3 if CI else 2) <= gb_s <= 50

  def test_interleave_compute_and_copy(self):
    q = TestHCQ.compute_queue()
    qc = TestHCQ.copy_queue()
    q.exec(TestHCQ.runner._prg, TestHCQ.d0.kernargs_ptr, TestHCQ.runner.p.global_size, TestHCQ.runner.p.local_size)  # b = [1, 2]
    q.signal(sig:=TestHCQ.d0._alloc_signal(value=0), value=1)
    qc.wait(sig, value=1)
    qc.copy(TestHCQ.a.lazydata.buffer._buf.va_addr, TestHCQ.b.lazydata.buffer._buf.va_addr, 8)
    qc.signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
    qc.submit(TestHCQ.d0)
    time.sleep(0.02) # give it time for the wait to fail
    q.submit(TestHCQ.d0)
    TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1
    val = TestHCQ.a.lazydata.buffer.as_buffer().cast("f")[0]
    assert val == 1.0, f"got val {val}"

  def test_cross_device_signal(self):
    d1 = Device[f"{Device.DEFAULT}:1"]
    q1 = TestHCQ.compute_queue()
    q2 = TestHCQ.compute_queue()
    q1.signal(sig:=TestHCQ.d0._alloc_signal(value=0), value=0xfff)
    q2.wait(sig, value=0xfff)
    q2.signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
    q2.submit(TestHCQ.d0)
    q1.signal(d1.timeline_signal, d1.timeline_value)
    q1.submit(d1)
    TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1
    d1._wait_signal(d1.timeline_signal, d1.timeline_value)
    d1.timeline_value += 1

  def test_timeline_signal_rollover(self):
    # NV 64bit, AMD 32bit
    TestHCQ.d0.timeline_value = (1 << 64) - 20 if Device.DEFAULT == "NV" else (1 << 32) - 20 # close value to reset
    TestHCQ.compute_queue().signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1).submit(TestHCQ.d0)
    TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1)

    for _ in range(40):
      q = TestHCQ.compute_queue()
      q.wait(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1)
      q.exec(TestHCQ.runner._prg, TestHCQ.d0.kernargs_ptr, TestHCQ.runner.p.global_size, TestHCQ.runner.p.local_size)
      q.signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
      TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
      TestHCQ.d0.timeline_value += 1
      val = TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[0]
      assert val == 1.0, f"got val {val}"

if __name__ == "__main__":
  unittest.main()
