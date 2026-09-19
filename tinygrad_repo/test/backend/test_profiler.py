import unittest, struct, contextlib, statistics
from tinygrad import Device, Tensor, dtypes, TinyJit
from tinygrad.helpers import DEV, Context, ProfileRangeEvent, cpu_profile, cpu_events, flatten, ansistrip
from tinygrad.device import Buffer, BufferSpec, Compiled, ProfileDeviceEvent, ProfileGraphEvent
from extra.hcq1.hcq import HCQCompiled
from tinygrad.runtime.support.hcq2 import HCQ_DEVS
from tinygrad.engine.realize import run_linear
from tinygrad.uop.ops import UOp, Ops
from tinygrad.codegen import to_program

MOCKGPU = DEV.interface.startswith("MOCK")
def _dev_base(d):
  p = d.split(":")
  return p[0] if len(p) < 2 or not p[1].isdigit() else f"{p[0]}:{p[1]}"

@contextlib.contextmanager
def helper_collect_profile(*devs):
  for dev in devs: dev.synchronize()
  saved = [x for x in Compiled.profile_events if isinstance(x, ProfileDeviceEvent) and x.device.startswith("METAL")]
  Compiled.profile_events.clear()
  for x in saved: Compiled.profile_events.append(x)

  cpu_events.clear()

  profile_list = []
  with Context(PROFILE=1):
    yield profile_list
    for dev in devs: dev.synchronize()
    for dev in devs: dev._at_profile_finalize()
    for x in Compiled.profile_events: profile_list.append(x)
    profile_list.extend(cpu_events)

def filter_ranges(p): return flatten([e.ents for e in p if isinstance(e, ProfileGraphEvent)])+[e for e in p if isinstance(e, ProfileRangeEvent)]

def helper_profile_filter_device(profile, device:str):
  assert any(getattr(x, "device", None) == device and isinstance(x, ProfileDeviceEvent) for x in profile), f"device {device} is not registred"
  dev_events = [x for x in profile if getattr(x, "device", None) == device and isinstance(x, ProfileDeviceEvent)]
  assert len(dev_events) == 1, "only one device registration event is expected"
  return [x for x in profile if getattr(x, "device", None) == device], dev_events[0]

@unittest.skipUnless(isinstance(Device[Device.DEFAULT], HCQCompiled) or Device.DEFAULT in HCQ_DEVS | {"CPU", "METAL"}, "Dev not supported")
class TestSimpleProfiler(unittest.TestCase):
  def test_profiler(self):
    with helper_collect_profile(Device[Device.DEFAULT]) as profile:
      Tensor.empty(32).add(1).realize()
    profile = filter_ranges(profile)
    self.assertTrue(any(e.device == Device.DEFAULT for e in profile))

# TODO: support in HCQCompiled
is_cpu_hcq = Device.DEFAULT in {"CPU"}

@unittest.skipUnless((issubclass(type(Device[Device.DEFAULT]), HCQCompiled) and not is_cpu_hcq) or Device.DEFAULT in HCQ_DEVS | {"METAL"},
                     "Dev not supported")
class TestProfiler(unittest.TestCase):
  @classmethod
  def setUpClass(self):
    TestProfiler.d0 = Device[Device.DEFAULT]

    TestProfiler.a = Tensor([0.,1.], device=Device.DEFAULT).realize()
    TestProfiler.b = self.a + 1
    si = self.b.schedule_linear().src[-1]

    TestProfiler.prg = to_program(si.src[0], TestProfiler.d0.renderer)
    run_linear(UOp(Ops.LINEAR, src=(TestProfiler.prg.call(TestProfiler.b.uop, TestProfiler.a.uop),)))

  def test_profile_kernel_run(self, wait=False):
    runner_name = ansistrip(TestProfiler.prg.src[0].arg.name)
    with helper_collect_profile(TestProfiler.d0) as profile:
      run_linear(UOp(Ops.LINEAR, src=(TestProfiler.prg.call(TestProfiler.b.uop, TestProfiler.a.uop),)), wait=wait)

    helper_profile_filter_device(profile, TestProfiler.d0.device)
    profile = filter_ranges(profile)
    kernel_runs = [x for x in profile if x.device == TestProfiler.d0.device]
    assert len(kernel_runs) == 1, "one kernel run is expected"
    assert ansistrip(kernel_runs[0].name) == runner_name, "kernel name is not correct"
    assert _dev_base(kernel_runs[0].device) == kernel_runs[0].device, "kernel should not be on a sub-device"

  def test_profile_kernel_run_wait(self):
    self.test_profile_kernel_run(wait=True)

  def test_profile_copyin(self):
    buf1 = Buffer(Device.DEFAULT, 2, dtypes.float, options=BufferSpec(nolru=True)).ensure_allocated()

    with helper_collect_profile(TestProfiler.d0) as profile:
      buf1.copy_from(Buffer("PYTHON", 2, dtypes.float, opaque=memoryview(bytearray(struct.pack("ff", 0, 1)))))

    profile = filter_ranges(profile)
    kernel_runs = [x for x in profile if x.device.startswith((TestProfiler.d0.device, "PYTHON"))]
    self.assertEqual(len(kernel_runs), 2 if Device.DEFAULT in HCQ_DEVS else 1)

  def test_profile_multiops(self):
    runner_name = ansistrip(TestProfiler.prg.src[0].arg.name)
    host = getattr(TestProfiler.d0, "host", "PYTHON") # allocator-owned host memory always maps, opaque bytearrays only when page aligned by luck
    buf1 = Buffer(Device.DEFAULT, 2, dtypes.float, options=BufferSpec(nolru=True)).ensure_allocated()
    src, dst = Buffer(host, 2, dtypes.float, initial_value=struct.pack("ff", 0, 1)), Buffer(host, 2, dtypes.float, preallocate=True)

    with helper_collect_profile(TestProfiler.d0) as profile:
      buf1.copy_from(src)
      run_linear(UOp(Ops.LINEAR, src=(TestProfiler.prg.call(UOp.from_buffer(buf1), TestProfiler.a.uop),)))
      dst.copy_from(buf1)

    profile = filter_ranges(profile)
    evs = [x for x in profile if x.device.startswith((TestProfiler.d0.device, f"{host}:COPY"))]

    assert len(evs) == 3, "unexpected number of kernel and copy events"
    # NOTE: order of events does not matter, the tool is responsible for sorting them
    prg_events = [e for e in evs if e.device == TestProfiler.d0.device]
    assert any(ansistrip(e.name) == runner_name for e in prg_events), "kernel name is not correct"

    #for i in range(1, 3):
    #  assert evs[i].st > evs[i-1].en, "timestamp not aranged"

  def test_profile_multidev(self):
    try: d1 = Device[f"{Device.DEFAULT}:1"]
    except Exception as e: self.skipTest(f"second device not available {e}")

    buf1 = Buffer(Device.DEFAULT, 2, dtypes.float, options=BufferSpec(nolru=True)).ensure_allocated()
    buf2 = Buffer(f"{Device.DEFAULT}:1", 2, dtypes.float, options=BufferSpec(nolru=True)).ensure_allocated()

    with helper_collect_profile(TestProfiler.d0, d1) as profile:
      buf1.copy_from(Buffer("PYTHON", 2, dtypes.float, opaque=memoryview(bytearray(struct.pack("ff", 0, 1)))))
      buf2.copy_from(Buffer("PYTHON", 2, dtypes.float, opaque=memoryview(bytearray(struct.pack("ff", 0, 1)))))

    profile = filter_ranges(profile)
    for dev in [TestProfiler.d0.device, d1.device]:
      evs = [x for x in profile if _dev_base(x.device) == dev]
      assert len(evs) == (0 if buf1._host_mv() is not None else 1), "one kernel runs are expected"

  def test_profile_multidev_transfer(self):
    try: d1 = Device[f"{Device.DEFAULT}:1"]
    except Exception as e: self.skipTest(f"second device not available {e}")
    if Device.DEFAULT == "CUDA": self.skipTest("CUDA has no p2p: a transfer is two staged copies")

    buf1 = Tensor.randn(10, 10, device=f"{Device.DEFAULT}:0").realize()
    with helper_collect_profile(TestProfiler.d0, d1) as profile:
      buf1.to(f"{Device.DEFAULT}:1").realize()

    profile = filter_ranges(profile)
    kernel_runs = [x for x in profile if x.device.startswith(TestProfiler.d0.device)]
    assert len(kernel_runs) == 1, "one kernel run is expected"

  @unittest.skipIf(Device.DEFAULT in "METAL", "METAL does not support queue wait interrupts")
  def test_profile_graph(self):
    try: d1 = Device[f"{Device.DEFAULT}:1"]
    except Exception as e: self.skipTest(f"second device not available {e}")
    if Device.DEFAULT == "CUDA": self.skipTest("CUDA has no p2p: a transfer is two staged copies")

    def f(a):
      x = (a + 1).realize()
      return x, x.to(d1.device).realize()

    a = Tensor.randn(10, 10, device=TestProfiler.d0.device).realize()
    with helper_collect_profile(TestProfiler.d0, d1) as profile:
      jf = TinyJit(f)
      for _ in range(3): jf(a)
      del jf

    graph_evs = [x for x in profile if isinstance(x, ProfileGraphEvent)]

    _, _ = helper_profile_filter_device(profile, TestProfiler.d0.device)
    _, _ = helper_profile_filter_device(profile, d1.device)

    if Device.DEFAULT in HCQ_DEVS:
      ents = flatten(e.ents for e in graph_evs)
      assert len(ents) == 4, "two kernel runs and two copies are expected"
      assert sum(ansistrip(e.name).startswith("copy") for e in ents) == 2, "two copies are expected"
    else:
      assert len(graph_evs) == 2, "2 graph events are expected"
      assert len(graph_evs[0].ents) == 2, "two entities are expected"

  @unittest.skipIf(MOCKGPU, "skip MOCKGPU")
  @unittest.skipUnless(issubclass(type(Device[Device.DEFAULT]), HCQCompiled), "must be HCQ")
  def test_dev_jitter_matrix(self):
    dev_cnt = 6
    try: devs = [Device[f"{Device.DEFAULT}:{i}"] for i in range(dev_cnt)]
    except Exception as e: self.skipTest(f"multiple devices not available {e}")

    for dev in devs: dev.synchronize()
    for dev in devs: dev._at_profile_finalize()

    def _sync_d2d(d1:HCQCompiled, d2:HCQCompiled):
      d1.hw_compute_queue_t().signal(d1.timeline_signal, d1.timeline_value).wait(d2.timeline_signal, d2.timeline_value) \
                             .timestamp(d1.timeline_signal).signal(d1.timeline_signal, d1.timeline_value+1).submit(d1)
      d2.hw_compute_queue_t().signal(d2.timeline_signal, d2.timeline_value).wait(d1.timeline_signal, d1.timeline_value) \
                             .timestamp(d2.timeline_signal).signal(d2.timeline_signal, d2.timeline_value+1).submit(d2)
      d1.timeline_value += 2
      d2.timeline_value += 2
      d1.timeline_signal.wait(d1.timeline_value - 1)
      d2.timeline_signal.wait(d2.timeline_value - 1)
      return d2.timeline_signal.timestamp - d1.timeline_signal.timestamp

    # then test it by timing the GPU to GPU times
    dev_evs = {x.device:x for x in Compiled.profile_events if isinstance(x, ProfileDeviceEvent)}
    jitter_matrix = [[float('nan')] * len(devs) for _ in range(len(devs))]
    pairs = [(p1, p2) for p1 in enumerate(devs) for p2 in enumerate(devs) if p1 != p2]
    for (i1, d1), (i2, d2) in pairs:
      cpu_diff = dev_evs[d1.device].tdiff - dev_evs[d2.device].tdiff
      jitter_matrix[i1][i2] = statistics.median(_sync_d2d(d1, d2) - _sync_d2d(d2, d1) for _ in range(20)) / 2 - cpu_diff

    print("pairwise clock jitter matrix (us):\n" + '\n'.join([''.join([f'{float(item):8.3f}' for item in row]) for row in jitter_matrix]))

    for (i1, d1), (i2, d2) in pairs:
      assert abs(jitter_matrix[i1][i2]) < 0.5, "jitter should be less than 0.5us"

  def test_cpu_profile(self):
    def test_fxn(err=False):
      if err: raise Exception()

    with helper_collect_profile(dev:=TestProfiler.d0) as profile:
      with cpu_profile("test_1", dev):
        test_fxn(err=False)
      with self.assertRaises(Exception):
        with cpu_profile("test_2", dev):
          test_fxn(err=True)

    range_events = [p for p in profile if isinstance(p, ProfileRangeEvent) and p.device == dev]
    self.assertEqual(len(range_events), 2)

if __name__ == "__main__":
  unittest.main()
