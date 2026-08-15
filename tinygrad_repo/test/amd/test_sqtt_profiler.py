import unittest, contextlib
from tinygrad import Device, Tensor, Context, TinyJit
from tinygrad.device import Compiled, ProfileProgramEvent, ProfileDeviceEvent
from tinygrad.engine.realize import run_linear
from tinygrad.codegen import to_program
from tinygrad.viz.serve import load_amd_counters, VizData

@contextlib.contextmanager
def save_sqtt():
  data = VizData()
  yield data.ctxs
  Device[Device.DEFAULT].synchronize()
  Device[Device.DEFAULT]._at_profile_finalize()
  load_amd_counters(data, Compiled.profile_events)
  data.ctxs[:] = [r for r in data.ctxs if r["name"].startswith("SQTT")]

@unittest.skipUnless(Device.DEFAULT == "AMD", "only runs on AMD")
class TestSQTTProfiler(unittest.TestCase):
  # TODO: can we enable SQTT profiling in context?
  @classmethod
  def setUpClass(cls):
    if not Device[Device.DEFAULT].sqtt_enabled: raise unittest.SkipTest("device must be in SQTT profiling mode")

  def setUp(self):
    Device[Device.DEFAULT].synchronize()
    Compiled.profile_events[:] = [e for e in Compiled.profile_events if isinstance(e, (ProfileProgramEvent, ProfileDeviceEvent))]

  def test_simple(self):
    t = Tensor.empty(1) + 1
    with save_sqtt() as sqtt:
      linear = t.schedule_linear()
      run_linear(linear)
    fn_name = to_program(linear.src[0].src[0], renderer=Device[Device.DEFAULT].renderer).arg.function_name
    self.assertEqual(len(sqtt), 1)
    self.assertEqual(sqtt[0]["name"], f"SQTT {fn_name}")

  def test_multiple_runs(self):
    t = Tensor.empty(1) + 1
    with save_sqtt() as sqtt:
      linear = t.schedule_linear()
      for _ in range(N:=3): run_linear(linear)
    fn_name = to_program(linear.src[0].src[0], renderer=Device[Device.DEFAULT].renderer).arg.function_name
    self.assertEqual(len(sqtt), N)
    for i in range(1, N):
      self.assertEqual(sqtt[i]["name"], f"SQTT {fn_name} n{i+1}")

  def test_multiple_kernels(self):
    t = ((Tensor.empty(1) + 1).contiguous() + 2)
    linear = t.schedule_linear()
    with save_sqtt() as sqtt:
      run_linear(linear)
    self.assertEqual(len(sqtt), len(linear.src))
    for i,call in enumerate(linear.src):
      fn_name = to_program(call.src[0], renderer=Device[Device.DEFAULT].renderer).arg.function_name
      self.assertEqual(sqtt[i]["name"], f"SQTT {fn_name}")

  def test_multiple_kernels_lower(self):
    t = ((Tensor.empty(1) + 1).contiguous() + 2)
    linear = t.schedule_linear()
    with save_sqtt() as sqtt:
      run_linear(linear)
    self.assertEqual(len(sqtt), len(linear.src))
    for i,call in enumerate(linear.src):
      fn_name = to_program(call.src[0], renderer=Device[Device.DEFAULT].renderer).arg.function_name
      self.assertEqual(sqtt[i]["name"], f"SQTT {fn_name}")

  def test_jit(self):
    @TinyJit
    def f(a): return a + 1
    t = Tensor.empty(1)
    with save_sqtt() as sqtt:
      for _ in range(N:=5):
        f(t).realize()
    self.assertEqual(len(sqtt), N)
    kernel_name = sqtt[0]["name"]
    for i,s in enumerate(sqtt[1:], start=1): self.assertEqual(s["name"], f"{kernel_name} n{i+1}")

  # TODO: can we trace SQTT for graphed kernels?
  def test_jit_graph(self, kernel_count=3*1):
    @TinyJit
    def f(a): return ((a + 1).contiguous() + 2).contiguous().sum()
    t = Tensor.empty(32)
    with save_sqtt() as sqtt:
      for _ in range(5):
        f(t).realize()
    names = [s["name"] for s in sqtt]
    k0, k1, k2 = names[:3]
    for i in range(3, len(sqtt), 3):
      n = (i // 3)+1
      self.assertEqual(names[i], f"{k0} n{n}")
      self.assertEqual(names[i+1], f"{k1} n{n}")
      self.assertEqual(names[i+2], f"{k2} n{n}")
    self.assertEqual(len(sqtt), kernel_count)

  @Context(JIT=2)
  def test_jit_multiple_kernels(self): self.test_jit_graph(kernel_count=3*5)

if __name__ == "__main__":
  unittest.main()
