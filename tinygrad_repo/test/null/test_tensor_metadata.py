import unittest
from tinygrad import Tensor, dtypes
from tinygrad.tensor import _METADATA
from tinygrad.engine.realize import capturing
from tinygrad.helpers import Context

@unittest.skip("tensor metadata is no longer supported")
class TestTensorMetadata(unittest.TestCase):
  def setUp(self) -> None:
    _METADATA.set(None)
    self._ctx = Context(SCACHE=0)
    self._ctx.__enter__()
  def tearDown(self) -> None:
    self._ctx.__exit__(None, None, None)

  @unittest.skip("why would this be true?")
  def test_exclude_noop_metadata(self):
    a = Tensor.rand(4, 4)*1
    self.assertEqual(a.uop.metadata[0].name, "__mul__")
    k = a.schedule_linear().src[-1]
    self.assertEqual([m.name for m in k.arg.metadata], ["rand"])

  @unittest.skip("metadata not reaching kernel schedule")
  def test_exclude_const_metadata(self):
    a = Tensor.arange(4)
    b = Tensor.full((4,), -1, dtype=dtypes.int).contiguous()
    sched = a.schedule_linear(b)
    self.assertEqual([m.name for m in sched.src[0].arg.metadata], ["arange"])
    self.assertEqual([m.name for m in sched.src[1].arg.metadata], ["contiguous"])

  def test_matmul(self):
    x = Tensor.rand(3)
    W = Tensor.rand(3, 3)
    out = x.matmul(W)
    self.assertEqual(out.uop.metadata[0].name, "matmul")
    si = out.schedule_linear().src[-1]
    self.assertEqual(len(si.arg.metadata), 1)
    self.assertEqual(si.arg.metadata[0].name, "matmul")

  def test_relu(self):
    x = Tensor.rand(3)
    out = x.relu()
    self.assertEqual(out.uop.metadata[0].name, "relu")
    si = out.schedule_linear().src[-1]
    self.assertEqual(len(si.arg.metadata), 1)
    self.assertEqual(si.arg.metadata[0].name, "relu")

  @unittest.skip("assign metadata no longer captured")
  def test_assign(self):
    x = Tensor.empty(10, 10).realize()
    x.assign(Tensor.ones(10, 10).contiguous())
    si = x.schedule_linear().src[-1]
    self.assertEqual(len(si.arg.metadata), 1)
    self.assertEqual(si.arg.metadata[0].name, "assign")

  def test_complex(self):
    x = Tensor.rand(3)
    y = Tensor.rand(3)
    out = x.relu() * y.sigmoid()
    self.assertEqual(out.uop.metadata[0].name, "__mul__")
    self.assertEqual(out.uop.src[0].metadata[0].name, "relu")
    self.assertEqual(out.uop.src[1].metadata[0].name, "sigmoid")
    si = out.schedule_linear().src[-1]
    self.assertEqual(len(si.arg.metadata), 3)
    self.assertEqual(set(m.name for m in si.arg.metadata), {"relu", "sigmoid", "__mul__"})

  @unittest.skip("flaky")
  def test_complex_backward(self):
    x = Tensor.rand(3).realize()
    y = Tensor.rand(3).realize()
    out = (x.relu() * y.sigmoid()).sum()
    self.assertEqual(out.uop.metadata[0].name, "sum")
    out.backward()
    self.assertEqual(x.grad.uop.metadata[0].name, "relu")
    #self.assertTrue(x.grad.uop.metadata[0].backward)  # TODO: backward flag is False
    self.assertEqual(y.grad.uop.metadata[0].name, "sigmoid")
    #self.assertTrue(y.grad.uop.metadata[0].backward)  # TODO: backward flag is False
    si = out.schedule_linear(x.grad, y.grad).src[-1]
    #self.assertEqual(len(si.arg.metadata), 3, f"failed with {si.arg.metadata}")
    # skip numpy, this is schedule cache
    self.assertSetEqual(set(m.name for m in si.arg.metadata if m.name != "numpy"), {"sigmoid", "relu"})
    #bw = [m for m in si.metadata if m.backward]
    #self.assertEqual(len(bw), 1)
    #self.assertEqual(bw[0].name, "sigmoid")

  def test_tracemeta_0(self):
    with Context(TRACEMETA=0):
      x = Tensor.rand(3)
      y = Tensor.rand(3)
      out = (x.relu() * y.sigmoid()).sum()
      self.assertIsNone(out.uop.metadata)
      self.assertIsNone(out.uop.src[0].metadata)
      si = out.schedule_linear().src[-1]
      self.assertEqual(si.arg.metadata, ())

  def _has_metadata(self, h, name):
    linears = []
    capturing.append(type("", (), {"add_linear": lambda _, linear, var_vals: linears.append(linear)})())
    try: h.realize()
    finally: capturing.clear()
    calls = [call for linear in linears for call in linear.src]
    return any(m.name == name for call in calls for m in call.arg.metadata)

  def test_metadata_survives_realize_pending_assign(self):
    shared = Tensor.rand(4)
    c = Tensor.zeros(8).contiguous().realize()
    c[:4].assign(shared)
    self.assertTrue(self._has_metadata(c[:4].relu(), "relu"))

  @unittest.expectedFailure
  def test_metadata_lost_realize_pending_assign(self):
    shared = Tensor.rand(4)
    c = Tensor.zeros(8).contiguous().realize()
    c[:4].assign(shared)
    self.assertTrue(self._has_metadata((c[:4] + shared).relu(), "relu"))

class TestTraceMetaShutdown(unittest.TestCase):
  def test_tracemeta_del_no_shutdown_error(self):
    import subprocess, os
    result = subprocess.run(['python3', '-c', 'from tinygrad import Tensor\n'
                             'x=Tensor.eye(3); (x@x).sum().backward()'],
                            env={**os.environ, "TRACEMETA": "2"}, capture_output=True)
    self.assertEqual(result.returncode, 0)
    self.assertNotIn(b"Exception", result.stderr)

if __name__ == '__main__':
  unittest.main()
