import unittest
from tinygrad import Tensor, dtypes
from tinygrad.tensor import _METADATA
from tinygrad.helpers import Context

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
    k = a.schedule()[-1]
    self.assertEqual([m.name for m in k.metadata], ["rand"])

  @unittest.skip("metadata not reaching kernel schedule")
  def test_exclude_const_metadata(self):
    a = Tensor.arange(4)
    b = Tensor.full((4,), -1, dtype=dtypes.int).contiguous()
    sched = Tensor.schedule(a, b)
    self.assertEqual([m.name for m in sched[0].metadata], ["arange"])
    self.assertEqual([m.name for m in sched[1].metadata], ["contiguous"])

  def test_matmul(self):
    x = Tensor.rand(3, requires_grad=True)
    W = Tensor.rand(3, 3, requires_grad=True)
    out = x.matmul(W)
    self.assertEqual(out.uop.metadata[0].name, "matmul")
    si = out.schedule()[-1]
    self.assertEqual(len(si.metadata), 1)
    self.assertEqual(si.metadata[0].name, "matmul")

  def test_relu(self):
    x = Tensor.rand(3, requires_grad=True)
    out = x.relu()
    self.assertEqual(out.uop.metadata[0].name, "relu")
    si = out.schedule()[-1]
    self.assertEqual(len(si.metadata), 1)
    self.assertEqual(si.metadata[0].name, "relu")

  @unittest.skip("assign metadata no longer captured")
  def test_assign(self):
    x = Tensor.empty(10, 10).realize()
    x.assign(Tensor.ones(10, 10).contiguous())
    si = x.schedule()[-1]
    self.assertEqual(len(si.metadata), 1)
    self.assertEqual(si.metadata[0].name, "assign")

  def test_complex(self):
    x = Tensor.rand(3, requires_grad=True)
    y = Tensor.rand(3, requires_grad=True)
    out = x.relu() * y.sigmoid()
    self.assertEqual(out.uop.metadata[0].name, "__mul__")
    self.assertEqual(out.uop.src[0].metadata[0].name, "relu")
    self.assertEqual(out.uop.src[1].metadata[0].name, "sigmoid")
    si = out.schedule()[-1]
    self.assertEqual(len(si.metadata), 3)
    self.assertEqual(set(m.name for m in si.metadata), {"relu", "sigmoid", "__mul__"})

  def test_complex_backward(self):
    x = Tensor.rand(3, requires_grad=True).realize()
    y = Tensor.rand(3, requires_grad=True).realize()
    out = (x.relu() * y.sigmoid()).sum()
    self.assertEqual(out.uop.metadata[0].name, "sum")
    out.backward()
    self.assertEqual(x.grad.uop.metadata[0].name, "relu")
    #self.assertTrue(x.grad.uop.metadata[0].backward)  # TODO: backward flag is False
    self.assertEqual(y.grad.uop.metadata[0].name, "sigmoid")
    #self.assertTrue(y.grad.uop.metadata[0].backward)  # TODO: backward flag is False
    si = Tensor.schedule(out, x.grad, y.grad)[-1]
    #self.assertEqual(len(si.metadata), 3, f"failed with {si.metadata}")
    # skip numpy, this is schedule cache
    self.assertSetEqual(set(m.name for m in si.metadata if m.name != "numpy"), {"sigmoid", "relu"})
    #bw = [m for m in si.metadata if m.backward]
    #self.assertEqual(len(bw), 1)
    #self.assertEqual(bw[0].name, "sigmoid")

  def test_tracemeta_0(self):
    with Context(TRACEMETA=0):
      x = Tensor.rand(3, requires_grad=True)
      y = Tensor.rand(3, requires_grad=True)
      out = (x.relu() * y.sigmoid()).sum()
      self.assertIsNone(out.uop.metadata)
      self.assertIsNone(out.uop.src[0].metadata)
      si = out.schedule()[-1]
      self.assertEqual(si.metadata, ())

if __name__ == '__main__':
  unittest.main()
