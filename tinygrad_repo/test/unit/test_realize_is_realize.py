import tempfile, unittest
import numpy as np
from tinygrad import Tensor, Device, dtypes, Variable

class TestRealizeIsRealized(unittest.TestCase):
  def test_list(self):
    t = Tensor([1, 2, 3]).realize()
    assert t.uop.is_realized

  def test_rand(self):
    t = Tensor.rand(4, 4).realize()
    assert t.uop.is_realized

  def test_contiguous(self):
    t = Tensor.zeros(10).contiguous().realize()
    assert t.uop.is_realized

  def test_bytes(self):
    t = Tensor(b'\x01\x02\x03').realize()
    assert t.uop.is_realized

  def test_numpy(self):
    t = Tensor(np.array([1, 2, 3])).realize()
    assert t.uop.is_realized

  def test_multi(self):
    d = Device.DEFAULT
    t = Tensor.ones(8).contiguous().shard((d, d), axis=0).realize()
    assert all(u.is_realized for u in t.uop.src)

  def test_empty(self):
    t = Tensor.empty(4, 4).realize()
    assert not t.uop.is_realized

  def test_disk(self):
    with tempfile.NamedTemporaryFile() as f:
      f.write(b'\x00' * 16)
      f.flush()
      t = Tensor.empty(4, dtype=dtypes.float32, device=f"disk:{f.name}").realize()
      assert not t.uop.is_realized

  def test_assign(self):
    t = Tensor([1, 2, 3])
    t += 1
    t.realize()
    assert t.uop.is_realized

  # TODO: these are not realized after .realize()

  def test_const_not_realized(self):
    t = Tensor(3.14).realize()
    assert not t.uop.is_realized

  def test_ones_not_realized(self):
    t = Tensor.ones(4, 4).realize()
    assert not t.uop.is_realized

  def test_none_not_realized(self):
    t = Tensor(None).realize()
    assert not t.uop.is_realized

  def test_variable_not_realized(self):
    t = Tensor(Variable("v", 1, 10).bind(3)).realize()
    assert not t.uop.is_realized

if __name__ == "__main__":
  unittest.main()
