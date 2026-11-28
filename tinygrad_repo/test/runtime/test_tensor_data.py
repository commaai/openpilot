import unittest, struct, pickle
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp
from tinygrad.nn.state import TensorIO

# format types: https://docs.python.org/3/library/struct.html

class TestTensorBytes(unittest.TestCase):
  def test_bytes(self):
    lst = Tensor(bytes(b"\xaa\xbb\xcc\xdd"))
    assert lst.tolist() == [170, 187, 204, 221]

  def test_float_bytes(self):
    lst = Tensor(bytes(struct.pack("ff", 0.234, 0.8585)), dtype=dtypes.float32)
    assert lst.shape == (2,)
    assert abs(lst.tolist()[0] - 0.234) < 1e-6
    assert abs(lst.tolist()[1] - 0.8585) < 1e-6

class TestTensorData(unittest.TestCase):
  def test_data_empty(self):
    a = Tensor([], dtype=dtypes.int32)
    dat = a.data()
    assert dat.itemsize == 4
    assert list(dat) == []
    assert dat.shape == (0,)

  def test_data_empty_multi_dim(self):
    a = Tensor([], dtype=dtypes.int32).reshape(0, 2)
    dat = a.data()
    assert dat.itemsize == 4
    assert list(dat) == []
    assert dat.shape == (0,)

  def test_data_uint8(self):
    a = Tensor([1,2,3,4], dtype=dtypes.uint8)
    dat = a.data()
    assert dat.format == "B"
    assert dat.itemsize == 1
    assert dat[0] == 1
    assert dat[1] == 2

  def test_data_nested(self):
    a = Tensor([[1,2],[3,4]], dtype=dtypes.int32)
    dat = a.data()
    assert dat.format == "i"
    assert dat.itemsize == 4
    assert dat.tolist() == [[1, 2], [3, 4]]
    assert dat.shape == (2,2)
    assert dat[0, 0] == 1
    assert dat[1, 1] == 4

  def test_data_const(self):
    a = Tensor(3, dtype=dtypes.int32)
    dat = a.data()
    assert dat.format == "i"
    assert dat.itemsize == 4
    assert dat.tolist() == 3
    assert dat.shape == ()

  def test_const_dtype_for_uop(self):
    self.assertEqual(Tensor.const(UOp.const(1.0).cast(dtypes.float32), dtypes.int8).dtype, dtypes.int8)
    self.assertEqual(Tensor.const(UOp.variable("x", 1, 10).bind(5), dtypes.int32).item(), 5)

  def test_data_float32(self):
    a = Tensor([[1,2.5],[3,4]], dtype=dtypes.float32)
    dat = a.data()
    assert dat.format == "f"
    assert dat[0, 1] == 2.5

  def test_tolist_empty_shapes(self):
    for shape, expected in (((0,), []), ((2, 0), [[], []]), ((0, 2), []),
                            ((2, 0, 3), [[], []]), ((2, 3, 0), [[[], [], []], [[], [], []]])):
      self.assertEqual(Tensor.ones(*shape).tolist(), expected)

class TestTensorIO(unittest.TestCase):
  def test_read(self):
    data = b"Hello World!"
    fobj = TensorIO(Tensor(data))
    self.assertEqual(fobj.read(1), data[:1])
    self.assertEqual(fobj.read(5), data[1:6])
    self.assertEqual(fobj.read(100), data[6:])
    self.assertEqual(fobj.read(100), b"")

  def test_read_nolen(self):
    data = b"Hello World!"
    fobj = TensorIO(Tensor(data))
    fobj.seek(2)
    self.assertEqual(fobj.read(), data[2:])

class TestToNumpy(unittest.TestCase):
  def test_numpy_is_numpy(self):
    output = Tensor.ones((1, 3, 4096)).realize().numpy()
    new = np.copy(output)
    print(type(new))
    serialized = pickle.dumps(new)
    out = pickle.loads(serialized)
    assert out.shape == (1,3,4096)
    assert (out==1).all()

if __name__ == '__main__':
  unittest.main()