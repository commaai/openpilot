import unittest, struct
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp

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
  def test_data(self):
    a = Tensor([1,2,3,4], dtype=dtypes.int32)
    dat = a.data()
    assert dat.itemsize == 4
    assert list(dat) == [1,2,3,4]
    assert dat.shape == (4,)
    assert dat[0] == 1
    assert dat[1] == 2

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
    self.assertEqual(Tensor.const(dtypes.int8, UOp.const(dtypes.float32, 1.0)).dtype, dtypes.int8)
    self.assertEqual(Tensor.const(dtypes.int32, UOp.variable("x", 1, 10).bind(5)).item(), 5)

  def test_data_float32(self):
    a = Tensor([[1,2.5],[3,4]], dtype=dtypes.float32)
    dat = a.data()
    assert dat.format == "f"
    assert dat[0, 1] == 2.5

  @unittest.skip("requires python 3.12")
  def test_data_float16(self):
    a = Tensor([[1,2.5],[3,4]], dtype=dtypes.float16)
    dat = a.data()
    assert dat.format == "e"
    assert dat.shape == (2,2)
    # NOTE: python can't deref float16

  def test_tolist_empty_shapes(self):
    for shape, expected in (((0,), []), ((2, 0), [[], []]), ((0, 2), []),
                            ((2, 0, 3), [[], []]), ((2, 3, 0), [[[], [], []], [[], [], []]])):
      self.assertEqual(Tensor.ones(*shape).tolist(), expected)

if __name__ == '__main__':
  unittest.main()