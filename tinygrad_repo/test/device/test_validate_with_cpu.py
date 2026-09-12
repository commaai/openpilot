import unittest
from tinygrad import Tensor, Context, Variable, Device
from test.helpers import needs_second_gpu

class TestValidateWithCPU(unittest.TestCase):
  def setUp(self):
    self.ctx = Context(VALIDATE_WITH_CPU=1)
    self.ctx.__enter__()
  def tearDown(self): self.ctx.__exit__(None, None, None)

  def test_add(self): self.assertListEqual((Tensor([1.,2,3])+Tensor([4.,5,6])).tolist(), [5.0, 7.0, 9.0])
  def test_mul(self): self.assertListEqual((Tensor([1.,2,3])*Tensor([4.,5,6])).tolist(), [4.0, 10.0, 18.0])
  def test_sum(self): self.assertEqual(Tensor([1.,2,3,4]).sum().item(), 10.0)
  def test_reduce_then_op(self): self.assertEqual((Tensor([1.,2,3,4]).sum() * 2).item(), 20.0)

  def test_assign(self):
    a = Tensor([1.,2,3]).realize()
    a.assign(a + 1).realize()
    self.assertListEqual(a.tolist(), [2.0, 3.0, 4.0])

  def test_buffer_view(self):
    self.assertListEqual((Tensor([1.,2,3,4,5,6,7,8])[2:6] + 1).tolist(), [4.0, 5.0, 6.0, 7.0])

  def test_symbolic(self):
    i = Variable('i', 1, 10)
    ones = Tensor.ones(10).contiguous()
    self.assertListEqual((ones[:i.bind(5)] + 1).contiguous()[:5].tolist(), [2.0]*5)

  def test_multi_kernel(self):
    a = (Tensor([1.,2,3]) + 1).contiguous()
    b = (a * 2).contiguous()
    self.assertListEqual((b - 1).tolist(), [3.0, 5.0, 7.0])

  @needs_second_gpu
  def test_sharded(self):
    t = Tensor([1.,2,3,4]).shard((f"{Device.DEFAULT}:0", f"{Device.DEFAULT}:1"), axis=0)
    self.assertListEqual((t + 1).tolist(), [2.0, 3.0, 4.0, 5.0])

if __name__ == "__main__":
  unittest.main()
