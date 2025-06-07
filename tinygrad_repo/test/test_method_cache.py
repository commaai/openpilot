import unittest
from tinygrad import Tensor, Device, Variable
from examples.gpt2 import Transformer
from tinygrad.nn.state import get_state_dict

class TestMethodCache(unittest.TestCase):
  def setUp(self):
    self.backup_compiler = Device[Device.DEFAULT].compiler
  def tearDown(self):
    Device[Device.DEFAULT].compiler = self.backup_compiler

  def test_simple_methodcache(self):
    a = Tensor([1])
    b = Tensor([2])
    c = Tensor([3])
    d = Tensor([4])
    (a+b).realize()
    Device[Device.DEFAULT].compiler = None
    (c+d).realize()

  def test_nested_methodcache(self):
    a,b,c,d = Tensor([1]), Tensor([2]), Tensor([3]), Tensor([4])
    ((a+b)+(a+b)).realize()
    Device[Device.DEFAULT].compiler = None
    ((c+d)+(c+d)).realize()

  def test_nested_methodcache_swap(self):
    a,b,c,d = Tensor([1]), Tensor([2]), Tensor([3]), Tensor([4])
    ((a+b)+(c+d)).realize()
    Device[Device.DEFAULT].compiler = None
    ((c+d)+(a+b)).realize()

  @unittest.skip("incorrect use of transformer")
  def test_small_transformer(self):
    args_tiny = {"dim": 16, "n_heads": 8, "n_layers": 8, "norm_eps": 1e-05, "vocab_size": 10}
    model = Transformer(**args_tiny)
    for v in get_state_dict(model).values(): v.assign(Tensor.empty(*v.shape, dtype=v.dtype).realize())
    # NOTE: you have to do this twice due to the k-v cache
    for i in range(3): model(Tensor([[1,2,3,4]]), Variable("start_pos", 0, 10).bind(i)).realize()
    for i in range(3): model(Tensor([[1,2,3,4]]), Variable("start_pos", 0, 10).bind(i)).realize()
    Device[Device.DEFAULT].compiler = None
    for i in range(3): model(Tensor([[1,2,3,4]]), Variable("start_pos", 0, 10).bind(i)).realize()

if __name__ == '__main__':
  unittest.main()



