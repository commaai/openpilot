#!/usr/bin/env python
import unittest
from tinygrad import Tensor
from tinygrad.nn import Conv2d, Linear, Embedding
from tinygrad.nn.state import load_state_dict

class TestNN(unittest.TestCase):
  def test_conv2d_same_padding_invalid_stride(self):
    self.assertRaises(ValueError, Conv2d, in_channels=16, out_channels=32, kernel_size=2, stride=2, padding='same')

  def test_conv2d_same_padding_invalid_padding_str(self):
    self.assertRaises(ValueError, Conv2d, in_channels=16, out_channels=32, kernel_size=2, stride=1, padding='not_same')

  def test_embedding_shape(self):
    vocab_size, embed_size = 10, 16
    layer = Embedding(vocab_size, embed_size)
    for rank in range(5):
      shp = (1,) * rank
      a = Tensor([3]).reshape(shp)
      result = layer(a)
      self.assertEqual(result.shape, shp + (embed_size,))

  def test_load_state_dict_shape_mismatch(self):
    d1, d2 = 2, 4
    layer = Linear(d1, d1, bias=False)
    state_dict = {'weight': Tensor.randn(d2, d2)}
    with self.assertRaisesRegex(ValueError, r'Shape mismatch in layer `weight`: Expected shape \(2, 2\), but found \(4, 4\) in state dict.'):
      load_state_dict(layer, state_dict)

if __name__ == '__main__':
  unittest.main()
