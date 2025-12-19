#!/usr/bin/env python
import pathlib
import unittest
import numpy as np
from tinygrad.tensor import Tensor

class TestVGG7(unittest.TestCase):
  def test_vgg7(self):
    from examples.vgg7_helpers.waifu2x import Vgg7, image_load

    # Create in tinygrad
    Tensor.manual_seed(1337)
    mdl = Vgg7()
    mdl.load_from_pretrained()

    # Scale up an image
    test_x = image_load(pathlib.Path(__file__).parent / 'waifu2x/input.png')
    test_y = image_load(pathlib.Path(__file__).parent / 'waifu2x/output.png')
    scaled = mdl.forward_tiled(test_x, 156)
    scaled = np.fmax(0, np.fmin(1, scaled))
    np.testing.assert_allclose(scaled, test_y, atol=5e-3, rtol=5e-3)

if __name__ == '__main__':
  unittest.main()
