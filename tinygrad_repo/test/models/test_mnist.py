#!/usr/bin/env python
import unittest
import numpy as np
from tinygrad import Tensor, Device
from tinygrad.helpers import CI
from tinygrad.nn.state import get_parameters
from tinygrad.nn import optim, BatchNorm2d
from extra.training import train, evaluate
from extra.datasets import fetch_mnist

# load the mnist dataset
X_train, Y_train, X_test, Y_test = fetch_mnist()

# create a model
class TinyBobNet:
  def __init__(self):
    self.l1 = Tensor.scaled_uniform(784, 128)
    self.l2 = Tensor.scaled_uniform(128, 10)

  def parameters(self):
    return get_parameters(self)

  def forward(self, x):
    return x.dot(self.l1).relu().dot(self.l2)

# create a model with a conv layer
class TinyConvNet:
  def __init__(self, has_batchnorm=False):
    # https://keras.io/examples/vision/mnist_convnet/
    conv = 3
    #inter_chan, out_chan = 32, 64
    inter_chan, out_chan = 8, 16   # for speed
    self.c1 = Tensor.scaled_uniform(inter_chan,1,conv,conv)
    self.c2 = Tensor.scaled_uniform(out_chan,inter_chan,conv,conv)
    self.l1 = Tensor.scaled_uniform(out_chan*5*5, 10)
    if has_batchnorm:
      self.bn1 = BatchNorm2d(inter_chan)
      self.bn2 = BatchNorm2d(out_chan)
    else:
      self.bn1, self.bn2 = lambda x: x, lambda x: x

  def parameters(self):
    return get_parameters(self)

  def forward(self, x:Tensor):
    x = x.reshape(shape=(-1, 1, 28, 28)) # hacks
    x = self.bn1(x.conv2d(self.c1)).relu().max_pool2d()
    x = self.bn2(x.conv2d(self.c2)).relu().max_pool2d()
    x = x.reshape(shape=[x.shape[0], -1])
    return x.dot(self.l1)

@unittest.skipIf(CI and Device.DEFAULT == "CPU", "slow")
class TestMNIST(unittest.TestCase):
  def test_sgd_onestep(self):
    np.random.seed(1337)
    model = TinyBobNet()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    train(model, X_train, Y_train, optimizer, BS=69, steps=1)
    for p in model.parameters(): p.realize()

  def test_sgd_threestep(self):
    np.random.seed(1337)
    model = TinyBobNet()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    train(model, X_train, Y_train, optimizer, BS=69, steps=3)

  def test_sgd_sixstep(self):
    np.random.seed(1337)
    model = TinyBobNet()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    train(model, X_train, Y_train, optimizer, BS=69, steps=6, noloss=True)

  def test_adam_onestep(self):
    np.random.seed(1337)
    model = TinyBobNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, X_train, Y_train, optimizer, BS=69, steps=1)
    for p in model.parameters(): p.realize()

  def test_adam_threestep(self):
    np.random.seed(1337)
    model = TinyBobNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, X_train, Y_train, optimizer, BS=69, steps=3)

  def test_conv_onestep(self):
    np.random.seed(1337)
    model = TinyConvNet()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    train(model, X_train, Y_train, optimizer, BS=69, steps=1, noloss=True)
    for p in model.parameters(): p.realize()

  def test_conv(self):
    np.random.seed(1337)
    model = TinyConvNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, X_train, Y_train, optimizer, steps=100)
    assert evaluate(model, X_test, Y_test) > 0.93   # torch gets 0.9415 sometimes

  def test_conv_with_bn(self):
    np.random.seed(1337)
    model = TinyConvNet(has_batchnorm=True)
    optimizer = optim.AdamW(model.parameters(), lr=0.003)
    train(model, X_train, Y_train, optimizer, steps=200)
    assert evaluate(model, X_test, Y_test) > 0.94

  def test_sgd(self):
    np.random.seed(1337)
    model = TinyBobNet()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    train(model, X_train, Y_train, optimizer, steps=600)
    assert evaluate(model, X_test, Y_test) > 0.94   # CPU gets 0.9494 sometimes

if __name__ == '__main__':
  unittest.main()
