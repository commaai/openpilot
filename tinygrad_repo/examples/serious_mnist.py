#!/usr/bin/env python
#inspired by https://github.com/Matuzas77/MNIST-0.17/blob/master/MNIST_final_solution.ipynb
import sys
import numpy as np
from tinygrad.nn.state import get_parameters
from tinygrad.tensor import Tensor
from tinygrad.nn import BatchNorm2d, optim
from tinygrad.helpers import getenv
from extra.datasets import fetch_mnist
from extra.augment import augment_img
from extra.training import train, evaluate
GPU = getenv("GPU")
QUICK = getenv("QUICK")
DEBUG = getenv("DEBUG")

class SqueezeExciteBlock2D:
  def __init__(self, filters):
    self.filters = filters
    self.weight1 = Tensor.scaled_uniform(self.filters, self.filters//32)
    self.bias1 = Tensor.scaled_uniform(1,self.filters//32)
    self.weight2 = Tensor.scaled_uniform(self.filters//32, self.filters)
    self.bias2 = Tensor.scaled_uniform(1, self.filters)

  def __call__(self, input):
    se = input.avg_pool2d(kernel_size=(input.shape[2], input.shape[3])) #GlobalAveragePool2D
    se = se.reshape(shape=(-1, self.filters))
    se = se.dot(self.weight1) + self.bias1
    se = se.relu()
    se = se.dot(self.weight2) + self.bias2
    se = se.sigmoid().reshape(shape=(-1,self.filters,1,1)) #for broadcasting
    se = input.mul(se)
    return se

class ConvBlock:
  def __init__(self, h, w, inp, filters=128, conv=3):
    self.h, self.w = h, w
    self.inp = inp
    #init weights
    self.cweights = [Tensor.scaled_uniform(filters, inp if i==0 else filters, conv, conv) for i in range(3)]
    self.cbiases = [Tensor.scaled_uniform(1, filters, 1, 1) for i in range(3)]
    #init layers
    self._bn = BatchNorm2d(128)
    self._seb = SqueezeExciteBlock2D(filters)

  def __call__(self, input):
    x = input.reshape(shape=(-1, self.inp, self.w, self.h))
    for cweight, cbias in zip(self.cweights, self.cbiases):
      x = x.pad(padding=[1,1,1,1]).conv2d(cweight).add(cbias).relu()
    x = self._bn(x)
    x = self._seb(x)
    return x

class BigConvNet:
  def __init__(self):
    self.conv = [ConvBlock(28,28,1), ConvBlock(28,28,128), ConvBlock(14,14,128)]
    self.weight1 = Tensor.scaled_uniform(128,10)
    self.weight2 = Tensor.scaled_uniform(128,10)

  def parameters(self):
    if DEBUG: #keeping this for a moment
      pars = [par for par in get_parameters(self) if par.requires_grad]
      no_pars = 0
      for par in pars:
        print(par.shape)
        no_pars += np.prod(par.shape)
      print('no of parameters', no_pars)
      return pars
    else:
      return get_parameters(self)

  def save(self, filename):
    with open(filename+'.npy', 'wb') as f:
      for par in get_parameters(self):
        #if par.requires_grad:
        np.save(f, par.numpy())

  def load(self, filename):
    with open(filename+'.npy', 'rb') as f:
      for par in get_parameters(self):
        #if par.requires_grad:
        try:
          par.numpy()[:] = np.load(f)
          if GPU:
            par.gpu()
        except:
          print('Could not load parameter')

  def forward(self, x):
    x = self.conv[0](x)
    x = self.conv[1](x)
    x = x.avg_pool2d(kernel_size=(2,2))
    x = self.conv[2](x)
    x1 = x.avg_pool2d(kernel_size=(14,14)).reshape(shape=(-1,128)) #global
    x2 = x.max_pool2d(kernel_size=(14,14)).reshape(shape=(-1,128)) #global
    xo = x1.dot(self.weight1) + x2.dot(self.weight2)
    return xo


if __name__ == "__main__":
  lrs = [1e-4, 1e-5] if QUICK else [1e-3, 1e-4, 1e-5, 1e-5]
  epochss = [2, 1] if QUICK else [13, 3, 3, 1]
  BS = 32

  lmbd = 0.00025
  lossfn = lambda out,y: out.sparse_categorical_crossentropy(y) + lmbd*(model.weight1.abs() + model.weight2.abs()).sum()
  X_train, Y_train, X_test, Y_test = fetch_mnist()
  X_train = X_train.reshape(-1, 28, 28).astype(np.uint8)
  X_test = X_test.reshape(-1, 28, 28).astype(np.uint8)
  steps = len(X_train)//BS
  np.random.seed(1337)
  if QUICK:
    steps = 1
    X_test, Y_test = X_test[:BS], Y_test[:BS]

  model = BigConvNet()

  if len(sys.argv) > 1:
    try:
      model.load(sys.argv[1])
      print('Loaded weights "'+sys.argv[1]+'", evaluating...')
      evaluate(model, X_test, Y_test, BS=BS)
    except:
      print('could not load weights "'+sys.argv[1]+'".')

  if GPU:
    params = get_parameters(model)
    [x.gpu_() for x in params]

  for lr, epochs in zip(lrs, epochss):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1,epochs+1):
      #first epoch without augmentation
      X_aug = X_train if epoch == 1 else augment_img(X_train)
      train(model, X_aug, Y_train, optimizer, steps=steps, lossfn=lossfn, BS=BS)
      accuracy = evaluate(model, X_test, Y_test, BS=BS)
      model.save(f'examples/checkpoint{accuracy * 1e6:.0f}')
