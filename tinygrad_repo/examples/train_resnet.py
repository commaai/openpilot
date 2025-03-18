#!/usr/bin/env python3
import numpy as np
from PIL import Image

from tinygrad.nn.state import get_parameters
from tinygrad.nn import optim
from tinygrad.helpers import getenv
from extra.training import train, evaluate
from extra.models.resnet import ResNet
from extra.datasets import fetch_mnist


class ComposeTransforms:
  def __init__(self, trans):
    self.trans = trans

  def __call__(self, x):
    for t in self.trans:
      x = t(x)
    return x

if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = fetch_mnist()
  X_train = X_train.reshape(-1, 28, 28).astype(np.uint8)
  X_test = X_test.reshape(-1, 28, 28).astype(np.uint8)
  classes = 10

  TRANSFER = getenv('TRANSFER')
  model = ResNet(getenv('NUM', 18), num_classes=classes)
  if TRANSFER:
    model.load_from_pretrained()

  lr = 5e-3
  transform = ComposeTransforms([
    lambda x: [Image.fromarray(xx, mode='L').resize((64, 64)) for xx in x],
    lambda x: np.stack([np.asarray(xx) for xx in x], 0),
    lambda x: x / 255.0,
    lambda x: np.tile(np.expand_dims(x, 1), (1, 3, 1, 1)).astype(np.float32),
  ])
  for _ in range(5):
    optimizer = optim.SGD(get_parameters(model), lr=lr, momentum=0.9)
    train(model, X_train, Y_train, optimizer, 100, BS=32, transform=transform)
    evaluate(model, X_test, Y_test, num_classes=classes, transform=transform)
    lr /= 1.2
    print(f'reducing lr to {lr:.7f}')
