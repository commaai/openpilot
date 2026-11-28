import unittest, time
import numpy as np
from tinygrad import Device
from tinygrad.nn.state import get_parameters
from tinygrad.nn import optim
from tinygrad.helpers import getenv
from extra.training import train
from extra.models.convnext import ConvNeXt

BS = getenv("BS", 2)

def train_one_step(model,X,Y):
  params = get_parameters(model)
  pcount = 0
  for p in params:
    pcount += np.prod(p.shape)
  optimizer = optim.SGD(params, lr=0.001)
  print("stepping %r with %.1fM params bs %d" % (type(model), pcount/1e6, BS))
  st = time.time()
  train(model, X, Y, optimizer, steps=1, BS=BS)
  et = time.time()-st
  print("done in %.2f ms" % (et*1000.))

def check_gc():
  if Device.DEFAULT == "CL":
    from extra.introspection import print_objects
    assert print_objects() == 0

class TestTrain(unittest.TestCase):
  def test_convnext(self):
    model = ConvNeXt(depths=[1], dims=[16])
    X = np.zeros((BS,3,64,64), dtype=np.float32)
    Y = np.zeros((BS), dtype=np.int32)
    train_one_step(model,X,Y)
    check_gc()

if __name__ == '__main__':
  unittest.main()
