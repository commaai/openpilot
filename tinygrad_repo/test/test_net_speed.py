#!/usr/bin/env python
import time
import unittest
import torch
from tinygrad import Tensor, Device
from tinygrad.helpers import Profiling, CI

@unittest.skipIf(CI and Device.DEFAULT in {"CUDA", "NV"}, "slow")
class TestConvSpeed(unittest.TestCase):

  def test_mnist(self):
    # https://keras.io/examples/vision/mnist_convnet/
    conv = 3
    inter_chan, out_chan = 32, 64

    # ****** torch baseline *******

    torch.backends.mkldnn.enabled = False

    conv = 3
    inter_chan, out_chan = 32, 64
    c1 = torch.randn(inter_chan,1,conv,conv, requires_grad=True)
    c2 = torch.randn(out_chan,inter_chan,conv,conv, requires_grad=True)
    l1 = torch.randn(out_chan*5*5, 10, requires_grad=True)

    c2d = torch.nn.functional.conv2d
    mp = torch.nn.MaxPool2d((2,2))
    lsm = torch.nn.LogSoftmax(dim=1)

    cnt = 5
    fpt, bpt = 0.0, 0.0
    for i in range(cnt):
      et0 = time.time()
      x = torch.randn(128, 1, 28, 28, requires_grad=True)
      x = mp(c2d(x,c1).relu())
      x = mp(c2d(x,c2).relu())
      x = x.reshape(x.shape[0], -1)
      out = lsm(x.matmul(l1))
      out = out.mean()
      et1 = time.time()
      out.backward()
      et2 = time.time()
      fpt += (et1-et0)
      bpt += (et2-et1)

    fpt_baseline = (fpt*1000/cnt)
    bpt_baseline = (bpt*1000/cnt)
    print("torch forward pass:  %.3f ms" % fpt_baseline)
    print("torch backward pass: %.3f ms" % bpt_baseline)

    # ****** tinygrad compare *******

    c1 = Tensor(c1.detach().numpy(), requires_grad=True)
    c2 = Tensor(c2.detach().numpy(), requires_grad=True)
    l1 = Tensor(l1.detach().numpy(), requires_grad=True)

    cnt = 5
    fpt, bpt = 0.0, 0.0
    for i in range(1+cnt):
      et0 = time.time()
      x = Tensor.randn(128, 1, 28, 28)
      x = x.conv2d(c1).relu().avg_pool2d()
      x = x.conv2d(c2).relu().max_pool2d()
      x = x.reshape(shape=(x.shape[0], -1))
      out = x.dot(l1).log_softmax()
      out = out.mean()
      out.backward()  # NOTE: we have to now compute this here, but it doesn't realize
      out.realize()
      et1 = time.time()
      [x.grad.realize() for x in [c1, c2, l1]]
      et2 = time.time()
      if i == 0:
        pr = Profiling(sort='time', frac=0.2)
        pr.__enter__()
      else:
        fpt += (et1-et0)
        bpt += (et2-et1)

    pr.__exit__()
    fpt = (fpt*1000/cnt)
    bpt = (bpt*1000/cnt)
    print("forward pass:  %.3f ms, %.2fx off baseline %.3f ms" % (fpt, fpt/fpt_baseline, fpt_baseline))
    print("backward pass: %.3f ms, %.2fx off baseline %.3f ms" % (bpt, bpt/bpt_baseline, bpt_baseline))


if __name__ == '__main__':
  unittest.main()
