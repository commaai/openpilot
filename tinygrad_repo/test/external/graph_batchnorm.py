import unittest
from tinygrad.nn.state import get_parameters
from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d, BatchNorm2d, optim

def model_step(lm):
  with Tensor.train():
    x = Tensor.ones(8,12,128,256, requires_grad=False)
    optimizer = optim.SGD(get_parameters(lm), lr=0.001)
    loss = lm.forward(x).sum()
    optimizer.zero_grad()
    loss.backward()
    del x,loss
    optimizer.step()

class TestBatchnorm(unittest.TestCase):
  def test_conv(self):
    class LilModel:
      def __init__(self):
        self.c = Conv2d(12, 32, 3, padding=1, bias=False)
      def forward(self, x):
        return self.c(x).relu()
    lm = LilModel()
    model_step(lm)

  def test_two_conv(self):
    class LilModel:
      def __init__(self):
        self.c = Conv2d(12, 32, 3, padding=1, bias=False)
        self.c2 = Conv2d(32, 32, 3, padding=1, bias=False)
      def forward(self, x):
        return self.c2(self.c(x)).relu()
    lm = LilModel()
    model_step(lm)

  def test_two_conv_bn(self):
    class LilModel:
      def __init__(self):
        self.c = Conv2d(12, 24, 3, padding=1, bias=False)
        self.bn = BatchNorm2d(24, track_running_stats=False)
        self.c2 = Conv2d(24, 32, 3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(32, track_running_stats=False)
      def forward(self, x):
        x = self.bn(self.c(x)).relu()
        return self.bn2(self.c2(x)).relu()
    lm = LilModel()
    model_step(lm)

  def test_conv_bn(self):
    class LilModel:
      def __init__(self):
        self.c = Conv2d(12, 32, 3, padding=1, bias=False)
        self.bn = BatchNorm2d(32, track_running_stats=False)
      def forward(self, x):
        return self.bn(self.c(x)).relu()
    lm = LilModel()
    model_step(lm)


if __name__ == '__main__':
  unittest.main()
