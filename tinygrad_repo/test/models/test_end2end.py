import torch
from torch import nn
import unittest
import numpy as np
from tinygrad.nn.state import get_parameters, get_state_dict
from tinygrad.nn import optim, Linear, Conv2d, BatchNorm2d
from tinygrad.tensor import Tensor
from extra.datasets import fetch_mnist
from tinygrad.helpers import CI

def compare_tiny_torch(model, model_torch, X, Y):
  with Tensor.train():
    model_torch.train()
    model_state_dict = get_state_dict(model)
    for k,v in model_torch.named_parameters():
      if not CI: print(f"initting {k} from torch")
      model_state_dict[k].assign(Tensor(v.detach().numpy())).realize()

    optimizer = optim.SGD(get_parameters(model), lr=0.001)
    optimizer_torch = torch.optim.SGD(model_torch.parameters(), lr=0.001)

    Xt = torch.Tensor(X.numpy())
    np.testing.assert_allclose(X.numpy(), Xt.detach().numpy())

    out = model(X)
    loss = (out * Y).mean()

    out_torch = model_torch(torch.Tensor(X.numpy()))
    loss_torch = (out_torch * torch.Tensor(Y.numpy())).mean()

    # zero and backward
    optimizer.zero_grad()
    loss.backward()
    optimizer_torch.zero_grad()
    loss_torch.backward()

    # assert losses match
    if not CI: print(loss.realize().numpy())
    if not CI: print(loss_torch.detach().numpy())
    np.testing.assert_allclose(loss.realize().numpy(), loss_torch.detach().numpy(), atol=1e-4)

    for k,v in list(model_torch.named_parameters())[::-1]:
      g = model_state_dict[k].grad.numpy()
      gt = v.grad.detach().numpy()
      if not CI: print("testing grads", k, model_state_dict[k].grad.dtype)
      np.testing.assert_allclose(g, gt, atol=1e-3, err_msg=f'grad mismatch {k}')

    # take the steps
    optimizer.step()
    optimizer_torch.step()

    # assert weights match
    for k,v in model_torch.named_parameters():
      if not CI: print("testing weight", k, model_state_dict[k].dtype)
      np.testing.assert_allclose(model_state_dict[k].numpy(), v.detach().numpy(), atol=1e-3, err_msg=f'weight mismatch {k}')

def get_mnist_data():
  _X_train, _Y_train, X_test, Y_test = fetch_mnist()
  BS = 32
  num_classes = 10
  X = Tensor(X_test[0:BS].astype(np.float32))
  Y = np.zeros((BS, num_classes), np.float32)
  Y[range(BS),Y_test[0:BS]] = -1.0*num_classes
  return X, Tensor(Y)

class TestEnd2End(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.X, cls.Y = get_mnist_data()

  def setUp(self):
    torch.manual_seed(123)

  def test_linear_mnist(self):
    class LinTiny:
      def __init__(self, bias=False):
        self.l1 = Linear(784, 128, bias=bias)
        self.l2 = Linear(128, 10, bias=bias)
      def __call__(self, x):
        return self.l2(self.l1(x).relu()).log_softmax(-1)
    class LinTorch(nn.Module):
      def __init__(self, bias=False):
        super().__init__()
        self.l1 = nn.Linear(784, 128, bias=bias)
        self.l2 = nn.Linear(128, 10, bias=bias)
      def forward(self, x):
        return self.l2(self.l1(x).relu()).log_softmax(-1)
    compare_tiny_torch(LinTiny(), LinTorch(), self.X, self.Y)

  def test_bn_mnist(self):
    class LinTiny:
      def __init__(self):
        self.l1 = Linear(784, 128)
        self.l2 = Linear(128, 10)
        self.bn1 = BatchNorm2d(128)
      def __call__(self, x):
        return self.l2(self.bn1(self.l1(x).reshape(x.shape[0], -1, 1, 1)).reshape(x.shape[0], -1).relu()).log_softmax(-1)
    class LinTorch(nn.Module):
      def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(784, 128)
        self.l2 = nn.Linear(128, 10)
        self.bn1 = nn.BatchNorm2d(128)
      def forward(self, x):
        return self.l2(self.bn1(self.l1(x).reshape(x.shape[0], -1, 1, 1)).reshape(x.shape[0], -1).relu()).log_softmax(-1)
    compare_tiny_torch(LinTiny(), LinTorch(), self.X, self.Y)

  def test_bn_alone(self):
    np.random.seed(1337)
    X = Tensor(np.random.randn(32, 10, 1, 1).astype(np.float32))
    Y = Tensor(np.random.randn(32, 10, 1, 1).astype(np.float32))
    compare_tiny_torch(BatchNorm2d(10), nn.BatchNorm2d(10), X, Y)

  def test_bn_linear(self):
    BS, K = 2, 1
    eps = 0
    X = Tensor([1,0]).reshape(BS, K, 1, 1)
    Y = Tensor([-1,0]).reshape(BS, K, 1, 1)
    class LinTiny:
      def __init__(self):
        self.l1 = Conv2d(K, K, 1, bias=False)
        self.bn1 = BatchNorm2d(K, affine=False, track_running_stats=False, eps=eps)
      def __call__(self, x): return self.bn1(self.l1(x))
    class LinTorch(nn.Module):
      def __init__(self):
        super().__init__()
        self.l1 = nn.Conv2d(K, K, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(K, affine=False, track_running_stats=False, eps=eps)
      def forward(self, x): return self.bn1(self.l1(x))
    model_torch = LinTorch()
    with torch.no_grad():
      model_torch.l1.weight[:] = 1.
    compare_tiny_torch(LinTiny(), model_torch, X, Y)

  def test_conv_mnist(self):
    class LinTiny:
      def __init__(self, has_batchnorm=False):
        self.c1 = Conv2d(1, 8, 3, stride=2)
        self.c2 = Conv2d(8, 16, 3, stride=2)
        self.l1 = Linear(16*6*6, 10)
        if has_batchnorm:
          self.bn1, self.bn2 = BatchNorm2d(8), BatchNorm2d(16)
        else:
          self.bn1, self.bn2 = lambda x: x, lambda x: x
      def __call__(self, x):
        return self.l1(self.bn2(self.c2(self.bn1(self.c1(x)).relu())).relu().reshape(x.shape[0], -1)).log_softmax(-1)
    class LinTorch(nn.Module):
      def __init__(self, has_batchnorm=False):
        super().__init__()
        self.c1 = nn.Conv2d(1, 8, 3, stride=2)
        self.c2 = nn.Conv2d(8, 16, 3, stride=2)
        self.l1 = nn.Linear(16*6*6, 10)
        if has_batchnorm:
          self.bn1, self.bn2 = nn.BatchNorm2d(8), nn.BatchNorm2d(16)
        else:
          self.bn1, self.bn2 = lambda x: x, lambda x: x
      def forward(self, x):
        return self.l1(self.bn2(self.c2(self.bn1(self.c1(x)).relu())).relu().reshape(x.shape[0], -1)).log_softmax(-1)
    for has_batchnorm in [False, True]:
      with self.subTest(has_batchnorm=has_batchnorm):
        compare_tiny_torch(LinTiny(has_batchnorm), LinTorch(has_batchnorm), self.X.reshape((-1, 1, 28, 28)), self.Y)

if __name__ == "__main__":
  unittest.main()
