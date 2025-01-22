from tinygrad.helpers import trange
from tinygrad.nn.datasets import mnist
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from functools import partial

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.c1 = nn.Conv2d(1, 32, 5)
    self.c2 = nn.Conv2d(32, 32, 5)
    self.bn1 = nn.BatchNorm(32)
    self.m1 = nn.MaxPool2d(2)
    self.c3 = nn.Conv2d(32, 64, 3)
    self.c4 = nn.Conv2d(64, 64, 3)
    self.bn2 = nn.BatchNorm(64)
    self.m2 = nn.MaxPool2d(2)
    self.lin = nn.Linear(576, 10)
  def __call__(self, x):
    x = mx.maximum(self.c1(x), 0)
    x = mx.maximum(self.c2(x), 0)
    x = self.m1(self.bn1(x))
    x = mx.maximum(self.c3(x), 0)
    x = mx.maximum(self.c4(x), 0)
    x = self.m2(self.bn2(x))
    return self.lin(mx.flatten(x, 1))

if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = mnist()
  X_train = mx.array(X_train.float().permute((0,2,3,1)).numpy())
  Y_train = mx.array(Y_train.numpy())
  X_test = mx.array(X_test.float().permute((0,2,3,1)).numpy())
  Y_test = mx.array(Y_test.numpy())

  model = Model()
  optimizer = optim.Adam(1e-3)
  def loss_fn(model, x, y): return nn.losses.cross_entropy(model(x), y).mean()

  state = [model.state, optimizer.state]
  @partial(mx.compile, inputs=state, outputs=state)
  def step(samples):
    # Compiled functions will also treat any inputs not in the parameter list as constants.
    X,Y = X_train[samples], Y_train[samples]
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad_fn(model, X, Y)
    optimizer.update(model, grads)
    return loss

  test_acc = float('nan')
  for i in (t:=trange(70)):
    samples = mx.random.randint(0, X_train.shape[0], (512,))  # putting this in JIT didn't work well
    loss = step(samples)
    if i%10 == 9: test_acc = ((model(X_test).argmax(axis=-1) == Y_test).sum() * 100 / X_test.shape[0]).item()
    t.set_description(f"loss: {loss.item():6.2f} test_accuracy: {test_acc:5.2f}%")
