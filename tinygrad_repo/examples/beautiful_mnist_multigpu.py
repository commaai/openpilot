# model based off https://towardsdatascience.com/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
from typing import List, Callable
from tinygrad import Tensor, TinyJit, nn, GlobalCounters, Device
from tinygrad.helpers import getenv, colored, trange
from tinygrad.nn.datasets import mnist

GPUS = tuple(f'{Device.DEFAULT}:{i}' for i in range(getenv("GPUS", 2)))

class Model:
  def __init__(self):
    self.layers: List[Callable[[Tensor], Tensor]] = [
      nn.Conv2d(1, 32, 5), Tensor.relu,
      nn.Conv2d(32, 32, 5), Tensor.relu,
      nn.BatchNorm2d(32), Tensor.max_pool2d,
      nn.Conv2d(32, 64, 3), Tensor.relu,
      nn.Conv2d(64, 64, 3), Tensor.relu,
      nn.BatchNorm2d(64), Tensor.max_pool2d,
      lambda x: x.flatten(1), nn.Linear(576, 10)]

  def __call__(self, x:Tensor) -> Tensor: return x.sequential(self.layers)

if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = mnist()
  # we shard the test data on axis 0
  X_test.shard_(GPUS, axis=0)
  Y_test.shard_(GPUS, axis=0)

  model = Model()
  for k, x in nn.state.get_state_dict(model).items(): x.to_(GPUS)  # we put a copy of the model on every GPU
  opt = nn.optim.Adam(nn.state.get_parameters(model))

  @TinyJit
  def train_step() -> Tensor:
    with Tensor.train():
      opt.zero_grad()
      samples = Tensor.randint(getenv("BS", 512), high=X_train.shape[0])
      Xt, Yt = X_train[samples].shard_(GPUS, axis=0), Y_train[samples].shard_(GPUS, axis=0)  # we shard the data on axis 0
      # TODO: this "gather" of samples is very slow. will be under 5s when this is fixed
      loss = model(Xt).sparse_categorical_crossentropy(Yt).backward()
      opt.step()
      return loss

  @TinyJit
  def get_test_acc() -> Tensor: return (model(X_test).argmax(axis=1) == Y_test).mean()*100

  test_acc = float('nan')
  for i in (t:=trange(getenv("STEPS", 70))):
    GlobalCounters.reset()   # NOTE: this makes it nice for DEBUG=2 timing
    loss = train_step()
    if i%10 == 9: test_acc = get_test_acc().item()
    t.set_description(f"loss: {loss.item():6.2f} test_accuracy: {test_acc:5.2f}%")

  # verify eval acc
  if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
    if test_acc >= target: print(colored(f"{test_acc=} >= {target}", "green"))
    else: raise ValueError(colored(f"{test_acc=} < {target}", "red"))
