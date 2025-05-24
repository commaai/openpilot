from tinygrad import dtypes, getenv, Device
from tinygrad.helpers import trange, colored, DEBUG, temp
from tinygrad.nn.datasets import mnist
import torch
from torch import nn, optim

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.c1 = nn.Conv2d(1, 32, 5)
    self.c2 = nn.Conv2d(32, 32, 5)
    self.bn1 = nn.BatchNorm2d(32)
    self.m1 = nn.MaxPool2d(2)
    self.c3 = nn.Conv2d(32, 64, 3)
    self.c4 = nn.Conv2d(64, 64, 3)
    self.bn2 = nn.BatchNorm2d(64)
    self.m2 = nn.MaxPool2d(2)
    self.lin = nn.Linear(576, 10)
  def forward(self, x):
    x = nn.functional.relu(self.c1(x))
    x = nn.functional.relu(self.c2(x), 0)
    x = self.m1(self.bn1(x))
    x = nn.functional.relu(self.c3(x), 0)
    x = nn.functional.relu(self.c4(x), 0)
    x = self.m2(self.bn2(x))
    return self.lin(torch.flatten(x, 1))

if __name__ == "__main__":
  if getenv("TINY_BACKEND"):
    import tinygrad.frontend.torch
    device = torch.device("tiny")
  else:
    device = torch.device({"METAL":"mps","NV":"cuda"}.get(Device.DEFAULT, "cpu"))
  if DEBUG >= 1: print(f"using torch backend {device}")
  X_train, Y_train, X_test, Y_test = mnist()
  X_train = torch.tensor(X_train.float().numpy(), device=device)
  Y_train = torch.tensor(Y_train.cast(dtypes.int64).numpy(), device=device)
  X_test = torch.tensor(X_test.float().numpy(), device=device)
  Y_test = torch.tensor(Y_test.cast(dtypes.int64).numpy(), device=device)

  if getenv("TORCHVIZ"): torch.cuda.memory._record_memory_history()
  model = Model().to(device)
  optimizer = optim.Adam(model.parameters(), 1e-3)

  loss_fn = nn.CrossEntropyLoss()
  #@torch.compile
  def step(samples):
    X,Y = X_train[samples], Y_train[samples]
    out = model(X)
    loss = loss_fn(out, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

  test_acc = float('nan')
  for i in (t:=trange(getenv("STEPS", 70))):
    samples = torch.randint(0, X_train.shape[0], (512,))  # putting this in JIT didn't work well
    loss = step(samples)
    if i%10 == 9: test_acc = ((model(X_test).argmax(axis=-1) == Y_test).sum() * 100 / X_test.shape[0]).item()
    t.set_description(f"loss: {loss.item():6.2f} test_accuracy: {test_acc:5.2f}%")

  # verify eval acc
  if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
    if test_acc >= target and test_acc != 100.0: print(colored(f"{test_acc=} >= {target}", "green"))
    else: raise ValueError(colored(f"{test_acc=} < {target}", "red"))
  if getenv("TORCHVIZ"):
    torch.cuda.memory._dump_snapshot(fp:=temp("torchviz.pkl", append_user=True))
    print(f"saved torch memory snapshot to {fp}, view in https://pytorch.org/memory_viz")
