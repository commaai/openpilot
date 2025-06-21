from tinygrad.codegen.kernel import Kernel
from tqdm import tqdm, trange
import math
import random
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save, safe_load, load_state_dict

# stuff needed to unpack a kernel
from tinygrad.uop.ops import LazyOp, TernaryOps, BinaryOps, UnaryOps, ReduceOps, BufferOps, MemBuffer, ConstBuffer
from tinygrad.dtype import dtypes
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
from tinygrad.uop.ops import Variable
inf, nan = float('inf'), float('nan')
from tinygrad.codegen.kernel import Opt, OptOps

from extra.optimization.helpers import lin_to_feats, MAX_DIMS

# NOTE: this is not real value of the state, it's just a prediction of the runtime
INNER = 512
class ValueNet:
  def __init__(self, feats=240, out=1):
    self.l1 = Linear(feats,INNER)
    self.l2 = Linear(INNER,INNER)
    self.l3 = Linear(INNER,INNER)
    self.l4 = Linear(INNER,out)
  def __call__(self, x):
    x = self.l1(x).relu()
    x = self.l2(x).relu()
    x = self.l3(x).relu().dropout(0.8)
    return self.l4(x)

if __name__ == "__main__":
  net = ValueNet()
  optim = Adam(get_parameters(net))

  TEST_SIZE = 256

  dset = open("/tmp/logtm").read().strip().split("\n")
  random.seed(1337)
  random.shuffle(dset)

  X,Y = [], []
  for i,x in enumerate(tqdm(dset)):
    ast, opts, tms = eval(x)
    lin = Kernel(ast)
    for o in opts: lin.apply_opt(o)
    if lin.shape_len >= MAX_DIMS: continue
    if min(tms) == float('inf'): continue
    X.append(lin_to_feats(lin))
    Y.append([math.log(min(tms))])
  print(f"got {len(X)} samples")

  X_test,Y_test = Tensor(X[-TEST_SIZE:]), Tensor(Y[-TEST_SIZE:])
  X,Y = X[:-TEST_SIZE], Y[:-TEST_SIZE]

  def get_minibatch(X,Y,bs):
    xs, ys = [], []
    for _ in range(bs):
      sel = random.randint(0, len(X)-1)
      xs.append(X[sel])
      ys.append(Y[sel])
    return Tensor(xs), Tensor(ys)

  Tensor.training = True
  losses = []
  test_losses = []
  test_loss = float('inf')
  for i in (t:=trange(2000)):
    x,y = get_minibatch(X,Y,bs=256)
    out = net(x)
    loss = (out-y).square().mean()
    optim.zero_grad()
    loss.backward()
    optim.step()
    t.set_description(f"loss {loss.numpy():7.2f}, test loss {test_loss:7.2f}")
    losses.append(loss.numpy().item())
    test_losses.append(test_loss)
    if i % 10: test_loss = (net(X_test)-Y_test).square().mean().numpy().item()

  safe_save(get_state_dict(net), "/tmp/valuenet.safetensors")

  import matplotlib.pyplot as plt
  plt.plot(losses[200:])
  plt.plot(test_losses[200:])
  plt.show()
