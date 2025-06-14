import os, sys, sqlite3, pickle, random
from tqdm import tqdm, trange
from copy import deepcopy
from tinygrad.nn import Linear
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save, safe_load, load_state_dict
from tinygrad.engine.search import actions
from extra.optimization.helpers import load_worlds, ast_str_to_lin, lin_to_feats, assert_same_lin
from tinygrad.codegen.kernel import Kernel
from tinygrad.helpers import getenv

# stuff needed to unpack a kernel
from tinygrad.uop.ops import LazyOp, TernaryOps, BinaryOps, UnaryOps, ReduceOps, BufferOps, MemBuffer, ConstBuffer
from tinygrad.dtype import dtypes
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
from tinygrad.uop.ops import Variable
inf, nan = float('inf'), float('nan')
from tinygrad.codegen.kernel import Opt, OptOps

INNER = 256
class PolicyNet:
  def __init__(self):
    self.l1 = Linear(1021,INNER)
    self.l2 = Linear(INNER,INNER)
    self.l3 = Linear(INNER,1+len(actions))
  def __call__(self, x):
    x = self.l1(x).relu()
    x = self.l2(x).relu().dropout(0.9)
    return self.l3(x).log_softmax()

def dataset_from_cache(fn):
  conn = sqlite3.connect(fn)
  cur = conn.cursor()
  cur.execute("SELECT * FROM beam_search")
  X,A = [], []
  for f in tqdm(cur.fetchall()):
    Xs,As = [], []
    try:
      lin = Kernel(eval(f[0]))
      opts = pickle.loads(f[-1])
      for o in opts:
        Xs.append(lin_to_feats(lin, use_sts=True))
        As.append(actions.index(o))
        lin.apply_opt(o)
      Xs.append(lin_to_feats(lin, use_sts=True))
      As.append(0)
    except Exception:
      pass
    X += Xs
    A += As
  return X,A

if __name__ == "__main__":
  if getenv("REGEN"):
    X,V = dataset_from_cache(sys.argv[1] if len(sys.argv) > 1 else "/tmp/tinygrad_cache")
    safe_save({"X": Tensor(X), "V": Tensor(V)}, "/tmp/dataset_policy")
  else:
    ld = safe_load("/tmp/dataset_policy")
    X,V = ld['X'].numpy(), ld['V'].numpy()

  print(X.shape, V.shape)
  order = list(range(X.shape[0]))
  random.shuffle(order)
  X, V = X[order], V[order]

  ratio = -256
  X_test, V_test = Tensor(X[ratio:]), Tensor(V[ratio:])
  X,V = X[:ratio], V[:ratio]
  print(X.shape, V.shape)

  net = PolicyNet()
  #if os.path.isfile("/tmp/policynet.safetensors"): load_state_dict(net, safe_load("/tmp/policynet.safetensors"))
  optim = Adam(get_parameters(net))

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
  test_accuracy = 0
  test_loss = float('inf')
  for i in (t:=trange(500)):
    x,y = get_minibatch(X,V,bs=256)
    out = net(x)
    loss = out.sparse_categorical_crossentropy(y)
    optim.zero_grad()
    loss.backward()
    optim.step()
    cat = out.argmax(axis=-1)
    accuracy = (cat == y).mean()
    t.set_description(f"loss {loss.numpy():7.2f} accuracy {accuracy.numpy()*100:7.2f}%, test loss {test_loss:7.2f} test accuracy {test_accuracy*100:7.2f}%")

    losses.append(loss.numpy().item())
    test_losses.append(test_loss)
    if i % 10:
      out = net(X_test)
      test_loss = out.sparse_categorical_crossentropy(V_test).square().mean().numpy().item()
      cat = out.argmax(axis=-1)
      test_accuracy = (cat == y).mean().numpy()

  safe_save(get_state_dict(net), "/tmp/policynet.safetensors")

  import matplotlib.pyplot as plt
  plt.plot(losses[10:])
  plt.plot(test_losses[10:])
  plt.show()
