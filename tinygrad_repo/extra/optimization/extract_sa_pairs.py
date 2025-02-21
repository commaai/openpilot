import sys, sqlite3, pickle, math
from collections import defaultdict
from tqdm import tqdm, trange
import numpy as np

# stuff needed to unpack a kernel
from tinygrad.ops import LazyOp, TernaryOps, BinaryOps, UnaryOps, ReduceOps, BufferOps, MemBuffer, ConstBuffer
from tinygrad.dtype import dtypes
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
from tinygrad.ops import Variable
inf, nan = float('inf'), float('nan')
from tinygrad.codegen.kernel import Opt, OptOps

# more stuff
from tinygrad.codegen.kernel import Kernel
from tinygrad.engine.search import actions
from extra.optimization.helpers import lin_to_feats
from extra.optimization.pretrain_valuenet import ValueNet
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save, safe_load, load_state_dict
import random
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv

def dataset_from_cache(fn):
  conn = sqlite3.connect(fn)
  cur = conn.cursor()
  cur.execute("SELECT * FROM time_linearizer")
  grouped = defaultdict(dict)
  for f in tqdm(cur.fetchall()): grouped[f[0]][f[1:-1]] = pickle.loads(f[-1])

  opts_to_outcome = {}

  for ast,sk in grouped.items():
    cnts = defaultdict(int)
    for sks,tm in sk.items():
      if sks[1] != 1: continue
      opts = eval(sks[0])
      cnts[(len(opts), sks[1])] += 1
      opts_to_outcome[(ast, tuple(opts))] = tm
    #print(cnts)

  S,A,V = [], [], []
  for ast,k in tqdm(opts_to_outcome):
    if len(k) == 0: continue
    old_tm = min(opts_to_outcome[(ast,k[:-1])])
    new_tm = min(opts_to_outcome[(ast,k)])
    if math.isinf(old_tm) or math.isinf(new_tm) or old_tm < 1e-9 or new_tm < 1e-9: continue
    try:
      lin = Kernel(eval(ast))
    except Exception:
      continue
    for opt in k[:-1]: lin.apply_opt(opt)
    act = k[-1]
    log_ratio = math.log(old_tm/new_tm)
    #print(f"ratio: {old_tm/new_tm:6.2f}x (log {log_ratio:5.2f}) from {str(act):50s} on {lin.colored_shape()}")
    S.append(lin_to_feats(lin, use_sts=True))
    A.append(actions.index(act))
    V.append([log_ratio])  # NOTE: i have written the bug many times with this having the wrong dim

  S, A, V = np.array(S), np.array(A), np.array(V, dtype=np.float32)
  X = np.zeros((S.shape[0], S.shape[1]+len(actions)), dtype=np.float32)
  X[:, :S.shape[1]] = S
  X[range(S.shape[0]), S.shape[1]+A] = 1.0
  return X, V

def log_likelihood(x:Tensor, mu:Tensor, log_sigma:Tensor):
  #print(x.shape, mu.shape, log_sigma.shape)
  #return (x-mu).abs() * (-log_sigma).exp() + log_sigma
  return (x-mu).square() * (-2*log_sigma).exp() / 2 + log_sigma

if __name__ == "__main__":
  if getenv("REGEN"):
    X,V = dataset_from_cache(sys.argv[1] if len(sys.argv) > 1 else "/tmp/tinygrad_cache")
    safe_save({"X": Tensor(X), "V": Tensor(V)}, "/tmp/dataset")
  else:
    ld = safe_load("/tmp/dataset")
    X,V = ld['X'].numpy(), ld['V'].numpy()

  print(X.shape, V.shape)
  order = list(range(X.shape[0]))
  random.shuffle(order)
  X, V = X[order], V[order]

  ratio = -512
  X_test, V_test = Tensor(X[ratio:]), Tensor(V[ratio:])
  X,V = X[:ratio], V[:ratio]
  print(X.shape, V.shape)

  #print(X[0], V[0])
  #print(X[-1], V[-1])
  print(X.shape)

  net = ValueNet(X.shape[1], 2)
  optim = Adam(get_parameters(net))

  def get_minibatch(X,Y,bs):
    xs, ys = [], []
    #random.seed(1337)
    for _ in range(bs):
      sel = random.randint(0, len(X)-1)
      xs.append(X[sel])
      ys.append(Y[sel])
    return Tensor(xs), Tensor(ys)

  Tensor.no_grad, Tensor.training = False, True
  losses = []
  test_losses = []
  test_loss = float('inf')
  for i in (t:=trange(2000)):
    x,y = get_minibatch(X,V,bs=256)
    out = net(x)
    #loss = (out-y).square().mean()
    loss = log_likelihood(y, out[:, 0:1], out[:, 1:2]).mean()
    optim.zero_grad()
    loss.backward()
    optim.step()
    t.set_description(f"loss {loss.numpy():7.2f}, test loss {test_loss:7.2f}")
    losses.append(loss.numpy().item())
    test_losses.append(test_loss)
    if i % 10: test_loss = (net(X_test)[:, 0:1]-V_test).square().mean().numpy().item()

  safe_save(get_state_dict(net), "/tmp/qnet.safetensors")

  import matplotlib.pyplot as plt
  plt.plot(losses[20:])
  plt.plot(test_losses[20:])
  plt.show()
