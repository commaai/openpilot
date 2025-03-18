import traceback
import time
from multiprocessing import Process, Queue
import numpy as np
from tinygrad.nn.state import get_parameters
from tinygrad.nn import optim
from tinygrad.helpers import getenv, trange
from tinygrad.tensor import Tensor
from extra.datasets import fetch_cifar
from extra.models.efficientnet import EfficientNet

class TinyConvNet:
  def __init__(self, classes=10):
    conv = 3
    inter_chan, out_chan = 8, 16   # for speed
    self.c1 = Tensor.uniform(inter_chan,3,conv,conv)
    self.c2 = Tensor.uniform(out_chan,inter_chan,conv,conv)
    self.l1 = Tensor.uniform(out_chan*6*6, classes)

  def forward(self, x):
    x = x.conv2d(self.c1).relu().max_pool2d()
    x = x.conv2d(self.c2).relu().max_pool2d()
    x = x.reshape(shape=[x.shape[0], -1])
    return x.dot(self.l1)

if __name__ == "__main__":
  IMAGENET = getenv("IMAGENET")
  classes = 1000 if IMAGENET else 10

  TINY = getenv("TINY")
  TRANSFER = getenv("TRANSFER")
  if TINY:
    model = TinyConvNet(classes)
  elif TRANSFER:
    model = EfficientNet(getenv("NUM", 0), classes, has_se=True)
    model.load_from_pretrained()
  else:
    model = EfficientNet(getenv("NUM", 0), classes, has_se=False)

  parameters = get_parameters(model)
  print("parameter count", len(parameters))
  optimizer = optim.Adam(parameters, lr=0.001)

  BS, steps = getenv("BS", 64 if TINY else 16), getenv("STEPS", 2048)
  print(f"training with batch size {BS} for {steps} steps")

  if IMAGENET:
    from extra.datasets.imagenet import fetch_batch
    def loader(q):
      while 1:
        try:
          q.put(fetch_batch(BS))
        except Exception:
          traceback.print_exc()
    q = Queue(16)
    for i in range(2):
      p = Process(target=loader, args=(q,))
      p.daemon = True
      p.start()
  else:
    X_train, Y_train, _, _ = fetch_cifar()
    X_train = X_train.reshape((-1, 3, 32, 32))
    Y_train = Y_train.reshape((-1,))

  with Tensor.train():
    for i in (t := trange(steps)):
      if IMAGENET:
        X, Y = q.get(True)
      else:
        samp = np.random.randint(0, X_train.shape[0], size=(BS))
        X, Y = X_train.numpy()[samp], Y_train.numpy()[samp]

      st = time.time()
      out = model.forward(Tensor(X.astype(np.float32), requires_grad=False))
      fp_time = (time.time()-st)*1000.0

      y = np.zeros((BS,classes), np.float32)
      y[range(y.shape[0]),Y] = -classes
      y = Tensor(y, requires_grad=False)
      loss = out.log_softmax().mul(y).mean()

      optimizer.zero_grad()

      st = time.time()
      loss.backward()
      bp_time = (time.time()-st)*1000.0

      st = time.time()
      optimizer.step()
      opt_time = (time.time()-st)*1000.0

      st = time.time()
      loss = loss.numpy()
      cat = out.argmax(axis=1).numpy()
      accuracy = (cat == Y).mean()
      finish_time = (time.time()-st)*1000.0

      # printing
      t.set_description("loss %.2f accuracy %.2f -- %.2f + %.2f + %.2f + %.2f = %.2f" %
        (loss, accuracy,
        fp_time, bp_time, opt_time, finish_time,
        fp_time + bp_time + opt_time + finish_time))

      del out, y, loss
