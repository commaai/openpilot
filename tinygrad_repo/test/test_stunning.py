import unittest
from tinygrad import nn, Tensor, Variable, Context, Device
from tinygrad.helpers import trange

class Model:
  def __init__(self): self.layer = nn.Linear(28*28, 10)
  def __call__(self, x:Tensor) -> Tensor: return self.layer(x.flatten(1))

class TestStunning(unittest.TestCase):
  def test_indexing_variable(self):
    a = Tensor.arange(100*10).reshape(100, 10).contiguous()

    # index without variable
    nv = a[12].tolist()

    # index with variable
    vi = Variable('i', 0, a.shape[0]-1)
    wv = a[vi.bind(12)].tolist()

    self.assertListEqual(nv, wv)

  def test_indexing_two_bind(self):
    a = Tensor.arange(100*10).reshape(100, 10).contiguous()

    nv = a[12].cat(a[76]).tolist()

    vi = Variable('i', 0, a.shape[0]-1)
    with self.assertRaisesRegex(AssertionError, "different values for the same key"):
      wv = a[vi.bind(12)].cat(a[vi.bind(76)]).tolist()
      self.assertListEqual(nv, wv)

  @unittest.skipIf(Device.DEFAULT in {"WEBGPU", "NV", "CUDA"}, "Too many buffers / too slow")
  @unittest.skip("This is binding a Variable to two different values")
  def test_simple_train(self, steps=6, bs=4, adam=True):
    X_train, Y_train, _, _ = nn.datasets.mnist()
    model = Model()
    if adam: opt = nn.optim.Adam(nn.state.get_parameters(model))
    else: opt = nn.optim.SGD(nn.state.get_parameters(model), momentum=0.1)
    samples = Tensor.randint(steps, bs, high=X_train.shape[0])
    Y_train = Y_train.one_hot(10)
    X_samp, Y_samp = X_train[samples], Y_train[samples]
    vi = Variable('i', 0, samples.shape[0]-1)
    with Context(FUSE_ARANGE=1, SPLIT_REDUCEOP=0):
      with Tensor.train():
        losses = []
        for i in range(samples.shape[0]):
          vib = vi.bind(i)
          opt.zero_grad()
          pred = model(X_samp[vib].realize())
          loss = (pred - Y_samp[vib]).square().mean()
          losses.append(loss.backward())
          opt.schedule_step()
        #losses = Tensor.stack(*losses)

    # run
    for i in (t:=trange(len(losses))): t.set_description(f"loss: {losses[i].item():6.2f}")

if __name__ == '__main__':
  unittest.main()
