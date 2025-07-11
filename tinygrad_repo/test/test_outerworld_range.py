import unittest
from tinygrad import Tensor, nn, Variable, UOp, dtypes

# outerworld range should support three things
# 1. full optimizer steps (test_model_bound_range)
# 2. gradient accumulation (you want to end the range before running the optimizer)
# 3. stacked linear layers

class Model:
  def __init__(self): self.w = nn.Linear(64, 8, bias=False)
  def __call__(self, x:Tensor) -> Tensor: return self.w(x)

def get_model_and_opt():
  Tensor.manual_seed(1337)
  m = Model()
  opt = nn.optim.SGD(nn.state.get_parameters(m), lr=0.1, weight_decay=0)
  return m, opt

class TestOuterworldRange(unittest.TestCase):
  STEPS = 5
  BS = 20

  @classmethod
  def setUpClass(cls):
    Tensor.manual_seed(1338)
    # it learns to compute mean
    cls.X = Tensor.randn(cls.STEPS, cls.BS, 64).contiguous().realize()
    cls.Y = cls.X.reshape(cls.STEPS, cls.BS, 8, 8).mean(axis=-1).contiguous().realize()
    cls.losses = cls._get_model_baseline()

  def _compare(self, losses):
    for i,(x,y) in enumerate(zip(self.losses, losses)):
      self.assertAlmostEqual(x, y, places=5, msg=f"mismatch at {i} in {self.losses} vs {losses}")

  @classmethod
  @Tensor.train()
  def _get_model_baseline(self):
    m, opt = get_model_and_opt()
    losses = []
    for i in range(self.STEPS):
      opt.zero_grad()
      loss = (m(self.X[i]) - self.Y[i]).square().mean()
      loss.backward()
      loss.realize(*opt.schedule_step())
      losses.append(loss.item())
    return losses

  @Tensor.train()
  def test_model_grad_acc(self):
    m, opt = get_model_and_opt()
    losses = []
    for i in range(self.STEPS):
      opt.zero_grad()
      sub_batch_size = self.BS//2
      loss = 0
      scaling_factor = self.BS//sub_batch_size
      for j in range(0, self.BS, sub_batch_size):
        sub_loss = (m(self.X[i][j:j+sub_batch_size]) - self.Y[i][j:j+sub_batch_size]).square().mean() / scaling_factor
        sub_loss.backward()
        loss += sub_loss
      loss.realize(*opt.schedule_step())
      losses.append(loss.item())
    self._compare(losses)

  @Tensor.train()
  def test_model_variable(self):
    m, opt = get_model_and_opt()
    losses = []
    vi = Variable('i', 0, self.STEPS-1)
    for i in range(self.STEPS):
      vib = vi.bind(i)
      opt.zero_grad()
      loss = (m(self.X[vib]) - self.Y[vib]).square().mean()
      loss.backward()
      loss.realize(*opt.schedule_step())
      losses.append(loss.item())
    self._compare(losses)

  @Tensor.train()
  def test_model_scheduled(self):
    m, opt = get_model_and_opt()
    losses = []
    for i in range(self.STEPS):
      opt.zero_grad()
      loss = (m(self.X[i]) - self.Y[i]).square().mean()
      loss.backward()
      opt.schedule_step()
      losses.append(loss)
    self._compare(Tensor.stack(*losses).tolist())

  @Tensor.train()
  def test_model_scheduled_setitem(self):
    m, opt = get_model_and_opt()
    losses = Tensor.empty(self.STEPS)
    for i in range(self.STEPS):
      opt.zero_grad()
      loss = (m(self.X[i]) - self.Y[i]).square().mean()
      loss.backward()
      opt.schedule_step()
      # TODO: this shouldn't realize
      losses[i] = loss.requires_grad_(False)
    self._compare(losses.tolist())

  @unittest.expectedFailure
  @Tensor.train()
  def test_model_scheduled_variable(self):
    m, opt = get_model_and_opt()
    losses = []
    vi = Variable('i', 0, self.STEPS-1)
    for i in range(self.STEPS):
      vib = vi.bind(i)
      opt.zero_grad()
      loss = (m(self.X[vib]) - self.Y[vib]).square().mean()
      loss.backward()
      opt.schedule_step()
      losses.append(loss)
    self._compare(Tensor.stack(*losses).tolist())

  @unittest.expectedFailure
  @Tensor.train()
  def test_model_scheduled_variable_setitem(self):
    m, opt = get_model_and_opt()
    losses = Tensor.empty(self.STEPS)
    vi = Variable('i', 0, self.STEPS-1)
    for i in range(self.STEPS):
      vib = vi.bind(i)
      opt.zero_grad()
      loss = (m(self.X[vib]) - self.Y[vib]).square().mean()
      loss.backward()
      opt.schedule_step()
      losses[vib] = loss.requires_grad_(False)
    self._compare(losses.tolist())

  @unittest.expectedFailure
  @Tensor.train()
  def test_model_bound_range(self):
    m, opt = get_model_and_opt()
    # TODO: should ranges be unique so you don't have to pass in the -1?
    rng = UOp.range(dtypes.int, self.STEPS, -1)
    vib = Variable('i', 0, self.STEPS-1).bind(rng)
    loss = (m(self.X[vib]) - self.Y[vib]).square().mean()
    loss.backward()
    losses = Tensor.empty(self.STEPS)
    losses[vib] = loss
    losses.realize(*opt.schedule_step())

if __name__ == "__main__":
  unittest.main()
