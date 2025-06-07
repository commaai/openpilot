import math
from typing import List
from tinygrad.nn.optim import Optimizer
from tinygrad.tensor import Tensor

class LR_Scheduler:
  def __init__(self, optimizer: Optimizer):
    self.optimizer = optimizer
    self.epoch_counter = Tensor([0], requires_grad=False, device=self.optimizer.device)

  def get_lr(self): pass

  def step(self) -> None:
    self.epoch_counter.assign(self.epoch_counter + 1).realize()
    self.optimizer.lr.assign(self.get_lr()).realize()

class LRSchedulerGroup:
  def __init__(self, *schedulers: LR_Scheduler): self.schedulers = schedulers
  def step(self) -> None:
    for s in self.schedulers: s.step()

class MultiStepLR(LR_Scheduler):
  def __init__(self, optimizer: Optimizer, milestones: List[int], gamma=0.1):
    super().__init__(optimizer)
    self.milestones = milestones
    self.gamma = gamma

  def get_lr(self) -> Tensor:
    if self.epoch_counter.numpy()[0] not in self.milestones:
      return self.optimizer.lr
    return self.optimizer.lr * self.gamma

class ReduceLROnPlateau(LR_Scheduler):
  def __init__(self, optimizer: Optimizer, mode="min", factor=0.1, patience=10, threshold=1e-4, threshold_mode="rel"):
    assert mode in ["min", "max"] and threshold_mode in ["rel", "abs"]
    super().__init__(optimizer)
    self.mode, self.factor, self.patience, self.threshold, self.threshold_mode = mode, factor, patience, threshold, threshold_mode
    self.best = float('inf') if mode == "min" else float('-inf')
    self.bad_epoch = 0

    if mode == "min": self.threshold *= -1

  def is_better(self, current: float) -> bool:
    dynamic_threshold = self.best*(1+self.threshold) if self.threshold_mode == "rel" else self.best+self.threshold
    if self.mode == "min":
      return current < dynamic_threshold
    return current > dynamic_threshold

  def step(self, current: float) -> None:
    self.epoch_counter.assign(self.epoch_counter + 1).realize()
    if self.is_better(current):
      self.bad_epoch = 0
      self.best = current
    else:
      self.bad_epoch += 1

    if self.bad_epoch > self.patience:
      self.optimizer.lr *= self.factor
      self.bad_epoch = 0

class CosineAnnealingLR(LR_Scheduler):
  def __init__(self, optimizer: Optimizer, T_max: int, eta_min=0):
    super().__init__(optimizer)
    self.T_max = T_max
    self.eta_min = eta_min
    self.eta_max = optimizer.lr.numpy()[0]

  def get_lr(self) -> Tensor:
    lr = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (1 + math.cos((self.epoch_counter.numpy()[0]/self.T_max) * math.pi))
    return Tensor([lr], device=self.optimizer.device, dtype=self.optimizer.lr.dtype)

class OneCycleLR(LR_Scheduler):
  def __init__(self, optimizer: Optimizer, max_lr: float, div_factor: float, final_div_factor: float, total_steps: int, pct_start: float,
               anneal_strategy: str = 'linear', cycle_momentum: bool = False):
    super().__init__(optimizer)
    self.initial_lr = max_lr / div_factor
    self.max_lr = max_lr
    self.min_lr = self.initial_lr / final_div_factor
    self.total_steps = total_steps
    self.pct_start = pct_start
    assert anneal_strategy == 'linear', 'only linear annealing supported'
    assert not cycle_momentum, 'cycle momentum not supported'
    self.optimizer.lr.assign(self.get_lr()).realize() # update the initial LR

  @staticmethod
  def _annealing_linear(start: float, end: float, pct: Tensor) -> Tensor: return (pct*(end-start)+start)

  def get_lr(self) -> Tensor:
    return (self.epoch_counter < self.total_steps*self.pct_start).where(
      self._annealing_linear(self.initial_lr, self.max_lr, self.epoch_counter/(self.total_steps*self.pct_start)),
      self._annealing_linear(self.max_lr, self.min_lr, (self.epoch_counter-(self.total_steps*self.pct_start))/(self.total_steps*(1-self.pct_start)))
    ).cast(self.optimizer.lr.dtype)
