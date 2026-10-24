import math
from tinygrad import dtypes, Tensor
from tinygrad.nn.optim import Optimizer

from extra.lr_scheduler import LR_Scheduler
from typing import Callable

# https://github.com/mlcommons/training/blob/e237206991d10449d9675d95606459a3cb6c21ad/image_classification/tensorflow2/lars_util.py
class PolynomialDecayWithWarmup(LR_Scheduler):
  def __init__(self, optimizer: Optimizer, initial_lr, end_lr, train_steps, warmup, power=2):
    super().__init__(optimizer)
    self.epoch_counter = self.epoch_counter.cast(dtypes.float32)
    assert train_steps > 0 and warmup > 0
    self.warmup = min(warmup, train_steps)
    self.initial_lr, self.end_lr, self.epochs, self.power = initial_lr, end_lr, train_steps, power

    # set lr for first warmup step
    self.optimizer.lr.assign(self.get_lr()).realize()

  def get_lr(self):
    # LR is 0 on the first step, matching the reference.
    warmup_lr = (self.epoch_counter * (1.0 / self.warmup)) * self.initial_lr
    x = (1 - (self.epoch_counter - self.warmup) / (self.epochs - self.warmup + 1))
    return (self.epoch_counter <= self.warmup).where(warmup_lr, (self.initial_lr - self.end_lr) * x ** self.power + self.end_lr).cast(self.optimizer.lr.dtype)

class CosineAnnealingLRWithWarmup(LR_Scheduler):
  def __init__(self, optimizer:Optimizer, base_lr, end_lr, warmup_steps:int, decay_steps:int):
    assert warmup_steps > 0 and decay_steps > 0
    super().__init__(optimizer)
    self.base_lr = base_lr
    self.end_lr = end_lr
    self.warmup_steps = warmup_steps
    self.decay_steps = decay_steps
    # set lr for first warmup step
    self.optimizer.lr.assign(self.get_lr()).realize()

  def get_lr(self):
    warmup_lr = ((self.epoch_counter+1) / self.warmup_steps) * self.base_lr
    decay_lr = self.end_lr + 0.5 * (self.base_lr-self.end_lr) * (1 + (((self.epoch_counter+1-self.warmup_steps)/self.decay_steps) * math.pi).cos())
    return (self.epoch_counter < self.warmup_steps).where(warmup_lr, decay_lr).cast(self.optimizer.lr.dtype)

# Reference: https://github.com/mlcommons/training/blob/64b14a9abc74e08779a175abca7d291f8c957632/stable_diffusion/ldm/lr_scheduler.py, Lines 36-97
class LambdaLinearScheduler:
  def __init__(self, warm_up_steps:int, f_min:float, f_max:float, f_start:float, cycle_lengths:int):
    self.lr_warm_up_steps, self.f_min, self.f_max, self.f_start, self.cycle_lengths = warm_up_steps, f_min, f_max, f_start, cycle_lengths

  def schedule(self, n:Tensor) -> Tensor:
    warm_up = (n < self.lr_warm_up_steps)
    f_warm_up = (self.f_max - self.f_start) / self.lr_warm_up_steps * n + self.f_start
    return warm_up.where(f_warm_up, self.f_min + (self.f_max - self.f_min) * (self.cycle_lengths - n) / (self.cycle_lengths))

# based on torch.optim.lr_scheduler.LambdaLR
class LambdaLR(LR_Scheduler):
  def __init__(self, optimizer:Optimizer, base_lr:Tensor, lr_lambda:Callable):
    super().__init__(optimizer)
    self.base_lr, self.lr_lambda = base_lr, lr_lambda
    self.step()

  def get_lr(self):
    return self.base_lr * self.lr_lambda(self.epoch_counter - 1)