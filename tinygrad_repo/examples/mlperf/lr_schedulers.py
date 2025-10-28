import math
from tinygrad import dtypes
from tinygrad.nn.optim import Optimizer

from extra.lr_scheduler import LR_Scheduler

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