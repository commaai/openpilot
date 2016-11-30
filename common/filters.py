"""Classes for filtering discrete time signals."""
import numpy as np


class FirstOrderLowpassFilter(object):
  def __init__(self, fc, dt, x1=0):
    self.kf = 2 * np.pi * fc * dt / (1 + 2 * np.pi * fc * dt)
    self.x1 = x1

  def __call__(self, x):
    self.x1 = (1 - self.kf) * self.x1 + self.kf * x

    # If previous or current is NaN, reset filter.
    if np.isnan(self.x1):
      self.x1 = x

    return self.x1
