from math import pi

class FirstOrderFilter:
  # first order low pass filter
  def __init__(self, x0, RC, dt, hz_mode=False):
    self.x = x0
    self.dt = dt
    self.hz = hz_mode
    self.update_alpha(RC)

  def update_alpha(self, val):
    if self.hz:
      omega = val
      RC_inv = (2 * pi * omega) 
      self.alpha = self.dt * RC_inv / (self.dt * RC_inv + 1)
    else:
      RC = val
      self.alpha = (self.dt) / (RC + self.dt)

  def update(self, x):
    self.x = (1. - self.alpha) * self.x + self.alpha * x
    return self.x

  def reset(self, x0):
    self.x = x0
