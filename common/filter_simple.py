import numpy as np


class FirstOrderFilter:
  def __init__(self, x0, rc, dt, initialized=True):
    self.x = x0
    self.dt = dt
    self.update_alpha(rc)
    self.initialized = initialized

  def update_alpha(self, rc):
    self.alpha = self.dt / (rc + self.dt)

  def update(self, x):
    if self.initialized:
      self.x = (1. - self.alpha) * self.x + self.alpha * x
    else:
      self.initialized = True
      self.x = x
    return self.x


class BounceFilter(FirstOrderFilter):
  def __init__(self, x0, rc, dt, initialized=True):
    super().__init__(x0, rc, dt, initialized)
    # self.velocity = FirstOrderFilter(0.0, rc * 1.5, dt)
    self.velocity = FirstOrderFilter(0.0, 0.1, dt, initialized)

  def update(self, x):
    prev_x = self.x
    # new_x = super().update(x)
    self.velocity.update((x - prev_x) / 10)
    # self.velocity.x *= 0.9
    self.velocity.update(0.0)
    # self.velocity.update(0.0)
    self.x += self.velocity.x
    print(self.velocity.x)
    # self.x = np.interp(abs(vel), [0, 100], [prev_x, new_x])
    # print(vel)
    # new_x += vel
    # return new_x
    # new_x = super().update(x)
    # slow the initial move with velocity
    return self.x

  #
  # def update(self, x):
  #   prev_x = self.x
  #   new_x = super().update(x)
  #   vel = self.velocity.update(x - prev_x)
  #   print(vel)
  #   self.x = np.interp(abs(vel), [0, 100], [prev_x, new_x])
  #   # print(vel)
  #   # new_x += vel
  #   # return new_x
  #   # new_x = super().update(x)
  #   # slow the initial move with velocity
  #   return self.x
