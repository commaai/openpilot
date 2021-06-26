class FirstOrderFilter:
  # first order filter
  def __init__(self, x0, RC, dt):
    self.x = x0
    self.dt = dt
    self.update_alpha(RC)

  def update_alpha(self, RC):
    self.alpha = (self.dt / RC) / (1. + self.dt / RC)

  def update(self, x):
    self.x = (1. - self.alpha) * self.x + self.alpha * x
    return self.x
