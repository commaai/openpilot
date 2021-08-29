class FirstOrderFilter:
  # first order filter
  def __init__(self, x0, rc, dt):
    self.x = x0
    self.dt = dt
    self.update_alpha(rc)

  def update_alpha(self, rc):
    self.alpha = self.dt / (rc + self.dt)

  def update(self, x):
    self.x = (1. - self.alpha) * self.x + self.alpha * x
    return self.x
