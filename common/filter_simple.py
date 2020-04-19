class FirstOrderFilter():
  # first order filter
  def __init__(self, x0, ts, dt):
    self.k = (dt / ts) / (1. + dt / ts)
    self.x = x0

  def update(self, x):
    self.x = (1. - self.k) * self.x + self.k * x
    return self.x
