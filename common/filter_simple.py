class FirstOrderFilter:
  def __init__(self, x0, rc, dt, initialized=True):
    self.x = x0
    self._dt = dt
    self.update_alpha(rc)
    self.initialized = initialized

  def update_dt(self, dt):
    if dt == self._dt:
      return
    self._dt = dt
    self.update_alpha(self._rc)

  def update_alpha(self, rc):
    self._rc = rc
    self._alpha = self._dt / (self._rc + self._dt)

  def update(self, x):
    if self.initialized:
      self.x = (1. - self._alpha) * self.x + self._alpha * x
    else:
      self.initialized = True
      self.x = x
    return self.x
