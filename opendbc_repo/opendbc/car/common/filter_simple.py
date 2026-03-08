class FirstOrderFilter:
  # first order filter
  def __init__(self, x0, rc, dt, initialized=True):
    self.x = x0
    self._dt = dt
    self.update_alpha(rc)
    self.initialized = initialized

  def update_dt(self, dt):
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


class HighPassFilter:
  # technically a band-pass filter
  def __init__(self, x0, rc1, rc2, dt, initialized=True):
    self.x = x0
    self._f1 = FirstOrderFilter(x0, rc1, dt, initialized)
    self._f2 = FirstOrderFilter(x0, rc2, dt, initialized)
    assert rc2 > rc1, "rc2 must be greater than rc1"

  def update_dt(self, dt):
    self._f1.update_dt(dt)
    self._f2.update_dt(dt)

  def update_alpha(self, rc1, rc2):
    self._f1.update_alpha(rc1)
    self._f2.update_alpha(rc2)

  def update(self, x):
    self.x = self._f1.update(x) - self._f2.update(x)
    return self.x
