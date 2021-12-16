from numbers import Number

from common.numpy_fast import clip, interp

def apply_deadzone(error, deadzone):
  if error > deadzone:
    error -= deadzone
  elif error < - deadzone:
    error += deadzone
  else:
    error = 0.
  return error

class PIController():
  def __init__(self, k_p, k_i, k_f=1., pos_limit=None, neg_limit=None, rate=100, sat_limit=0.8):
    self._k_p = k_p  # proportional gain
    self._k_i = k_i  # integral gain
    self.k_f = k_f   # feedforward gain
    if isinstance(self._k_p, Number):
      self._k_p = [[0], [self._k_p]]
    if isinstance(self._k_i, Number):
      self._k_i = [[0], [self._k_i]]

    self.pos_limit = pos_limit
    self.neg_limit = neg_limit

    self.sat_count_rate = 1.0 / rate
    self.i_unwind_rate = 1.0 / rate
    self.i_rate = 1.0 / rate
    self.sat_limit = sat_limit

    self.reset()

  @property
  def k_p(self):
    return interp(self.speed, self._k_p[0], self._k_p[1])

  @property
  def k_i(self):
    return interp(self.speed, self._k_i[0], self._k_i[1])

  def _check_saturation(self, control, check_saturation, error):
    saturated = (control < self.neg_limit) or (control > self.pos_limit)

    if saturated and check_saturation and abs(error) > 0.1:
      self.sat_count += self.sat_count_rate
    else:
      self.sat_count -= self.sat_count_rate

    self.sat_count = clip(self.sat_count, 0.0, 1.0)

    return self.sat_count > self.sat_limit

  def reset(self):
    self.p = 0.0
    self.i = 0.0
    self.f = 0.0
    self.sat_count = 0.0
    self.saturated = False
    self.control = 0

  def update(self, setpoint, measurement, last_output, speed=0.0, check_saturation=True, feedforward=0., deadzone=0.):
    self.speed = speed

    i_bf = self.i_unwind_rate * (self.p + self.i + self.f - last_output)

    error = float(apply_deadzone(setpoint - measurement, deadzone))
    self.p = error * self.k_p
    self.i = self.i + error * self.k_i * self.i_rate - i_bf
    self.f = feedforward * self.k_f

    # ensure PI controller action is not clipped when ff is large
    self.f = clip(self.f, self.neg_limit, self.pos_limit)

    control = self.p + self.i + self.f
    self.saturated = self._check_saturation(control, check_saturation, error)

    self.control = clip(control, self.neg_limit, self.pos_limit)
    return self.control
