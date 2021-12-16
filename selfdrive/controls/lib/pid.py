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
  def __init__(self, k_p, k_i, k_f=1., pos_limit=None, neg_limit=None, rate=100):
    self._k_p = k_p  # proportional gain
    self._k_i = k_i  # integral gain
    self.k_f = k_f   # feedforward gain
    if isinstance(self._k_p, Number):
      self._k_p = [[0], [self._k_p]]
    if isinstance(self._k_i, Number):
      self._k_i = [[0], [self._k_i]]

    self.pos_limit = pos_limit
    self.neg_limit = neg_limit

    self.i_unwind_rate = 0.3 / rate
    self.i_rate = 1.0 / rate

    self.reset()

  @property
  def k_p(self):
    return interp(self.speed, self._k_p[0], self._k_p[1])

  @property
  def k_i(self):
    return interp(self.speed, self._k_i[0], self._k_i[1])

  def reset(self):
    self.p = 0.0
    self.i = 0.0
    self.f = 0.0
    self.control = 0

  def update(self, setpoint, measurement, last_output, speed=0.0, feedforward=0., deadzone=0.):
    self.speed = speed

    i_bf = self.i_unwind_rate * (self.p + self.i + self.f - last_output)

    error = float(apply_deadzone(setpoint - measurement, deadzone))
    self.p = error * self.k_p
    self.i = self.i + error * self.k_i * self.i_rate - i_bf
    self.f = feedforward * self.k_f

    # ensure PI controller action is not clipped when ff is large
    self.f = clip(self.f, self.neg_limit, self.pos_limit)

    control = self.p + self.i + self.f

    self.control = clip(control, self.neg_limit, self.pos_limit)
    return self.control
