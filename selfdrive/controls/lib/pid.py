import numpy as np
from common.numpy_fast import clip, interp
import numbers

def apply_deadzone(error, deadzone):
  if error > deadzone:
    error -= deadzone
  elif error < - deadzone:
    error += deadzone
  else:
    error = 0.
  return error

class PIController(object):
  def __init__(self, k_p, k_i, k_f=0., pos_limit=None, neg_limit=None, rate=100, sat_limit=0.8, convert=None):
    self._k_p = k_p # proportional gain
    self._k_i = k_i # integrale gain
    self.k_f = k_f  # feedforward gain

    self.pos_limit = pos_limit
    self.neg_limit = neg_limit

    self.sat_count_rate = 1.0 / rate
    self.i_unwind_rate = 0.3 / rate
    self.i_rate = 1.0 / rate
    self.sat_limit = sat_limit
    self.jerk_factor = 0.0
    self.convert = convert

    self.reset()

  @property
  def k_p(self):
    if isinstance(self._k_p, numbers.Number):
      kp = self._k_p
    else:
      kp = interp(self.speed, self._k_p[0], self._k_p[1])

    return (1.0 + self.jerk_factor) * kp

  @property
  def k_i(self):
    if isinstance(self._k_i, numbers.Number):
      ki = self._k_i
    else:
      ki = interp(self.speed, self._k_i[0], self._k_i[1])

    return (1.0 + self.jerk_factor) * ki

  def _check_saturation(self, control, override, error):
    saturated = (control < self.neg_limit) or (control > self.pos_limit)

    if saturated and not override and abs(error) > 0.1:
      self.sat_count += self.sat_count_rate
    else:
      self.sat_count -= self.sat_count_rate

    self.sat_count = clip(self.sat_count, 0.0, 1.0)

    return self.sat_count > self.sat_limit

  def reset(self):
    self.p = 0.0
    self.i = 0.0
    self.sat_count = 0.0
    self.saturated = False
    self.control = 0

  def update(self, setpoint, measurement, speed=0.0, check_saturation=True, jerk_factor=0.0, override=False, feedforward=0., deadzone=0.):
    self.speed = speed
    self.jerk_factor = jerk_factor

    error = float(apply_deadzone(setpoint - measurement, deadzone))
    self.p = error * self.k_p
    f = feedforward * self.k_f

    if override:
      self.i -= self.i_unwind_rate * float(np.sign(self.i))
    else:
      i = self.i + error * self.k_i * self.i_rate
      control = self.p + f + i

      if self.convert is not None:
        control = self.convert(control, speed=self.speed)

      # Update when changing i will move the control away from the limits
      # or when i will move towards the sign of the error
      if (error >= 0 and (control <= self.pos_limit or i < 0.0)) or \
         (error <= 0 and (control >= self.neg_limit or i > 0.0)):
        self.i = i

    control = self.p + f + self.i
    if self.convert is not None:
      control = self.convert(control, speed=self.speed)

    if check_saturation:
      self.saturated = self._check_saturation(control, override, error)
    else:
      self.saturated = False

    self.control = clip(control, self.neg_limit, self.pos_limit)
    return self.control
