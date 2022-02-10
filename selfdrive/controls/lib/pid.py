import numpy as np
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
  def __init__(self, k_p, k_i, k_f=0., pos_limit=None, neg_limit=None, rate=100):
    self.sanitize_gains(k_p, k_i, k_f)

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

  def sanitize_gains(self, k_p, k_i, k_f):
    if isinstance(k_p, Number):
      k_p = [[0], [k_p]]
    if not k_p[0]:
      k_p[0] = [0.0]
    if not k_p[1]:
      k_p = [[0], [0]]

    if isinstance(k_i, Number):
      k_i = [[0], [k_i]]
    if not k_i[0]:
      k_i[0] = [0.0]
    if not k_i[1]:
      k_i = [[0], [0]]
    
    if not isinstance(k_f, Number):
      k_f = 0
    if not k_i[0]:
      k_f = 0

    self._k_p = k_p  # proportional gain
    self._k_i = k_i  # integral gain
    self.k_f = k_f   # feedforward gain

  def reset(self):
    self.p = 0.0
    self.i = 0.0
    self.f = 0.0
    self.control = 0

  def update(self, setpoint, measurement, speed=0.0, override=False, feedforward=0., deadzone=0., freeze_integrator=False):
    self.speed = speed

    error = float(apply_deadzone(setpoint - measurement, deadzone))
    self.p = error * self.k_p
    self.f = feedforward * self.k_f

    if override:
      self.i -= self.i_unwind_rate * float(np.sign(self.i))
    else:
      i = self.i + error * self.k_i * self.i_rate
      control = self.p + self.f + i

      # Update when changing i will move the control away from the limits
      # or when i will move towards the sign of the error
      if ((error >= 0 and (control <= self.pos_limit or i < 0.0)) or
          (error <= 0 and (control >= self.neg_limit or i > 0.0))) and \
         not freeze_integrator:
        self.i = i

    control = self.p + self.f + self.i

    self.control = clip(control, self.neg_limit, self.pos_limit)
    return self.control
