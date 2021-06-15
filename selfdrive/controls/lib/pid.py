import numpy as np
from common.numpy_fast import clip, interp
from common.realtime import DT_CTRL
from math import exp

def apply_deadzone(error, deadzone):
  if error > deadzone:
    error -= deadzone
  elif error < - deadzone:
    error += deadzone
  else:
    error = 0.
  return error

class LPF():
  def __init__(self, omega): # 100 rad/s @ 100 hz ( e^-wT )
    self.alpha = exp(-omega * DT_CTRL)
    self.y = 0.0

  def filter(self, x):
    self.y = self.y * self.alpha + (1 - self.alpha) * x
    return self.y

class PIDController():
  def __init__(self, k_p, k_i, k_d, k_f=1., pos_limit=None, neg_limit=None, rate=100, sat_limit=0.8, convert=None):
    self._k_p = k_p  # proportional gain
    self._k_i = k_i  # integral gain
    self._k_d = k_d  # derivative gain
    self.k_f = k_f  # feedforward gain

    self.pos_limit = pos_limit
    self.neg_limit = neg_limit

    self.sat_count_rate = 1.0 / rate
    self.sat_limit = sat_limit

    self.convert = convert

    self.input = LPF(20)

    self.reset()

  @property
  def k_p(self):
    return interp(self.speed, self._k_p[0], self._k_p[1])

  @property
  def k_i(self):
    return interp(self.speed, self._k_i[0], self._k_i[1])

  @property
  def k_d(self):
    return interp(self.speed, self._k_d[0], self._k_d[1])

  def _check_saturation(self, control, check_saturation, error):
    saturated = (control < self.neg_limit) or (control > self.pos_limit)

    if saturated and check_saturation and abs(error) > 0.1:
      self.sat_count += self.sat_count_rate
    else:
      self.sat_count -= self.sat_count_rate

    self.sat_count = clip(self.sat_count, 0.0, 1.0)

    return self.sat_count > self.sat_limit

  def reset(self):
    self.u0, self.u1, self.u2 = 0.0, 0.0, 0.0
    self.e0, self.e1, self.e2 = 0.0, 0.0, 0.0

    self.p, self.p1, self.p2 = 0.0, 0.0, 0.0
    self.i, self.i1, self.i2 = 0.0, 0.0, 0.0
    self.d, self.d1, self.d2 = 0.0, 0.0, 0.0
    self.f = 0.0

    self.sat_count = 0.0
    self.saturated = False
    self.control = 0

  def update(self, setpoint, measurement, speed=0.0, check_saturation=True, override=False, feedforward=0., deadzone=0., freeze_integrator=False, output_steer_last=0.):
    self.speed = speed    

    #TODO: param
    _N = int(1. / DT_CTRL)
    _Ts = DT_CTRL
    
    Kp = self.k_p
    Ki = self.k_i
    Kd = self.k_d

    a0 = (1 + _N*_Ts)
    a1 = -(2 + _N*_Ts)
    a2 = 1
    
    b0 = Kp*a0 + Ki*_Ts*a0 + Kd*_N
    b1 = Kp*a1 - Ki*_Ts  - 2*Kd*_N
    b2 = Kp    +             Kd*_N
    
    #ku[0] = 1
    self.ku1 = a1 / a0
    self.ku2 = a2 / a0
    
    self.ke0 = b0 / a0
    self.ke1 = b1 / a0
    self.ke2 = b2 / a0

    self.e2 = self.e1
    self.e1 = self.e0
    self.u2 = self.u1
    self.u1 = self.u0
    
    self.e0 = float(apply_deadzone(setpoint - measurement, deadzone))
    self.e0 = self.input.filter(self.e0)

    self.u0 =  self.ke0*self.e0 + self.ke1*self.e1 + self.ke2*self.e2 - self.ku1*self.u1 - self.ku2*self.u2

    #logging only
    self.p2 = self.p1
    self.p1 = self.p
    self.i2 =  self.i1
    self.i1 = self.i
    self.d2 =  self.d1
    self.d1 = self.d

    self.p = (Kp*(    a0*self.e0 + a1*self.e1 + self.e2) / a0) -self.ku1*self.p1 - self.ku2*self.p2
    self.i = (Ki*_Ts*(a0*self.e0 -    self.e1)           / a0) -self.ku1*self.i1 - self.ku2*self.i2
    self.d = (Kd*_N*(    self.e0 -  2*self.e1 + self.e2) / a0) -self.ku1*self.d1 - self.ku2*self.d2
    #ylno gniggol

    self.f = self.k_f*feedforward
    self.f = clip(self.f, self.neg_limit, self.pos_limit)

    control = self.u0 + self.f
    self.saturated = self._check_saturation(control, check_saturation, self.e0)
    self.control = clip(control, self.neg_limit, self.pos_limit)

    if self.convert is not None:
      control = self.convert(control, speed=self.speed)

    return self.control

class PIController():
  def __init__(self, k_p, k_i, k_f=1., pos_limit=None, neg_limit=None, rate=100, sat_limit=0.8, convert=None):
    self._k_p = k_p  # proportional gain
    self._k_i = k_i  # integral gain
    self.k_f = k_f  # feedforward gain

    self.pos_limit = pos_limit
    self.neg_limit = neg_limit

    self.sat_count_rate = 1.0 / rate
    self.i_unwind_rate = 0.3 / rate
    self.i_rate = 1.0 / rate
    self.sat_limit = sat_limit
    self.convert = convert

    self.reset()

  @property
  def k_p(self):
    return interp(self.speed, self._k_p[0], self._k_p[1])

  @property
  def k_i(self):
    return interp(self.speed, self._k_i[0], self._k_i[1])

  def _check_saturation(self, control, check_saturation, error):
    saturated = (control <= self.neg_limit) or (control >= self.pos_limit)

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

  def update(self, setpoint, measurement, speed=0.0, check_saturation=True, override=False, feedforward=0., deadzone=0., freeze_integrator=False):
    self.speed = speed

    error = float(apply_deadzone(setpoint - measurement, deadzone))
    self.p = error * self.k_p
    self.f = feedforward * self.k_f

    if override:
      self.i -= self.i_unwind_rate * float(np.sign(self.i))
    else:
      i = self.i + error * self.k_i * self.i_rate
      control = self.p + self.f + i

      if self.convert is not None:
        control = self.convert(control, speed=self.speed)

      # Update when changing i will move the control away from the limits
      # or when i will move towards the sign of the error
      if ((error >= 0 and (control <= self.pos_limit or i < 0.0)) or
          (error <= 0 and (control >= self.neg_limit or i > 0.0))) and \
         not freeze_integrator:
        self.i = i

    control = self.p + self.f + self.i
    if self.convert is not None:
      control = self.convert(control, speed=self.speed)

    self.saturated = self._check_saturation(control, check_saturation, error)

    self.control = clip(control, self.neg_limit, self.pos_limit)
    return self.control
