#!/usr/bin/env python3
import random
import unittest
import sys

from common.numpy_fast import clip, interp
from common.filter_simple import FirstOrderFilter
from selfdrive.controls.lib.pid import PIController

gain_range = 10 # (1 / range -> 1 ->  range   )
ratio = 4 #   I = (P / ratio -> P -> P * ratio)

num_divs = 8# (2*n + 1)**2 graphs
_p = [[0, num_divs, 2*num_divs],[1/gain_range, 1, gain_range]]

rate_limit = 1
sec = 15

if __name__ == "__main__":
  plot_debug = False
  debug_noise = False
  debug_override = False
  debug_ramp = False
  if len(sys.argv) > 1:
    if "noise" in sys.argv:
      debug_noise = True
    if "override" in sys.argv:
      debug_override = True
    if "ramp" in sys.argv:
      debug_ramp = True
    if "plot" in sys.argv:
      plot_debug = True
    if "debug_only" in sys.argv:
      plot_all = False
    else:
      plot_all = True
    if (plot_debug and not (debug_noise or debug_override or debug_ramp)) or "debug" in sys.argv:
      debug_noise = True
      debug_override = True
      debug_ramp = True
    if (plot_debug):
      num_divs = 4 # (2*n + 1)**2 graphs (81 graphs)
      _p = [[0, num_divs, 2*num_divs],[1/gain_range, 1, gain_range]]
    #remove args so unittest is not freaked out
    while len(sys.argv) > 1:
      sys.argv.pop()
  
if plot_debug and (debug_noise or debug_override or debug_ramp):
  import matplotlib.pyplot as plt


class TestPID(unittest.TestCase):
  def test_error_noise(self):
    #TODO New Tests
    #TODO check that the average deviation from the noiseless reference is driven to 0
    #TODO check that the saturated ramp up condition(s?) are not degraded. 

    self.assertFalse((debug_ramp or debug_override) and not debug_noise, msg="Test skipped for debugging")
    for i in range(0, 2*num_divs + 1):
      for j in range (0, 2*num_divs + 1):
        kp = interp(i, _p[0], _p[1])
        _i = [[0, num_divs, 2*num_divs], [kp/ratio, kp, ratio*kp]]
        ki = interp(j, _i[0], _i[1])
        pid_s = PI_Simple(kp, ki, pos_limit=100, neg_limit=-100)
        pid_c = PI_Classic(kp, ki, pos_limit=100, neg_limit=-100)
        pid_t = PIDTrapazoid(([0, 1], [kp, kp]), ([0, 1], [ki, ki]), ([0, 1], [0, 0]), k_f=0, pos_limit=100, neg_limit=-100)
        pid_n = PIController(([0, 1], [kp, kp]), ([0, 1], [ki, ki]), k_f=0, pos_limit=100, neg_limit=-100)
        output_s = FirstOrderFilter(0, 1, .01)
        output_c = FirstOrderFilter(0, 1, .01)
        output_t = FirstOrderFilter(0, 1, .01)
        output_n = FirstOrderFilter(0, 1, .01)
        last_pid_s_control = 0
        last_pid_c_control = 0
        last_pid_t_control = 0
        last_pid_n_control = 0
        target = 0
        noise = 0
        if plot_debug and debug_noise:
          x = []
          y = []
          y2 = []
          y3 = []
          y4 = []
          y5 = []
        for t in range(0, int(sec*100)):
          if plot_debug and debug_noise:
            x.append(t)
            y.append(output_s.x)
            y2.append(output_c.x)
            y3.append(output_t.x)
            y4.append(output_n.x)
            y5.append(target+noise)
          noise = 0
          target = 0
          if (t > 10):
            target = 50

          if target != 0:
            noise = 10 * (rate_limit / kp) * ((random.randint(0, 200) - 100) / 100)

          pid_s_control_raw = pid_s.update(target+noise, output_s.x)
          pid_s_control = clip(pid_s_control_raw, last_pid_s_control - rate_limit, last_pid_s_control + rate_limit)

          pid_c_control_raw = pid_c.update(target+noise, output_c.x)
          pid_c_control = clip(pid_c_control_raw, last_pid_c_control - rate_limit, last_pid_c_control + rate_limit)

          pid_t_control_raw = pid_t.update(target+noise, output_t.x, last_output=last_pid_t_control)
          pid_t_control = clip(pid_t_control_raw, last_pid_t_control - rate_limit, last_pid_t_control + rate_limit)

          pid_n_control_raw = pid_n.update(target+noise, output_n.x, last_output=last_pid_n_control)
          pid_n_control = clip(pid_n_control_raw, last_pid_n_control - rate_limit, last_pid_n_control + rate_limit)
          
          last_pid_s_control = pid_s_control
          last_pid_c_control = pid_c_control
          last_pid_t_control = pid_t_control
          last_pid_n_control = pid_n_control
          output_s.update(pid_s_control)
          output_c.update(pid_c_control)
          output_t.update(pid_t_control)
          output_n.update(pid_n_control)

        if plot_debug and debug_noise and plot_all:
          plt.plot(x, y5, 'k', label="Target")
          plt.plot(x, y, label="Naive")
          plt.plot(x, y2, label="Classic")
          plt.plot(x, y3, label="Recalculation")
          plt.plot(x, y4, label="pid.py")
          plt.title(f"Noise Test: P = {kp:5.3}, I = {ki:5.3}")
          plt.legend()
          plt.show()
  
  def test_override(self):
    #TODO New Tests
    #TODO check that output is stable?
    #TODO check that overshoot is reduced?
    #TODO check that integrator motion is reduced / reversed during override

    self.assertFalse((debug_ramp or debug_noise) and not debug_override, msg="Test skipped for debugging")
    for i in range(0, 2*num_divs + 1):
      for j in range (0, 2*num_divs + 1):
        kp = interp(i, _p[0], _p[1])
        _i = [[0, num_divs, 2*num_divs], [kp/ratio, kp, ratio*kp]]
        ki = interp(j, _i[0], _i[1])
        pid_s = PI_Simple(kp, ki, pos_limit=100, neg_limit=-100)
        pid_c = PI_Classic(kp, ki, pos_limit=100, neg_limit=-100)
        pid_t = PIDTrapazoid(([0, 1], [kp, kp]), ([0, 1], [ki, ki]), ([0, 1], [0, 0]), k_f=0, pos_limit=100, neg_limit=-100)
        pid_n = PIController(([0, 1], [kp, kp]), ([0, 1], [ki, ki]), k_f=0, pos_limit=100, neg_limit=-100)
        output_s = FirstOrderFilter(0, 1, .01)
        output_c = FirstOrderFilter(0, 1, .01)
        output_t = FirstOrderFilter(0, 1, .01)
        output_n = FirstOrderFilter(0, 1, .01)
        last_pid_s_control = 0
        last_pid_c_control = 0
        last_pid_t_control = 0
        last_pid_n_control = 0
        target = 0
        if plot_debug and debug_override:
          x = []
          y = []
          y2 = []
          y3 = []
          y4 = []
          y5 = []
        for t in range(0, int(sec*100)):
          if plot_debug and debug_override:
            x.append(t)
            y.append(output_s.x)
            y2.append(output_c.x)
            y3.append(output_t.x)
            y4.append(output_n.x)
            y5.append(target)
          target = 0
          if (t > 10):
            target = 50

          pid_s_control_raw = pid_s.update(target, output_s.x)
          pid_s_control = clip(pid_s_control_raw, last_pid_s_control - rate_limit, last_pid_s_control + rate_limit)

          pid_c_control_raw = pid_c.update(target, output_c.x)
          pid_c_control = clip(pid_c_control_raw, last_pid_c_control - rate_limit, last_pid_c_control + rate_limit)

          pid_t_control_raw = pid_t.update(target, output_t.x, last_output=last_pid_t_control)
          pid_t_control = clip(pid_t_control_raw, last_pid_t_control - rate_limit, last_pid_t_control + rate_limit)

          pid_n_control_raw = pid_n.update(target, output_n.x, last_output=last_pid_n_control)
          pid_n_control = clip(pid_n_control_raw, last_pid_n_control - rate_limit, last_pid_n_control + rate_limit)

                    # override / hold output of controller at 0 
          if t > 500:
            last_pid_s_control = pid_s_control
            last_pid_c_control = pid_c_control
            last_pid_t_control = pid_t_control
            last_pid_n_control = pid_n_control
            output_s.update(pid_s_control)
            output_c.update(pid_c_control)
            output_t.update(pid_t_control)
            output_n.update(pid_n_control)
          else: 
            last_pid_s_control = 0
            last_pid_c_control = 0
            last_pid_t_control = 0
            last_pid_n_control = 0
            

        if plot_debug and debug_override and plot_all:
          plt.plot(x, y5, 'k', label="Target")
          plt.plot(x, y, label="Naive")
          plt.plot(x, y2, label="Classic")
          plt.plot(x, y3, label="Recalculation")
          plt.plot(x, y4, label="pid.py")
          plt.title(f"Override Test: P = {kp:5.3}, I = {ki:5.3}")
          plt.legend()
          plt.show()

  def test_ramp_up(self):
    #TODO New Tests
    #TODO check that output is stable?
    #TODO check that overshoot is reduced?
    #TODO check that integrator motion is restricted or reversed?

    self.assertFalse((debug_override or debug_noise) and not debug_ramp, msg="Test skipped for debugging")
    for i in range(0, 2*num_divs + 1):
      for j in range (0, 2*num_divs + 1):
        kp = interp(i, _p[0], _p[1])
        _i = [[0, num_divs, 2*num_divs], [kp/ratio, kp, ratio*kp]]
        ki = interp(j, _i[0], _i[1])
        pid_s = PI_Simple(kp, ki, pos_limit=100, neg_limit=-100)
        pid_c = PI_Classic(kp, ki, pos_limit=100, neg_limit=-100)
        pid_t = PIDTrapazoid(([0, 1], [kp, kp]), ([0, 1], [ki, ki]), ([0, 1], [0, 0]), k_f=0, pos_limit=100, neg_limit=-100)
        pid_n = PIController(([0, 1], [kp, kp]), ([0, 1], [ki, ki]), k_f=0, pos_limit=100, neg_limit=-100)
        output_s = FirstOrderFilter(0, 1, .01)
        output_c = FirstOrderFilter(0, 1, .01)
        output_t = FirstOrderFilter(0, 1, .01)
        output_n = FirstOrderFilter(0, 1, .01)
        last_pid_s_control = 0
        last_pid_c_control = 0
        last_pid_t_control = 0
        last_pid_n_control = 0
        target = 0
        if plot_debug and debug_ramp:
          x = []
          y = []
          y2 = []
          y3 = []
          y4 = []
          y5 = []
        for t in range(0, int(sec*100)):
          if plot_debug and debug_ramp:
            x.append(t)
            y.append(output_s.x)
            y2.append(output_c.x)
            y3.append(output_t.x)
            y4.append(output_n.x)
            y5.append(target)
          target = 0
          if (t > 10):
            target = 50

          pid_s_control_raw = pid_s.update(target, output_s.x)
          pid_s_control = clip(pid_s_control_raw, last_pid_s_control - rate_limit, last_pid_s_control + rate_limit)

          pid_c_control_raw = pid_c.update(target, output_c.x)
          pid_c_control = clip(pid_c_control_raw, last_pid_c_control - rate_limit, last_pid_c_control + rate_limit)

          pid_t_control_raw = pid_t.update(target, output_t.x, last_output=last_pid_t_control)
          pid_t_control = clip(pid_t_control_raw, last_pid_t_control - rate_limit, last_pid_t_control + rate_limit)

          pid_n_control_raw = pid_n.update(target, output_n.x, last_output=last_pid_n_control)
          pid_n_control = clip(pid_n_control_raw, last_pid_n_control - rate_limit, last_pid_n_control + rate_limit)

          last_pid_s_control = pid_s_control
          last_pid_c_control = pid_c_control
          last_pid_t_control = pid_t_control
          last_pid_n_control = pid_n_control
          output_s.update(pid_s_control)
          output_c.update(pid_c_control)
          output_t.update(pid_t_control)
          output_n.update(pid_n_control)
            

        if plot_debug and debug_ramp and plot_all:
          plt.plot(x, y5, 'k', label="Target")
          plt.plot(x, y, label="Naive")
          plt.plot(x, y2, label="Classic")
          plt.plot(x, y3, label="Recalculation")
          plt.plot(x, y4, label="pid.py")
          plt.title(f"Slew Rate Limit Test: P = {kp:5.3}, I = {ki:5.3}")
          plt.legend()
          plt.show()

class PI_Simple():
  def __init__(self, k_p, k_i, pos_limit=None, neg_limit=None, rate=100):
    self.kp = k_p
    self.ki = k_i
    self.pos_limit = pos_limit
    self.neg_limit = neg_limit
    self.i_rate = 1.0 / rate

    self.reset()

  def reset(self):
    self.i = 0.0

  def update(self, setpoint, measurement,):
    error = float(setpoint - measurement)
    self.p = error * self.kp
    self.i = self.i + error * self.ki * self.i_rate
    control = self.p + self.i

    control = clip(control, self.neg_limit, self.pos_limit)
    return control


class PI_Classic():
  def __init__(self, k_p, k_i, pos_limit=None, neg_limit=None, rate=100, sat_limit=0.8):
    self.kp = k_p
    self.ki = k_i

    self.pos_limit = pos_limit
    self.neg_limit = neg_limit

    self.sat_count_rate = 1.0 / rate
    self.i_unwind_rate = 0.3 / rate
    self.i_rate = 1.0 / rate
    self.sat_limit = sat_limit

    self.reset()

  def reset(self):
    self.p = 0.0
    self.i = 0.0

  def update(self, setpoint, measurement, override=False, freeze_integrator=False):

    error = float(setpoint - measurement)
    self.p = error * self.kp

    if override:
      self.i *= (1 - self.i_unwind_rate)
    else:
      i = self.i + error * self.ki * self.i_rate
      control = self.p + i

      # Update when changing i will move the control away from the limits
      # or when i will move towards the sign of the error
      if ((error >= 0 and (control <= self.pos_limit or i < 0.0)) or
          (error <= 0 and (control >= self.neg_limit or i > 0.0))) and \
         not freeze_integrator:
        self.i = i

    control = self.p + self.i

    control = clip(control, self.neg_limit, self.pos_limit)
    return control

def apply_deadzone(error, deadzone):
  if error > deadzone:
    error -= deadzone
  elif error < - deadzone:
    error += deadzone
  else:
    error = 0.
  return error

class PIDTrapazoid():
  def __init__(self, k_p, k_i, k_d, k_f=0., pos_limit=None, neg_limit=None, rate=100, sat_limit=0.4, convert=None):
    self._k_p = k_p  # proportional gain
    self._k_i = k_i  # integral gain
    self._k_d = k_d  # derivative gain
    self.k_f = k_f  # feedforward gain

    self.rate = rate

    self.pos_limit = pos_limit
    self.neg_limit = neg_limit
    self.sat_limit = sat_limit * self.rate

    self.convert = convert

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

  def _check_saturation(self, control, check_saturation, saturation):
    thresh = 0.05
    contr = control

    if (abs(saturation) > thresh) and check_saturation:
      self.sat_count += 1
    else:
      self.sat_count -= 1

    self.sat_count = clip(self.sat_count, 0.0, self.sat_limit * 1.25)

    contr *= 1.0 + thresh

    self.saturated = self.saturated or (contr <= self.neg_limit) or (contr >= self.pos_limit)
    self.saturated = self.saturated and (self.sat_count > self.sat_limit)

  def reset(self):
    self.e0, self.e1, self.e2 = 0.0, 0.0, 0.0
    self.u0, self.u1, self.u2 = 0.0, 0.0, 0.0

    self.p, self.p1, self.p2 = 0.0, 0.0, 0.0
    self.i, self.i1, self.i2 = 0.0, 0.0, 0.0
    self.d, self.d1, self.d2 = 0.0, 0.0, 0.0
    self.f = 0.0
    self.speed = 0.0
    self.sat_count = 0.0
    self.saturated = False

  def update(self, setpoint, measurement, last_output, speed=0.0, check_saturation=True, override=False, feedforward=0., deadzone=0., freeze_integrator=False):
    self.speed = speed
    
    _N = 25
    _Ts = 1 / self.rate
    
    # calculate coefficients
    Kp, Ki, Kd = self.k_p, self.k_i, self.k_d
    a0, a1, a2 = (2*_Ts), 0, -(2*_Ts)
    b0, b1, b2 = (2*Kp*_Ts + Ki*_Ts*_Ts + 4*Kd), (2*Ki*_Ts*_Ts - 8*Kd), (Ki*_Ts*_Ts - 2*Kp*_Ts + 4*Kd)
    ke0, ke1, ke2 = b0/a0, b1/a0, b2/a0
    ku1, ku2 = a1/a0, a2/a0

    #recalculate the last error from corrected u0
    self.u0 = last_output
    self.e0 = (self.u0 + ku1*self.u1 + ku2*self.u2 - ke1*self.e1 - ke2*self.e2) / ke0 

    #calculate the last logging partials (self.u0 is already = last_output)
    self.p2, self.p1 = self.p1, self.p
    self.i2, self.i1 = self.i1, self.i
    self.d2, self.d1 = self.d1, self.d
    self.p = (Kp*(    a0*self.e0 + a1*self.e1 + self.e2) / a0) - ku1*self.p1 - ku2*self.p2
    self.i = (Ki*_Ts*(a0*self.e0 -    self.e1)           / a0) - ku1*self.i1 - ku2*self.i2 
    self.d = (Kd*_N*(    self.e0 -  2*self.e1 + self.e2) / a0) - ku1*self.d1 - ku2*self.d2

    #calculate next step desired
    self.e2, self.e1, self.e0 = self.e1, self.e0, float(apply_deadzone(setpoint - measurement, deadzone))
    self.u2, self.u1, self.u0 = self.u1, self.u0, (ke0*self.e0 + ke1*self.e1 + ke2*self.e2 - ku1*self.u1 - ku2*self.u2)

    saturation = self.u0
    saturation -= clip(saturation, self.neg_limit, self.pos_limit)
    self._check_saturation(last_output, check_saturation, saturation)

    return clip(self.u0, self.neg_limit, self.pos_limit)

if __name__ == "__main__":
  unittest.main()
