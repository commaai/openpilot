#!/usr/bin/env python3
import random
import unittest
import sys

from common.numpy_fast import clip, interp
from common.filter_simple import FirstOrderFilter
from selfdrive.controls.lib.pid import PIController

gain_range = 10 # (1 / range -> 1 ->  range   )
ratio = 4 #   I = (P / ratio -> P -> P * ratio)

num_divs = 4# (2*n + 1)**2 graphs
_p = [[0, num_divs, 2*num_divs],[1/gain_range, 1, gain_range]]


limit = 100 # max output 
step_size = 0.5 # percent of output to request during tests. 
rate = 100 # number of steps per second
ttm = 1 # seconds to max rate actuator
rate_limit = limit / (ttm * rate) # max change in request per cycle
sec = 60 # duration of data in sec
start_offset = 0.1 #sec
override = 5 # seconds
override_allowance_percent = 1
override_allowance = override_allowance_percent / 100

noise_intensity = 10 # noise = rand() * (noise_intensity * rate_limit) / kp

# 1.0 is perfect FF

if __name__ == "__main__":
  plot_all = False
  plot = False
  debug = False
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
      plot = True
    if "debug" in sys.argv:
      debug = True
    else:
      plot_all = True
    if plot and not (debug_noise or debug_override or debug_ramp):
      debug_noise = True
      debug_override = True
      debug_ramp = True
    if (plot):
      num_divs = 4 # (2*n + 1)**2 graphs (81 graphs)
      _p = [[0, num_divs, 2*num_divs],[1/gain_range, 1, gain_range]]
    #remove args so unittest is not freaked out
    while len(sys.argv) > 1:
      sys.argv.pop()
  
if plot and (debug_noise or debug_override or debug_ramp):
  import matplotlib.pyplot as plt

def ratelimit(new, last):
   return clip(new, last - rate_limit, last + rate_limit)

class TestPID(unittest.TestCase):
  def test_error_noise(self):
    self.assertFalse((debug_ramp or debug_override) and not debug_noise, msg="Test skipped for debugging")
    self.assertFalse(noise_intensity < 1, msg="Noise not disastrous enough")
    for i in range(0, 2*num_divs + 1):
      for j in range (0, 2*num_divs + 1):

        kp = interp(i, _p[0], _p[1])
        _i = [[0, num_divs, 2*num_divs], [kp/ratio, kp, ratio*kp]]
        ki = interp(j, _i[0], _i[1])

        pid_n = PIController(([0, 1], [kp, kp]), ([0, 1], [ki, ki]), k_f=0, rate=rate, pos_limit=limit, neg_limit=-limit)
        pid_c = PIController(([0, 1], [kp, kp]), ([0, 1], [ki, ki]), k_f=0, rate=rate, pos_limit=limit, neg_limit=-limit)
        pid_r = PIController(([0, 1], [kp, kp]), ([0, 1], [ki, ki]), k_f=0, rate=rate, pos_limit=limit, neg_limit=-limit)
        output_n = FirstOrderFilter(0, 1, 1/rate)
        output_c = FirstOrderFilter(0, 1, 1/rate)
        output_r = FirstOrderFilter(0, 1, 1/rate)
        ss_error = 0
        pid_n_control = 0
        pid_c_control = 0
        pid_r_control = 0
        last_pid_r_control = 0
        for t in range(0, int(noise_intensity*sec*rate)):
          target = 0
          if (t > start_offset*rate):
            target = step_size * limit
          
          # threshold base on time and error_intensity. multiply runtime by intensity or divide error by intensity
          noise = 0
          if target != 0:
            noise = noise_intensity * (rate_limit / kp) * ((random.randint(0, 200*gain_range) - 100*gain_range) / (100*gain_range))

          pid_n_control = ratelimit(pid_n.update(target+noise, output_n.x, last_output=pid_n_control, feedforward=target+noise), pid_n_control)
          pid_c_control = ratelimit(pid_c.update(target, output_c.x, last_output=pid_c_control, feedforward=target), pid_c_control)
          pid_r_control = pid_r.update(target+noise, output_r.x, last_output=pid_r_control, feedforward=target+noise)

          last_pid_r_control = ratelimit(pid_r_control, last_pid_r_control)

          output_n.update(pid_n_control)
          output_c.update(pid_c_control)
          output_r.update(last_pid_r_control)

          # calculate error of controller w/ and without rate limits against no noise output with limits
          ss_error += ((output_c.x - output_n.x) - ss_error) / (t+1)
          ss_error += ((output_r.x - output_n.x) - ss_error) / (t+1)

        # assert that moving average of SS error is falling to zero
        # threshold base on time and error_intensity. multiply runtime by intensity or divide error by intensity
        self.assertTrue(abs(ss_error) < limit / rate, msg=f"Possible SS_error: {ss_error} detected in controller due to noise")
  
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

        pid_n = PIController(([0, 1], [kp, kp]), ([0, 1], [ki, ki]), k_f=1, rate=rate, pos_limit=limit, neg_limit=-limit)
        pid_r = PIController(([0, 1], [kp, kp]), ([0, 1], [ki, ki]), k_f=1, rate=rate, pos_limit=limit, neg_limit=-limit)
        output_n = FirstOrderFilter(0, 1, 1/rate)
        output_r = FirstOrderFilter(0, 1, 1/rate)
        pid_n_control = 0
        pid_r_control = 0
        sum_overshoot_error = 0
        for t in range(0, int(sec*rate)):
          target = 0
          if (t > start_offset*rate):
            target = step_size * limit

          pid_n_control = ratelimit(pid_n.update(target, output_n.x, last_output=pid_n_control), pid_n_control)

          # override / hold output of controller at 0 
          if t > override*rate:
            pid_r_control = ratelimit(pid_r.update(target, output_r.x, last_output=pid_r_control), pid_r_control)
            output_r.update(pid_r_control)
            output_n.update(pid_n_control)
          else: 
            pid_n_control = 0

          # only sum up overshoot in excess of the allowance of both the unclamped signal and the target. 
          if output_n.x > target:
            sum_overshoot_error += min(max(abs(output_r.x - output_n.x)-output_r.x*(override_allowance), 0), max(abs(target - output_n.x)-target*(override_allowance), 0)) / rate
        
        # assert that the overshoot characteristics of a new input response and releasing the controller on an overriden input are the same
        self.assertFalse(sum_overshoot_error/target > override_allowance, msg=f"Exessive Overshoot measured after override. Exeeded {100*override_allowance}% : {100*sum_overshoot_error/target}% ")

  def test_ramp_up(self):
    #TODO New Tests
    #TODO Check that sum of error is low above target. 

    self.assertFalse((debug_override or debug_noise) and not debug_ramp, msg="Test skipped for debugging")
    for i in range(0, 2*num_divs + 1):
      for j in range (0, 2*num_divs + 1):

        kp = interp(i, _p[0], _p[1])
        _i = [[0, num_divs, 2*num_divs], [kp/ratio, kp, ratio*kp]]
        ki = interp(j, _i[0], _i[1])

        pid_n = PIController(([0, 1], [kp, kp]), ([0, 1], [ki, ki]), k_f=1, pos_limit=limit, neg_limit=-limit)
        pid_r = PIController(([0, 1], [kp, kp]), ([0, 1], [ki, ki]), k_f=1, pos_limit=limit, neg_limit=-limit)
        output_n = FirstOrderFilter(0, 1, 1/rate)
        output_r = FirstOrderFilter(0, 1, 1/rate)
        last_pid_n_control = 0
        last_pid_r_control = 0

        if plot and debug_ramp:
          x = []
          y4 = []
          y5 = []
        for t in range(0, int(sec*rate)):
          if plot and debug_ramp:
            x.append(t / rate)
            y4.append(output_n.x)
            y5.append(output_r.x)

          target = 0
          if (t > start_offset * rate):
            target = step_size * limit

          pid_n_control_raw = pid_n.update(target, output_n.x, last_output=last_pid_n_control)
          pid_n_control = clip(pid_n_control_raw, last_pid_n_control - rate_limit, last_pid_n_control + rate_limit)

          last_pid_r_control = pid_r.update(target, output_r.x, last_output=last_pid_r_control)

          last_pid_n_control = pid_n_control
          output_n.update(pid_n_control)
          output_r.update(last_pid_r_control)

        if plot and debug_ramp and plot_all:
          plt.plot(x, y5, 'k', label="No Rate Limit")
          plt.plot(x, y4, label="Rate Limit")
          plt.title(f"Slew Rate Limit Test: P = {kp:5.3}, I = {ki:5.3}")
          plt.legend()
          plt.show()

if __name__ == "__main__":
  unittest.main()
