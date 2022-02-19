#!/usr/bin/env python3
import random
import unittest

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
sec = 15 # duration of data in sec
start_offset = 0.1 #sec
override = 5 # seconds
overshoot_allowance_percent = 1
overshoot_allowance = overshoot_allowance_percent / 100

noise_intensity = 10 # noise = rand() * noise_intensity

# NOTE 1.0 is perfect FF, FF does not impact stability

def ratelimit(new, last):
   return clip(new, last - rate_limit, last + rate_limit)

class TestPID(unittest.TestCase):
  def test_error_noise(self):
    self.assertFalse(noise_intensity < 5, msg="Noise not disastrous enough")
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
        for t in range(0, int(4*sec*rate)):
          target = 0
          if (t > start_offset*rate):
            target = step_size * limit
          
          # threshold base on time and error_intensity. multiply runtime by intensity or divide error by intensity
          noise = 0
          if target != 0:
            noise = noise_intensity * ((random.randint(0, 2000) - 1000) / (1000*limit))

          pid_n_control = ratelimit(pid_n.update(target+noise, output_n.x, last_output=pid_n_control, feedforward=target), pid_n_control)
          pid_c_control = ratelimit(pid_c.update(target, output_c.x, last_output=pid_c_control, feedforward=target), pid_c_control)
          pid_r_control = pid_r.update(target+noise, output_r.x, last_output=pid_r_control, feedforward=target)

          last_pid_r_control = ratelimit(pid_r_control, last_pid_r_control)

          output_n.update(pid_n_control)
          output_c.update(pid_c_control)
          output_r.update(last_pid_r_control)

          # calculate error of controller w/ and without rate limits against no noise output with limits
          ss_error += ((output_c.x - output_n.x) - ss_error) / (t+1)
          ss_error += ((output_r.x - output_n.x) - ss_error) / (t+1)

        # assert that moving average of SS error is falling to zero
        # threshold base on time and error_intensity. multiply runtime by intensity or divide error by intensity
        self.assertFalse(abs(ss_error) > 0.1 * noise_intensity * limit / rate, msg=f"Possible SS_error: {ss_error} detected in controller due to noise")
  
  def test_override(self):
    for i in range(0, 2*num_divs + 1):
      for j in range (0, 2*num_divs + 1):

        kp = interp(i, _p[0], _p[1])
        _i = [[0, num_divs, 2*num_divs], [kp/ratio, kp, ratio*kp]]
        ki = interp(j, _i[0], _i[1])

        pid_n = PIController(([0, 1], [kp, kp]), ([0, 1], [ki, ki]), k_f=0, rate=rate, pos_limit=limit, neg_limit=-limit)
        pid_r = PIController(([0, 1], [kp, kp]), ([0, 1], [ki, ki]), k_f=0, rate=rate, pos_limit=limit, neg_limit=-limit)
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
          if output_n.x > target or output_r.x > target:
            sum_overshoot_error += min(max(abs(output_r.x - output_n.x)-output_r.x*(overshoot_allowance), 0), max(abs(target - output_n.x)-target*(overshoot_allowance), 0)) / rate
        
        # assert that the overshoot characteristics of a new input response and releasing the controller on an overriden input are the same
        self.assertFalse(sum_overshoot_error/target > overshoot_allowance, msg=f"Exessive Overshoot measured after override. Exeeded {100*overshoot_allowance}% : {100*sum_overshoot_error/target}% ")

  def test_ramp_up(self):
    for i in range(0, 2*num_divs + 1):
      for j in range (0, 2*num_divs + 1):

        kp = interp(i, _p[0], _p[1])
        _i = [[0, num_divs, 2*num_divs], [kp/ratio, kp, ratio*kp]]
        ki = interp(j, _i[0], _i[1])

        pid_n = PIController(([0, 1], [kp, kp]), ([0, 1], [ki, ki]), k_f=0, pos_limit=limit, neg_limit=-limit)
        pid_r = PIController(([0, 1], [kp, kp]), ([0, 1], [ki, ki]), k_f=0, pos_limit=limit, neg_limit=-limit)
        output_n = FirstOrderFilter(0, 1, 1/rate)
        output_r = FirstOrderFilter(0, 1, 1/rate)
        pid_n_control = 0
        pid_r_control = 0
        last_pid_r_control = 0
        sum_overshoot_error = 0
        for t in range(0, int(sec*rate)):

          target = 0
          if (t > start_offset * rate):
            target = step_size * limit

          pid_n_control = ratelimit(pid_n.update(target, output_n.x, last_output=pid_n_control), pid_n_control)

          pid_r_control = pid_r.update(target, output_r.x, last_output=pid_r_control)

          last_pid_r_control = ratelimit(pid_r_control, last_pid_r_control)

          output_n.update(pid_n_control)
          output_r.update(last_pid_r_control)

          # sum up the (-)overshoot as good, sum up the undershoot as bad
          if output_r.x > target or output_n.x > target:
            sum_overshoot_error += ((output_n.x - output_r.x) + max(target - output_n.x, 0)) / rate

        # assert that the overshoot characteristics of a slew rate limit are improved or equal with compensation on
        self.assertFalse(sum_overshoot_error/target > overshoot_allowance, msg=f"Rate limit compensation caused excessive damping or instability. {100*sum_overshoot_error/target}% additional error from target")    

if __name__ == "__main__":
  unittest.main()
