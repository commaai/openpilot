import math
import numpy as np
from common.numpy_fast import clip

def calc_curvature(v_ego, angle_steers, VP, angle_offset=0):
  deg_to_rad = np.pi/180.
  angle_steers_rad = (angle_steers - angle_offset) * deg_to_rad
  curvature = angle_steers_rad/(VP.steer_ratio * VP.wheelbase * (1. + VP.slip_factor * v_ego**2))
  return curvature

def calc_d_lookahead(v_ego):
  #*** this function computes how far too look for lateral control
  # howfar we look ahead is function of speed
  offset_lookahead = 1.
  coeff_lookahead = 4.4
  # sqrt on speed is needed to keep, for a given curvature, the y_offset 
  # proportional to speed. Indeed, y_offset is prop to d_lookahead^2
  # 26m at 25m/s
  d_lookahead = offset_lookahead + math.sqrt(max(v_ego, 0)) * coeff_lookahead
  return d_lookahead

def calc_lookahead_offset(v_ego, angle_steers, d_lookahead, VP, angle_offset):
  #*** this function return teh lateral offset given the steering angle, speed and the lookahead distance
  curvature = calc_curvature(v_ego, angle_steers, VP, angle_offset)

  # clip is to avoid arcsin NaNs due to too sharp turns
  y_actual = d_lookahead * np.tan(np.arcsin(np.clip(d_lookahead * curvature, -0.999, 0.999))/2.)
  return y_actual, curvature

def pid_lateral_control(v_ego, y_actual, y_des, Ui_steer, steer_max,
                        steer_override, sat_count, enabled, half_pid, rate):

  sat_count_rate = 1./rate
  sat_count_limit = 0.8       # after 0.8s of continuous saturation, an alert will be sent

  error_steer = y_des - y_actual
  Ui_unwind_speed = 0.3/rate   #.3 per second
  if not half_pid:
    Kp, Ki = 12.0, 1.0
  else:
    Kp, Ki = 6.0, .5           # 2x limit in ILX
  Up_steer = error_steer*Kp
  Ui_steer_new = Ui_steer + error_steer*Ki * 1./rate
  output_steer_new = Ui_steer_new + Up_steer

  # Anti-wind up for integrator: do not integrate if we are against the steer limits
  if (
    (error_steer >= 0. and (output_steer_new < steer_max or Ui_steer < 0)) or
    (error_steer <= 0. and
     (output_steer_new > -steer_max or Ui_steer > 0))) and not steer_override:
    #update integrator
    Ui_steer = Ui_steer_new
  # unwind integrator if driver is maneuvering the steering wheel
  elif steer_override:
    Ui_steer -= Ui_unwind_speed * np.sign(Ui_steer)

  # still, intergral term should not be bigger then limits
  Ui_steer = clip(Ui_steer, -steer_max, steer_max)

  output_steer = Up_steer + Ui_steer

  # don't run steer control if at very low speed
  if v_ego < 0.3 or not enabled:
    output_steer = 0.
    Ui_steer = 0.

  # useful to know if control is against the limit
  lateral_control_sat = False
  if abs(output_steer) > steer_max:
    lateral_control_sat = True

  output_steer = clip(output_steer, -steer_max, steer_max)

  # if lateral control is saturated for a certain period of time, send an alert for taking control of the car
  # wind
  if lateral_control_sat and not steer_override and v_ego > 10 and abs(error_steer) > 0.1:
    sat_count += sat_count_rate
  # unwind
  else:
    sat_count -= sat_count_rate

  sat_flag = False
  if sat_count >= sat_count_limit:
    sat_flag = True

  sat_count = clip(sat_count, 0, 1)

  return output_steer, Up_steer, Ui_steer, lateral_control_sat, sat_count, sat_flag

class LatControl(object):
  def __init__(self):
    self.Up_steer = 0.
    self.sat_count = 0
    self.y_des = 0.0
    self.lateral_control_sat = False
    self.Ui_steer = 0.
    self.reset()

  def reset(self):
    self.Ui_steer = 0.

  def update(self, enabled, v_ego, angle_steers, steer_override, d_poly, angle_offset, VP):
    rate = 100

    steer_max = 1.0

    # how far we look ahead is function of speed
    d_lookahead = calc_d_lookahead(v_ego)

    # calculate actual offset at the lookahead point
    self.y_actual, _ = calc_lookahead_offset(v_ego, angle_steers,
                                             d_lookahead, VP, angle_offset)

    # desired lookahead offset
    self.y_des = np.polyval(d_poly, d_lookahead)

    output_steer, self.Up_steer, self.Ui_steer, self.lateral_control_sat, self.sat_count, sat_flag = pid_lateral_control(
      v_ego, self.y_actual, self.y_des, self.Ui_steer, steer_max,
      steer_override, self.sat_count, enabled, VP.torque_mod, rate)

    final_steer = clip(output_steer, -steer_max, steer_max)
    return final_steer, sat_flag
