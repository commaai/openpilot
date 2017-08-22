import math
import numpy as np
from common.numpy_fast import clip, interp
from selfdrive.config import Conversions as CV

_K_CURV_V = [1., 0.6]
_K_CURV_BP = [0., 0.002]

def calc_d_lookahead(v_ego, d_poly):
  #*** this function computes how far too look for lateral control
  # howfar we look ahead is function of speed and how much curvy is the path
  offset_lookahead = 1.
  k_lookahead = 7.
  # integrate abs value of second derivative of poly to get a measure of path curvature
  pts_len = 50.  # m
  if len(d_poly)>0:
    pts = np.polyval([6*d_poly[0], 2*d_poly[1]], np.arange(0, pts_len))
  else:
    pts = 0.
  curv = np.sum(np.abs(pts))/pts_len

  k_curv = interp(curv, _K_CURV_BP, _K_CURV_V)

  # sqrt on speed is needed to keep, for a given curvature, the y_des
  # proportional to speed. Indeed, y_des is prop to d_lookahead^2
  # 36m at 25m/s
  d_lookahead = offset_lookahead + math.sqrt(max(v_ego, 0)) * k_lookahead * k_curv
  return d_lookahead

def calc_lookahead_offset(v_ego, angle_steers, d_lookahead, VM, angle_offset):
  #*** this function returns the lateral offset given the steering angle, speed and the lookahead distance
  sa = (angle_steers - angle_offset) * CV.DEG_TO_RAD
  curvature = VM.calc_curvature(sa, v_ego)
  # clip is to avoid arcsin NaNs due to too sharp turns
  y_actual = d_lookahead * np.tan(np.arcsin(np.clip(d_lookahead * curvature, -0.999, 0.999))/2.)
  return y_actual, curvature

def calc_desired_steer_angle(v_ego, y_des, d_lookahead, VM, angle_offset):
  # inverse of the above function
  curvature = np.sin(np.arctan(y_des / d_lookahead) * 2.) / d_lookahead
  steer_des = VM.get_steer_from_curvature(curvature, v_ego) * CV.RAD_TO_DEG + angle_offset
  return steer_des, curvature

def pid_lateral_control(v_ego, sa_actual, sa_des, Ui_steer, steer_max,
                        steer_override, sat_count, enabled, Kp, Ki, rate):

  sat_count_rate = 1./rate
  sat_count_limit = 0.8      # after 0.8s of continuous saturation, an alert will be sent

  error_steer = sa_des - sa_actual
  Ui_unwind_speed = 0.3/rate   #.3 per second

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

  def update(self, enabled, v_ego, angle_steers, steer_override, d_poly, angle_offset, VM):
    rate = 100

    steer_max = 1.0

    # how far we look ahead is function of speed and desired path
    d_lookahead = calc_d_lookahead(v_ego, d_poly)

    # desired lookahead offset
    self.y_des = np.polyval(d_poly, d_lookahead)

    # calculate actual offset at the lookahead point
    self.angle_steers_des, _ = calc_desired_steer_angle(v_ego, self.y_des,
                                                d_lookahead, VM, angle_offset)

    output_steer, self.Up_steer, self.Ui_steer, self.lateral_control_sat, self.sat_count, sat_flag = pid_lateral_control(
      v_ego, angle_steers, self.angle_steers_des, self.Ui_steer, steer_max,
      steer_override, self.sat_count, enabled, VM.CP.steerKp, VM.CP.steerKi, rate)

    final_steer = clip(output_steer, -steer_max, steer_max)
    return final_steer, sat_flag
