import numpy as np
from common.numpy_fast import clip, interp

def rate_limit(new_value, last_value, dw_step, up_step):
  return clip(new_value, last_value + dw_step, last_value + up_step)

def learn_angle_offset(lateral_control, v_ego, angle_offset, d_poly, y_des, steer_override):
  # simple integral controller that learns how much steering offset to put to have the car going straight
  min_offset = -1.  # deg
  max_offset =  1.  # deg
  alpha = 1./36000. # correct by 1 deg in 2 mins, at 30m/s, with 50cm of error, at 20Hz
  min_learn_speed = 1.

  # learn less at low speed or when turning
  alpha_v = alpha*(max(v_ego - min_learn_speed, 0.))/(1. + 0.5*abs(y_des))

  # only learn if lateral control is active and if driver is not overriding:
  if lateral_control and not steer_override:
    angle_offset += d_poly[3] * alpha_v
    angle_offset = clip(angle_offset, min_offset, max_offset)

  return angle_offset

def actuator_hystereses(final_brake, braking, brake_steady, v_ego, civic):
  # hyst params... TODO: move these to VehicleParams
  brake_hyst_on = 0.055 if civic else 0.1    # to activate brakes exceed this value
  brake_hyst_off = 0.005                     # to deactivate brakes below this value
  brake_hyst_gap = 0.01                      # don't change brake command for small ocilalitons within this value

  #*** histeresys logic to avoid brake blinking. go above 0.1 to trigger
  if (final_brake < brake_hyst_on and not braking) or final_brake < brake_hyst_off:
    final_brake = 0.
  braking = final_brake > 0.

  # for small brake oscillations within brake_hyst_gap, don't change the brake command
  if final_brake == 0.:
    brake_steady = 0.
  elif final_brake > brake_steady + brake_hyst_gap:
    brake_steady = final_brake - brake_hyst_gap
  elif final_brake < brake_steady - brake_hyst_gap:
    brake_steady = final_brake + brake_hyst_gap
  final_brake = brake_steady

  if not civic:
    brake_on_offset_v  = [.25, .15]   # min brake command on brake activation. below this no decel is perceived
    brake_on_offset_bp = [15., 30.]     # offset changes VS speed to not have too abrupt decels at high speeds
    # offset the brake command for threshold in the brake system. no brake torque perceived below it
    brake_on_offset = interp(v_ego, brake_on_offset_bp, brake_on_offset_v)
    brake_offset = brake_on_offset - brake_hyst_on
    if final_brake > 0.0:
      final_brake += brake_offset

  return final_brake, braking, brake_steady
