import math
import numpy as np
from selfdrive.controls.lib.pid import PIController
from selfdrive.controls.lib.drive_helpers import MPC_COST_LAT
from selfdrive.controls.lib.lateral_mpc import libmpc_py
from common.numpy_fast import interp
from common.realtime import sec_since_boot
from selfdrive.swaglog import cloudlog
from cereal import car

_DT = 0.01    # 100Hz
_DT_MPC = 0.05  # 20Hz


def calc_states_after_delay(states, v_ego, steer_angle, curvature_factor, steer_ratio, delay):
  states[0].x = v_ego * delay
  states[0].psi = v_ego * curvature_factor * math.radians(steer_angle) / steer_ratio * delay
  return states


def get_steer_max(CP, v_ego):
  return interp(v_ego, CP.steerMaxBP, CP.steerMaxV)

def apply_deadzone(angle, deadzone):
  if angle > deadzone:
    angle -= deadzone
  elif angle < -deadzone:
    angle += deadzone
  else:
    angle = 0.
  return angle


class LatControl(object):
  def __init__(self, CP):
    self.pid = PIController((CP.steerKpBP, CP.steerKpV),
                            (CP.steerKiBP, CP.steerKiV),
                            k_f=CP.steerKf, pos_limit=1.0)
    self.last_cloudlog_t = 0.0
    self.setup_mpc(CP.steerRateCost)
    self.smooth_factor = 2.0 * CP.steerActuatorDelay / _DT      # Multiplier for inductive component (feed forward)
    self.projection_factor = 5.0 * _DT                       #  Mutiplier for reactive component (PI)
    self.accel_limit = 2.0                                 # Desired acceleration limit to prevent "whip steer" (resistive component)
    self.ff_angle_factor = 0.5         # Kf multiplier for angle-based feed forward
    self.ff_rate_factor = 5.0         # Kf multiplier for rate-based feed forward
    self.prev_angle_rate = 0
    self.feed_forward = 0.0
    self.steerActuatorDelay = CP.steerActuatorDelay
    self.last_mpc_ts = 0.0
    self.angle_steers_des = 0.0
    self.angle_steers_des_mpc = 0.0
    self.angle_steers_des_time = 0.0
    self.avg_angle_steers = 0.0
    self.projected_angle_steers = 0.0

  def setup_mpc(self, steer_rate_cost):
    self.libmpc = libmpc_py.libmpc
    self.libmpc.init(MPC_COST_LAT.PATH, MPC_COST_LAT.LANE, MPC_COST_LAT.HEADING, steer_rate_cost)

    self.mpc_solution = libmpc_py.ffi.new("log_t *")
    self.cur_state = libmpc_py.ffi.new("state_t *")
    self.mpc_angles = [0.0, 0.0, 0.0]
    self.mpc_times = [0.0, 0.0, 0.0]
    self.mpc_updated = False
    self.mpc_nans = False
    self.cur_state[0].x = 0.0
    self.cur_state[0].y = 0.0
    self.cur_state[0].psi = 0.0
    self.cur_state[0].delta = 0.0

    self.blindspot_blink_counter_left_check = 0
    self.blindspot_blink_counter_right_check = 0


  def reset(self):
    self.pid.reset()

  def update(self, active, v_ego, angle_steers, angle_rate, steer_override, d_poly, angle_offset, CP, VM, PL,blindspot,leftBlinker,rightBlinker):
    self.mpc_updated = False

    # TODO: this creates issues in replay when rewinding time: mpc won't run
    if self.last_mpc_ts < PL.last_md_ts:
      cur_time = sec_since_boot()
      self.last_mpc_ts = PL.last_md_ts

      self.curvature_factor = VM.curvature_factor(v_ego)

      # Use steering rate from the last 2 samples to estimate acceleration for a more realistic future steering rate
      accelerated_angle_rate = 2.0 * angle_rate - self.prev_angle_rate

      # Determine future angle steers using accelerated steer rate
      self.projected_angle_steers = float(angle_steers) + self.projection_factor * float(accelerated_angle_rate)

      # Determine a proper delay time that includes the model's processing time, which is variable
      plan_age = cur_time - float(self.last_mpc_ts / 1000000000.0)
      total_delay = CP.steerActuatorDelay + plan_age

      self.l_poly = libmpc_py.ffi.new("double[4]", list(PL.PP.l_poly))
      self.r_poly = libmpc_py.ffi.new("double[4]", list(PL.PP.r_poly))
      self.p_poly = libmpc_py.ffi.new("double[4]", list(PL.PP.p_poly))

      # account for actuation delay and the age of the plan
      self.cur_state = calc_states_after_delay(self.cur_state, v_ego, self.projected_angle_steers, self.curvature_factor, CP.steerRatio, total_delay)

      v_ego_mpc = max(v_ego, 5.0)  # avoid mpc roughness due to low speed
      self.libmpc.run_mpc(self.cur_state, self.mpc_solution,
                          self.l_poly, self.r_poly, self.p_poly,
                          PL.PP.l_prob, PL.PP.r_prob, PL.PP.p_prob, self.curvature_factor, v_ego_mpc, PL.PP.lane_width)

      self.mpc_updated = True

      #  Check for infeasable MPC solution
      self.mpc_nans = np.any(np.isnan(list(self.mpc_solution[0].delta)))
      if not self.mpc_nans:
        self.mpc_angles = [self.angle_steers_des,
                          float(math.degrees(self.mpc_solution[0].delta[1] * CP.steerRatio) + angle_offset),
                          float(math.degrees(self.mpc_solution[0].delta[2] * CP.steerRatio) + angle_offset)]

        self.mpc_times = [self.angle_steers_des_time,
                        float(self.last_mpc_ts / 1000000000.0) + _DT_MPC,
                        float(self.last_mpc_ts / 1000000000.0) + _DT_MPC + _DT_MPC]

        self.angle_steers_des_mpc = self.mpc_angles[1]
      else:
        self.libmpc.init(MPC_COST_LAT.PATH, MPC_COST_LAT.LANE, MPC_COST_LAT.HEADING, CP.steerRateCost)
        self.cur_state[0].delta = math.radians(angle_steers) / CP.steerRatio

        if cur_time > self.last_cloudlog_t + 5.0:
          self.last_cloudlog_t = cur_time
          cloudlog.warning("Lateral mpc - nan: True")

    if v_ego < 0.3 or not active:
      output_steer = 0.0
      self.feed_forward = 0.0
      self.pid.reset()
      self.angle_steers_des = angle_steers
      self.avg_angle_steers = angle_steers
      self.cur_state[0].delta = math.radians(angle_steers - angle_offset) / CP.steerRatio
    else:
      cur_time = sec_since_boot()
      # Interpolate desired angle between MPC updates
      self.angle_steers_des = np.interp(cur_time, self.mpc_times, self.mpc_angles)
      self.angle_steers_des_time = cur_time
      self.avg_angle_steers = (4.0 * self.avg_angle_steers + angle_steers) / 5.0
      self.cur_state[0].delta = math.radians(self.angle_steers_des - angle_offset) / CP.steerRatio

      # Determine the target steer rate for desired angle, but prevent the acceleration limit from being exceeded
      # Restricting the steer rate creates the resistive component needed for resonance
      restricted_steer_rate = np.clip(self.angle_steers_des - float(angle_steers) , float(angle_rate) - self.accel_limit, float(angle_rate) + self.accel_limit)

      # Determine projected desired angle that is within the acceleration limit (prevent the steering wheel from jerking)
      projected_angle_steers_des = self.angle_steers_des + self.projection_factor * restricted_steer_rate

      # Determine future angle steers using accelerated steer rate
      self.projected_angle_steers = float(angle_steers) + self.projection_factor * float(angle_rate)

      deadzone = 0.0
      steers_max = get_steer_max(CP, v_ego)
      self.pid.pos_limit = steers_max
      self.pid.neg_limit = -steers_max

      if CP.steerControlType == car.CarParams.SteerControlType.torque:
        # Decide which feed forward mode should be used (angle or rate).  Use more dominant mode, and only if conditions are met
        # Spread feed forward out over a period of time to make it more inductive (for resonance)
        if abs(self.ff_rate_factor * float(restricted_steer_rate)) > abs(self.ff_angle_factor * float(self.angle_steers_des) - float(angle_offset)) - 0.5 \
            and (abs(float(restricted_steer_rate)) > abs(angle_rate) or (float(restricted_steer_rate) < 0) != (angle_rate < 0)) \
            and (float(restricted_steer_rate) < 0) == (float(self.angle_steers_des) - float(angle_offset) - 0.5 < 0):
          self.feed_forward = (((self.smooth_factor - 1.) * self.feed_forward) + self.ff_rate_factor * v_ego**2 * float(restricted_steer_rate)) / self.smooth_factor
        elif abs(self.angle_steers_des - float(angle_offset)) > 0.5:
          self.feed_forward = (((self.smooth_factor - 1.) * self.feed_forward) + self.ff_angle_factor * v_ego**2 * float(apply_deadzone(float(self.angle_steers_des) - float(angle_offset), 0.5))) / self.smooth_factor
        else:
          self.feed_forward = (((self.smooth_factor - 1.) * self.feed_forward) + 0.0) / self.smooth_factor

      # Use projected desired and actual angles instead of "current" values, in order to make PI more reactive (for resonance)
      output_steer = self.pid.update(projected_angle_steers_des, self.projected_angle_steers, check_saturation=(v_ego > 10), override=steer_override,
                                     feedforward=self.feed_forward, speed=v_ego, deadzone=deadzone)

      if rightBlinker:
        if blindspot:
          self.blindspot_blink_counter_right_check = 0
          print "debug: blindspot detected"
        self.blindspot_blink_counter_right_check += 1
        if self.blindspot_blink_counter_right_check > 150:
          self.angle_steers_des -= 0#15

      else:
        self.blindspot_blink_counter_right_check = 0

      if leftBlinker:
        if blindspot:
          self.blindspot_blink_counter_left_check = 0
          print "debug: blindspot detected"
        self.blindspot_blink_counter_left_check += 1
        if self.blindspot_blink_counter_left_check > 150:
          self.angle_steers_des += 0#15
      else:
        self.blindspot_blink_counter_left_check = 0
      # Hide angle error if being overriden
      if steer_override:
        self.projected_angle_steers = self.mpc_angles[1]
        self.avg_angle_steers = self.mpc_angles[1]
        
    self.sat_flag = self.pid.saturated
    self.prev_angle_rate = angle_rate

    # ALCA works better with the non-interpolated angle
    if CP.steerControlType == car.CarParams.SteerControlType.torque:
      return output_steer, float(self.angle_steers_des_mpc)
    else:
      return float(self.angle_steers_des_mpc), float(self.angle_steers_des)
