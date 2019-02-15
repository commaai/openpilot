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


def calc_states_after_delay(states, v_ego, steer_angle, curvature_factor, steer_ratio, delay, long_camera_offset):
  states[0].x = max(0.0, v_ego * delay + long_camera_offset)
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

    if CP.steerResistance > 0 and CP.steerReactance >= 0 and CP.steerInductance > 0:
      self.smooth_factor = CP.steerInductance * 2.0 * CP.steerActuatorDelay / _DT    # Multiplier for inductive component (feed forward)
      self.projection_factor = CP.steerReactance * CP.steerActuatorDelay / 2.0       # Mutiplier for reactive component (PI)
      self.accel_limit = 2.0 / CP.steerResistance                                    # Desired acceleration limit to prevent "whip steer" (resistive component)
      self.ff_angle_factor = 1.0                                                     # Kf multiplier for angle-based feed forward
      self.ff_rate_factor = 10.0                                                      # Kf multiplier for rate-based feed forward
      # Eliminate break-points, since they aren't needed (and would cause problems for resonance)
      KpV = [np.interp(25.0, CP.steerKpBP, CP.steerKpV)]
      KiV = [np.interp(25.0, CP.steerKiBP, CP.steerKiV) * _DT / self.projection_factor]
      self.pid = PIController(([0.], KpV),
                              ([0.], KiV),
                              k_f=CP.steerKf, pos_limit=1.0)
    else:
      self.pid = PIController((CP.steerKpBP, CP.steerKpV),
                              (CP.steerKiBP, CP.steerKiV),
                              k_f=CP.steerKf, pos_limit=1.0)
      self.smooth_factor = 1.0
      self.projection_factor = 0.0
      self.accel_limit = 0.0
      self.ff_angle_factor = 1.0
      self.ff_rate_factor = 0.0
    self.last_cloudlog_t = 0.0
    self.setup_mpc(CP.steerRateCost)
    self.prev_angle_rate = 0
    self.feed_forward = 0.0
    self.last_mpc_ts = 0.0
    self.angle_steers_des = 0.0
    self.angle_steers_des_time = 0.0
    self.angle_steers_des_mpc = 0.0
    self.steer_counter = 1.0
    self.steer_counter_prev = 0.0
    self.rough_steers_rate = 0.0
    self.prev_angle_steers = 0.0
    self.calculate_rate = True

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

  def reset(self):
    self.pid.reset()

  def update(self, active, v_ego, angle_steers, angle_rate, steer_override, d_poly, angle_offset, CP, VM, PL):
    self.mpc_updated = False

    if angle_rate == 0.0 and self.calculate_rate:
      if angle_steers != self.prev_angle_steers:
        self.steer_counter_prev = self.steer_counter
        self.rough_steers_rate = (self.rough_steers_rate + 100.0 * (angle_steers - self.prev_angle_steers) / self.steer_counter_prev) / 2.0
        self.steer_counter = 0.0
      elif self.steer_counter >= self.steer_counter_prev:
        self.rough_steers_rate = (self.steer_counter * self.rough_steers_rate) / (self.steer_counter + 1.0)
      self.steer_counter += 1.0
      angle_rate = self.rough_steers_rate

      # Don't use accelerated rate unless it's from CAN
      accelerated_angle_rate = angle_rate
    else:
      # If non-zero angle_rate is provided, use it instead
      self.calculate_rate = False
      # Use steering rate from the last 2 samples to estimate acceleration for a likely future steering rate
      accelerated_angle_rate = 2.0 * angle_rate - self.prev_angle_rate

    # TODO: this creates issues in replay when rewinding time: mpc won't run
    if self.last_mpc_ts < PL.last_md_ts:
      self.last_mpc_ts = PL.last_md_ts
      cur_time = sec_since_boot()
      mpc_time = float(self.last_mpc_ts / 1000000000.0)
      curvature_factor = VM.curvature_factor(v_ego)

      # Determine future angle steers using steer rate
      projected_angle_steers = float(angle_steers) + CP.steerActuatorDelay * float(accelerated_angle_rate)

      # Determine a proper delay time that includes the model's variable processing time
      plan_age = _DT_MPC + cur_time - mpc_time
      total_delay = CP.steerActuatorDelay + plan_age

      l_poly = libmpc_py.ffi.new("double[4]", list(PL.PP.l_poly))
      r_poly = libmpc_py.ffi.new("double[4]", list(PL.PP.r_poly))
      p_poly = libmpc_py.ffi.new("double[4]", list(PL.PP.p_poly))

      # account for actuation delay and the age of the plan
      self.cur_state = calc_states_after_delay(self.cur_state, v_ego, projected_angle_steers, curvature_factor,
                                                      CP.steerRatio, total_delay, CP.eonToFront)

      v_ego_mpc = max(v_ego, 5.0)  # avoid mpc roughness due to low speed
      self.libmpc.run_mpc(self.cur_state, self.mpc_solution,
                          l_poly, r_poly, p_poly,
                          PL.PP.l_prob, PL.PP.r_prob, PL.PP.p_prob, curvature_factor, v_ego_mpc, PL.PP.lane_width)

      self.mpc_updated = True

      #  Check for infeasable MPC solution
      self.mpc_nans = np.any(np.isnan(list(self.mpc_solution[0].delta)))
      if not self.mpc_nans:
        self.mpc_angles = [self.angle_steers_des,
                          float(math.degrees(self.mpc_solution[0].delta[1] * CP.steerRatio) + angle_offset),
                          float(math.degrees(self.mpc_solution[0].delta[2] * CP.steerRatio) + angle_offset)]

        self.mpc_times = [self.angle_steers_des_time,
                          mpc_time + _DT_MPC,
                          mpc_time + _DT_MPC + _DT_MPC]

        self.angle_steers_des_mpc = self.mpc_angles[1]
      else:
        self.libmpc.init(MPC_COST_LAT.PATH, MPC_COST_LAT.LANE, MPC_COST_LAT.HEADING, CP.steerRateCost)
        self.cur_state[0].delta = math.radians(angle_steers) / CP.steerRatio

        if cur_time > self.last_cloudlog_t + 5.0:
          self.last_cloudlog_t = cur_time
          cloudlog.warning("Lateral mpc - nan: True")

    cur_time = sec_since_boot()
    self.angle_steers_des_time = cur_time

    if v_ego < 0.3 or not active:
      output_steer = 0.0
      self.feed_forward = 0.0
      self.pid.reset()
      self.angle_steers_des = angle_steers
      self.cur_state[0].delta = math.radians(angle_steers - angle_offset) / CP.steerRatio
    else:
      # Interpolate desired angle between MPC updates
      self.angle_steers_des = np.interp(cur_time, self.mpc_times, self.mpc_angles)
      self.angle_steers_des_time = cur_time
      self.cur_state[0].delta = math.radians(self.angle_steers_des - angle_offset) / CP.steerRatio

      # Determine the target steer rate for desired angle, but prevent the acceleration limit from being exceeded
      # Restricting the steer rate creates the resistive component needed for resonance
      restricted_steer_rate = np.clip(self.angle_steers_des - float(angle_steers) , float(accelerated_angle_rate) - self.accel_limit,
                                                                                    float(accelerated_angle_rate) + self.accel_limit)

      # Determine projected desired angle that is within the acceleration limit (prevent the steering wheel from jerking)
      projected_angle_steers_des = self.angle_steers_des + self.projection_factor * restricted_steer_rate

      # Determine future angle steers using accellerated steer rate
      projected_angle_steers = float(angle_steers) + self.projection_factor * float(accelerated_angle_rate)

      steers_max = get_steer_max(CP, v_ego)
      self.pid.pos_limit = steers_max
      self.pid.neg_limit = -steers_max
      deadzone = 0.0

      if CP.steerControlType == car.CarParams.SteerControlType.torque:
        # Decide which feed forward mode should be used (angle or rate).  Use more dominant mode, but only if conditions are met
        # Spread feed forward out over a period of time to make it inductive (for resonance)
        if abs(self.ff_rate_factor * float(restricted_steer_rate)) > abs(self.ff_angle_factor * float(self.angle_steers_des) - float(angle_offset)) - 0.5 \
            and (abs(float(restricted_steer_rate)) > abs(accelerated_angle_rate) or (float(restricted_steer_rate) < 0) != (accelerated_angle_rate < 0)) \
            and (float(restricted_steer_rate) < 0) == (float(self.angle_steers_des) - float(angle_offset) - 0.5 < 0):
          self.feed_forward = (((self.smooth_factor - 1.) * self.feed_forward) + self.ff_rate_factor * v_ego**2 * float(restricted_steer_rate)) / self.smooth_factor
        elif abs(self.angle_steers_des - float(angle_offset)) > 0.5:
          self.feed_forward = (((self.smooth_factor - 1.) * self.feed_forward) + self.ff_angle_factor * v_ego**2 \
                              * float(apply_deadzone(float(self.angle_steers_des) - float(angle_offset), 0.5))) / self.smooth_factor
        else:
          self.feed_forward = (((self.smooth_factor - 1.) * self.feed_forward) + 0.0) / self.smooth_factor

        # Use projected desired and actual angles instead of "current" values, in order to make PI more reactive (for resonance)
        output_steer = self.pid.update(projected_angle_steers_des, projected_angle_steers, check_saturation=(v_ego > 10),
                                        override=steer_override, feedforward=self.feed_forward, speed=v_ego, deadzone=deadzone)

    self.sat_flag = self.pid.saturated
    self.prev_angle_rate = angle_rate
    self.prev_angle_steers = angle_steers

    # return MPC angle in the unused output (for ALCA)
    if CP.steerControlType == car.CarParams.SteerControlType.torque:
      return output_steer, self.angle_steers_des
    else:
      return self.angle_steers_des_mpc, float(self.angle_steers_des)
