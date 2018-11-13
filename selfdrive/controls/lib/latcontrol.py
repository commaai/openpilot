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
  def __init__(self, VM):
    self.pid = PIController((VM.CP.steerKpBP, VM.CP.steerKpV),
                            (VM.CP.steerKiBP, VM.CP.steerKiV),
                            k_f=VM.CP.steerKf, pos_limit=1.0)
    self.last_cloudlog_t = 0.0
    self.setup_mpc(VM.CP.steerRateCost)

  def setup_mpc(self, steer_rate_cost):
    self.libmpc = libmpc_py.libmpc
    self.libmpc.init(MPC_COST_LAT.PATH, MPC_COST_LAT.LANE, MPC_COST_LAT.HEADING, steer_rate_cost)

    self.mpc_solution = libmpc_py.ffi.new("log_t *")
    self.cur_state = libmpc_py.ffi.new("state_t *")
    self.mpc_updated = False
    self.mpc_nans = False
    self.cur_state[0].x = 0.0
    self.cur_state[0].y = 0.0
    self.cur_state[0].psi = 0.0
    self.cur_state[0].delta = 0.0

    self.last_mpc_ts = 0.0
    self.angle_steers_des = 0.0
    self.angle_steers_des_mpc = 0.0
    self.starting_angle_steers = 0.0
    self.avg_angle_rate = 0.0
    self.angle_rate_count = 0.0

  def reset(self):
    self.pid.reset()

  def update(self, active, v_ego, angle_steers, angle_rate, steer_override, d_poly, angle_offset, VM, PL):
    self.mpc_updated = False
    # TODO: this creates issues in replay when rewinding time: mpc won't run
    if self.last_mpc_ts < PL.last_md_ts:
      self.last_mpc_ts = PL.last_md_ts
      self.starting_angle_steers = angle_steers
      self.avg_angle_rate = 0.0
      self.angle_rate_count = 0.0

      curvature_factor = VM.curvature_factor(v_ego)

      l_poly = libmpc_py.ffi.new("double[4]", list(PL.PP.l_poly))
      r_poly = libmpc_py.ffi.new("double[4]", list(PL.PP.r_poly))
      p_poly = libmpc_py.ffi.new("double[4]", list(PL.PP.p_poly))

      # account for actuation delay
      self.cur_state = calc_states_after_delay(self.cur_state, v_ego, angle_steers, curvature_factor, VM.CP.steerRatio, VM.CP.steerActuatorDelay)

      v_ego_mpc = max(v_ego, 5.0)  # avoid mpc roughness due to low speed
      self.libmpc.run_mpc(self.cur_state, self.mpc_solution,
                          l_poly, r_poly, p_poly,
                          PL.PP.l_prob, PL.PP.r_prob, PL.PP.p_prob, curvature_factor, v_ego_mpc, PL.PP.lane_width)

      # reset to current steer angle if not active or overriding
      if active:
        delta_desired = self.mpc_solution[0].delta[1]
      else:
        delta_desired = math.radians(angle_steers - angle_offset) / VM.CP.steerRatio

      self.cur_state[0].delta = delta_desired

      self.angle_steers_des_mpc = float(math.degrees(delta_desired * VM.CP.steerRatio) + angle_offset)
      self.mpc_updated = True

      #  Check for infeasable MPC solution
      self.mpc_nans = np.any(np.isnan(list(self.mpc_solution[0].delta)))
      t = sec_since_boot()
      if self.mpc_nans:
        self.libmpc.init(MPC_COST_LAT.PATH, MPC_COST_LAT.LANE, MPC_COST_LAT.HEADING, VM.CP.steerRateCost)
        self.cur_state[0].delta = math.radians(angle_steers) / VM.CP.steerRatio

        if t > self.last_cloudlog_t + 5.0:
          self.last_cloudlog_t = t
          cloudlog.warning("Lateral mpc - nan: True")

    if v_ego < 0.3 or not active:
      output_steer = 0.0
      self.pid.reset()
    else:
      # Project future steering angle using average steer rate since last MPC update
      self.avg_angle_rate = (self.avg_angle_rate * self.angle_rate_count + angle_rate) / (self.angle_rate_count + 1.) 
      self.angle_rate_count += 1.0
      future_angle_steers = (self.avg_angle_rate * _DT_MPC) + self.starting_angle_steers
      steers_max = get_steer_max(VM.CP, v_ego)
      self.pid.pos_limit = steers_max
      self.pid.neg_limit = -steers_max
      if VM.CP.steerControlType == car.CarParams.SteerControlType.torque:
        steer_feedforward = apply_deadzone(self.angle_steers_des_mpc - float(angle_offset), 0.5) 
        steer_feedforward *= v_ego**2  # proportional to realigning tire momentum (~ lateral accel)
      else:
        steer_feedforward = self.angle_steers_des_mpc   # feedforward desired angle
      deadzone = 0.0

      output_steer = self.pid.update(self.angle_steers_des_mpc, future_angle_steers, check_saturation=(v_ego > 10), override=steer_override,
                                     feedforward=steer_feedforward, speed=v_ego, deadzone=deadzone)
    self.sat_flag = self.pid.saturated
    return output_steer, float(self.angle_steers_des_mpc)
