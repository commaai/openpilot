import math
from selfdrive.controls.lib.pid import PIController
from selfdrive.controls.lib.lateral_mpc import libmpc_py
from common.numpy_fast import clip, interp


# 100ms is a rule of thumb estimation of lag from image processing to actuator command
ACTUATORS_DELAY = 0.1


def calc_states_after_delay(states, v_ego, steer_angle, curvature_factor, steer_ratio):
  states[0].x = v_ego * ACTUATORS_DELAY
  states[0].psi = v_ego * curvature_factor * math.radians(steer_angle) / steer_ratio * ACTUATORS_DELAY
  return states


def get_steer_max(CP, v_ego):
  return interp(v_ego, CP.steerMaxBP, CP.steerMaxV)


class LatControl(object):
  def __init__(self, VM):
    self.pid = PIController(VM.CP.steerKp, VM.CP.steerKi, pos_limit=1.0)
    self.setup_mpc()

    self.y_des = -1  # Legacy

  def setup_mpc(self):
    self.libmpc = libmpc_py.libmpc
    self.libmpc.init()

    self.mpc_solution = libmpc_py.ffi.new("log_t *")
    self.cur_state = libmpc_py.ffi.new("state_t *")
    self.mpc_updated = False
    self.cur_state[0].x = 0.0
    self.cur_state[0].y = 0.0
    self.cur_state[0].psi = 0.0
    self.cur_state[0].delta = 0.0

    self.last_mpc_ts = 0.0
    self.angle_steers_des = 0

  def reset(self):
    self.pid.reset()

  def update(self, active, v_ego, angle_steers, steer_override, d_poly, angle_offset, VM, PL):
    self.mpc_updated = False
    if self.last_mpc_ts + 0.001 < PL.last_md_ts:
      self.last_mpc_ts = PL.last_md_ts

      curvature_factor = VM.curvature_factor(v_ego)

      l_poly = libmpc_py.ffi.new("double[4]", list(PL.PP.l_poly))
      r_poly = libmpc_py.ffi.new("double[4]", list(PL.PP.r_poly))
      p_poly = libmpc_py.ffi.new("double[4]", list(PL.PP.p_poly))

      # account for actuation delay
      self.cur_state = calc_states_after_delay(self.cur_state, v_ego, angle_steers, curvature_factor, VM.CP.sR)

      self.libmpc.run_mpc(self.cur_state, self.mpc_solution,
                          l_poly, r_poly, p_poly,
                          PL.PP.l_prob, PL.PP.r_prob, PL.PP.p_prob, curvature_factor, v_ego, PL.PP.lane_width)

      delta_desired = self.mpc_solution[0].delta[1]
      self.cur_state[0].delta = delta_desired

      self.angle_steers_des = math.degrees(delta_desired * VM.CP.sR) + angle_offset
      self.mpc_updated = True

    if v_ego < 0.3 or not active:
      output_steer = 0.0
      self.pid.reset()
    else:
      steer_max = get_steer_max(VM.CP, v_ego)
      self.pid.pos_limit = steer_max
      self.pid.neg_limit = -steer_max
      output_steer = self.pid.update(self.angle_steers_des, angle_steers, check_saturation=(v_ego > 10), override=steer_override)

    self.sat_flag = self.pid.saturated
    return output_steer
