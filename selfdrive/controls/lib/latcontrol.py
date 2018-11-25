import zmq
import math
import numpy as np
import time
import json
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
    self.ff_smoothing = 3. * _DT_MPC / _DT

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
    self.steer_rate_cost = steer_rate_cost

    self.last_mpc_ts = 0.0
    self.angle_steers_des = 0.0
    self.angle_steers_des_mpc = 0.0
    self.angle_steers_des_prev = 0.0
    self.angle_steers_des_time = 0.0
    self.context = zmq.Context()
    self.steerpub = self.context.socket(zmq.PUB)
    self.steerpub.bind("tcp://*:8594")
    self.steerdata = ""
    self.steerpub2 = self.context.socket(zmq.PUB)
    self.steerpub2.bind("tcp://*:8596")
    self.steerdata2 = ""
    self.ratioExp = 2.6
    self.ratioScale = 0.
    self.steer_steps = [0., 0., 0., 0., 0.]
    self.probFactor = 0.
    self.prev_output_steer = 0.
    self.rough_angle_array = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
    self.steer_speed_array = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
    self.tiny_angle_array = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    self.steer_torque_array = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
    self.steer_torque_count = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    self.tiny_torque_array = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    self.tiny_torque_count = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    self.center_angle = 0.
    self.center_count = 0.
    self.save_steering = False
    self.steer_zero_crossing = 0.0
    self.steer_initialized = False
    self.avg_angle_steers = 0.0
    self.feed_forward_rate = 0.0
    self.feed_forward_angle = 0.0
    self.feed_forward = 0.0
    self.angle_rate_desired = 0.0
    self.pCost = 0.0
    self.lCost = 0.0
    self.rCost = 0.0
    self.hCost = 0.0
    self.srCost = 0.0
    self.frame = 0
    self.angle_steer_des_step = 0.0
    self.last_ff_a = 0.0
    self.last_ff_r = 0.0
    self.ff_angle_factor = 1.0
    self.ff_rate_factor = 1.0

  def reset(self):
    self.pid.reset()

  def update(self, active, v_ego, angle_steers, angle_rate, steer_override, d_poly, angle_offset, CP, VM, PL):
    self.mpc_updated = False

    enable_enhancements = True
    self.frame += 1

    cur_Steer_Ratio = CP.steerRatio * 1.

    # TODO: this creates issues in replay when rewinding time: mpc won't run
    if self.last_mpc_ts < PL.last_md_ts:
      self.last_mpc_ts = PL.last_md_ts
      self.starting_angle_steers = angle_steers
      self.avg_angle_rate = 0.
      self.angle_rate_count = 0

      curvature_factor = VM.curvature_factor(v_ego)

      self.l_poly = libmpc_py.ffi.new("double[4]", list(PL.PP.l_poly))
      self.r_poly = libmpc_py.ffi.new("double[4]", list(PL.PP.r_poly))
      self.p_poly = libmpc_py.ffi.new("double[4]", list(PL.PP.p_poly))

      # account for actuation delay
      self.cur_state = calc_states_after_delay(self.cur_state, v_ego, angle_steers, curvature_factor, cur_Steer_Ratio, CP.steerActuatorDelay) 

      v_ego_mpc = max(v_ego, 5.0)  # avoid mpc roughness due to low speed
      self.libmpc.run_mpc(self.cur_state, self.mpc_solution,
                          self.l_poly, self.r_poly, self.p_poly,
                          PL.PP.l_prob, PL.PP.r_prob, PL.PP.p_prob, curvature_factor, v_ego_mpc, PL.PP.lane_width)


      # reset to current steer angle if not active or overriding
      if active:
        self.isActive = 1
        delta_desired = self.mpc_solution[0].delta[1]
      else:
        self.isActive = 0
        delta_desired = math.radians(angle_steers - angle_offset) / cur_Steer_Ratio

      self.cur_state[0].delta = delta_desired

      if self.angle_steers_des == 0.0 and self.angle_steer_des_step == 0.0:
        self.angle_steers_des = angle_steers
      self.angle_steers_des_mpc = float(math.degrees(delta_desired * cur_Steer_Ratio) + angle_offset)
      self.angle_steer_des_step = _DT * (self.angle_steers_des_mpc - self.angle_steers_des) / _DT_MPC
      self.feed_forward_rate = self.ff_rate_factor * (self.angle_steers_des_mpc - angle_steers) / _DT_MPC
      self.feed_forward_angle = self.ff_angle_factor * self.angle_steers_des_mpc - float(angle_offset)
     
      self.mpc_updated = True

      #  Check for infeasable MPC solution
      self.mpc_nans = np.any(np.isnan(list(self.mpc_solution[0].delta)))
      t = sec_since_boot()
      if self.mpc_nans:
        self.libmpc.init(MPC_COST_LAT.PATH, MPC_COST_LAT.LANE, MPC_COST_LAT.HEADING, CP.steerRateCost)
        self.cur_state[0].delta = math.radians(angle_steers) / cur_Steer_Ratio

        if t > self.last_cloudlog_t + 5.0:
          self.last_cloudlog_t = t
          cloudlog.warning("Lateral mpc - nan: True")

    if self.steerdata != "" and (self.frame % 50) == 3:
      self.steerpub.send(self.steerdata)
      self.steerdata = ""

    if v_ego < 0.3 or not active:
      output_steer = 0.0
      self.angle_steer_des_step = 0.0
      self.angle_steers_des = 0.0
      self.feed_forward_angle = 0.0
      self.feed_forward_rate = 0.0
      self.feed_forward = 0.0
      self.last_ff_a = 0.0
      self.last_ff_r = 0.0
      self.pid.reset()
    else:
      self.angle_steers_des += self.angle_steer_des_step
      future_angle_steers = angle_steers + (angle_rate * _DT)

      steers_max = get_steer_max(CP, v_ego)
      self.pid.pos_limit = steers_max
      self.pid.neg_limit = -steers_max
      ff_type = ""
      if CP.steerControlType == car.CarParams.SteerControlType.torque:
        if (abs(self.feed_forward_angle) - 0.5 > abs(self.feed_forward_rate)) or ((self.feed_forward_rate < 0) != (self.prev_output_steer < 0)):
          ff_type = "a"
          self.feed_forward = (((self.ff_smoothing - 1.0) * self.feed_forward) + (v_ego**2 * self.feed_forward_angle)) / self.ff_smoothing 
          if self.last_ff_a == 0.0:
            self.last_ff_r = 0.0
            self.last_ff_a = sec_since_boot() + 1.0
        else:
          ff_type = "r"
          self.feed_forward = (((self.ff_smoothing - 1.0) * self.feed_forward) + (v_ego**2 * self.feed_forward_rate)) / self.ff_smoothing      
          if self.last_ff_r == 0.0:
            self.last_ff_a = 0.0
            self.last_ff_r = sec_since_boot() + 1.0
      else:
        self.feed_forward = self.angle_steers_des_mpc   # feedforward desired angle
      deadzone = 0.0

      if abs(angle_steers) > 3.0:
        self.last_ff_a = 0.0
        self.last_ff_r = 0.0
      elif ff_type == "r" and self.last_ff_r > 0.0 and sec_since_boot() > self.last_ff_r:
        if (self.pid.p > 0) == (self.feed_forward > 0):
          self.ff_rate_factor *= 1.001
        else:
          self.ff_rate_factor *= 0.999    
        self.last_ff_r = sec_since_boot() + 1.0
      elif ff_type == "a" and self.last_ff_a > 0.0 and sec_since_boot() > self.last_ff_a:
        if (self.pid.p > 0) == (self.feed_forward > 0):
          self.ff_angle_factor *= 1.001
        else:
          self.ff_angle_factor *= 0.999    
        self.last_ff_a = sec_since_boot() + 1.0
            
      output_steer =  self.pid.update(self.angle_steers_des, future_angle_steers, check_saturation=False, override=steer_override,
                                     feedforward=self.feed_forward, speed=v_ego, deadzone=deadzone)

      if not steer_override and v_ego > 10. and abs(angle_steers) <= 10:
        self.steerdata += ("%d,%s,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%d|" % (self.isActive, \
        ff_type, self.ff_angle_factor, self.ff_rate_factor, self.pCost, self.lCost, self.rCost, self.hCost, self.srCost, future_angle_steers, angle_rate, self.steer_zero_crossing, self.center_angle, angle_steers, self.angle_steers_des, angle_offset, \
        self.angle_steers_des_mpc, cur_Steer_Ratio, CP.steerKf, CP.steerKpV[0], CP.steerKiV[0], CP.steerRateCost, PL.PP.l_prob, \
        PL.PP.r_prob, PL.PP.c_prob, PL.PP.p_prob, self.l_poly[0], self.l_poly[1], self.l_poly[2], self.l_poly[3], self.r_poly[0], self.r_poly[1], self.r_poly[2], self.r_poly[3], \
        self.p_poly[0], self.p_poly[1], self.p_poly[2], self.p_poly[3], PL.PP.c_poly[0], PL.PP.c_poly[1], PL.PP.c_poly[2], PL.PP.c_poly[3], PL.PP.d_poly[0], PL.PP.d_poly[1], \
        PL.PP.d_poly[2], PL.PP.lane_width, PL.PP.lane_width_estimate, PL.PP.lane_width_certainty, v_ego, self.pid.p, self.pid.i, self.pid.f, int(time.time() * 100) * 10000000))

    self.sat_flag = self.pid.saturated
    self.prev_output_steer = output_steer
    return output_steer, float(self.angle_steers_des_mpc)
