import os
import math
import numpy as np

from common.realtime import sec_since_boot
from selfdrive.services import service_list
from selfdrive.swaglog import cloudlog
from selfdrive.controls.lib.lateral_mpc import libmpc_py
from selfdrive.controls.lib.drive_helpers import MPC_COST_LAT
from selfdrive.controls.lib.model_parser import ModelParser
import selfdrive.messaging as messaging

LOG_MPC = os.environ.get('LOG_MPC', False)

def calc_states_after_delay(states, v_ego, steer_angle, curvature_factor, steer_ratio, delay):
  states[0].x = v_ego * delay
  states[0].psi = v_ego * curvature_factor * math.radians(steer_angle) / steer_ratio * delay
  states[0].delta = math.radians(steer_angle) / steer_ratio
  return states


class PathPlanner(object):
  def __init__(self, CP):
    self.MP = ModelParser()

    self.l_poly = libmpc_py.ffi.new("double[4]")
    self.r_poly = libmpc_py.ffi.new("double[4]")
    self.p_poly = libmpc_py.ffi.new("double[4]")

    self.last_cloudlog_t = 0

    self.plan = messaging.pub_sock(service_list['pathPlan'].port)
    self.livempc = messaging.pub_sock(service_list['liveMpc'].port)

    self.setup_mpc(CP.steerRateCost)
    self.solution_invalid_cnt = 0

  def setup_mpc(self, steer_rate_cost):
    self.libmpc = libmpc_py.libmpc
    self.libmpc.init(MPC_COST_LAT.PATH, MPC_COST_LAT.LANE, MPC_COST_LAT.HEADING, steer_rate_cost)

    self.mpc_solution = libmpc_py.ffi.new("log_t *")
    self.cur_state = libmpc_py.ffi.new("state_t *")
    self.cur_state[0].x = 0.0
    self.cur_state[0].y = 0.0
    self.cur_state[0].psi = 0.0
    self.cur_state[0].delta = 0.0
    self.mpc_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    self.mpc_rates = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    self.mpc_times = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    self.mpc_probs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    self.angle_steers_des = 0.0
    self.angle_steers_des_mpc = 0.0
    self.angle_steers_des_prev = 0.0
    self.angle_steers_des_time = 0.0

  def update(self, sm, CP, VM):
    v_ego = sm['carState'].vEgo
    angle_steers = sm['carState'].steeringAngle
    active = sm['controlsState'].active  
    cur_time = sec_since_boot()
    angle_offset_average = sm['liveParameters'].angleOffsetAverage
    angle_offset_bias = -sm['controlsState'].angleModelBias + angle_offset_average

    self.MP.update(v_ego, sm['model'])

    # Run MPC
    self.angle_steers_des_prev = self.angle_steers_des_mpc
    VM.update_params(sm['liveParameters'].stiffnessFactor, sm['liveParameters'].steerRatio)
    curvature_factor = VM.curvature_factor(v_ego)
    self.l_poly = list(self.MP.l_poly)
    self.r_poly = list(self.MP.r_poly)
    self.p_poly = list(self.MP.p_poly)

    # account for actuation delay
    self.cur_state = calc_states_after_delay(self.cur_state, v_ego, angle_steers - angle_offset_bias, curvature_factor, VM.sR, CP.steerActuatorDelay)
    self.angle_steers_des_prev = np.interp(cur_time, self.mpc_times, self.mpc_angles)

    v_ego_mpc = max(v_ego, 5.0)  # avoid mpc roughness due to low speed
    self.libmpc.run_mpc(self.cur_state, self.mpc_solution,
                        self.l_poly, self.r_poly, self.p_poly,
                        self.MP.l_prob, self.MP.r_prob, self.MP.p_prob, curvature_factor, v_ego_mpc, self.MP.lane_width)

    #  Check for infeasable MPC solution
    mpc_nans = np.any(np.isnan(list(self.mpc_solution[0].delta)))

    if not mpc_nans:
      self.mpc_angles[0] = angle_steers
      self.mpc_times[0] = sm.logMonoTime['model'] * 1e-9
      oversample_limit = 19 if v_ego == 0 else 4 + min(15, int(400.0 / v_ego))
      for i in range(1,20):
        if i < 6:
          self.mpc_times[i] = self.mpc_times[i-1] + 0.05
          self.mpc_rates[i-1] = (float(math.degrees(self.mpc_solution[0].rate[i-1] * VM.sR)) * self.MP.c_prob \
                                + self.mpc_rates[i] * self.mpc_probs[i]) / (self.MP.c_prob + self.mpc_probs[i] + 0.0001)
          self.mpc_probs[i-1] = (self.MP.c_prob**2 + self.mpc_probs[i]**2) / (self.MP.c_prob + self.mpc_probs[i] + 0.0001)
        elif i <= oversample_limit:
          self.mpc_times[i] = self.mpc_times[i-1] + 0.15
          self.mpc_rates[i-1] = (float(math.degrees(self.mpc_solution[0].rate[i-1] * VM.sR)) * self.MP.c_prob \
                      + 0.33 * self.mpc_rates[i] * self.mpc_probs[i] \
                      + 0.66 * self.mpc_rates[i-1] * self.mpc_probs[i-1]) \
                      / (self.MP.c_prob + 0.66 * self.mpc_probs[i-1] + 0.33 * self.mpc_probs[i] + 0.0001)
          self.mpc_probs[i-1] = (self.MP.c_prob**2 + 0.33 * self.mpc_probs[i]**2 + 0.66 * self.mpc_probs[i-1]**2) \
                      / (self.MP.c_prob + 0.66 * self.mpc_probs[i-1] + 0.33 * self.mpc_probs[i] + 0.0001)
        else:
          self.mpc_times[i] = self.mpc_times[i-1] + 0.15
          self.mpc_rates[i-1] = float(math.degrees(self.mpc_solution[0].rate[i-1] * VM.sR))
          self.mpc_probs[i-1] = self.MP.c_prob
        self.mpc_angles[i] = (self.mpc_times[i] - self.mpc_times[i-1]) * self.mpc_rates[i-1] + self.mpc_angles[i-1]

      rate_desired = math.degrees(self.mpc_solution[0].rate[0] * VM.sR)
      self.angle_steers_des_mpc = self.mpc_angles[1]

    else:
      self.libmpc.init(MPC_COST_LAT.PATH, MPC_COST_LAT.LANE, MPC_COST_LAT.HEADING, CP.steerRateCost)

      if cur_time > self.last_cloudlog_t + 5.0:
        self.last_cloudlog_t = cur_time
        cloudlog.warning("Lateral mpc - nan: True")

    if self.mpc_solution[0].cost > 20000. or mpc_nans:   # TODO: find a better way to detect when MPC did not converge
      self.solution_invalid_cnt += 1
    else:
      self.solution_invalid_cnt = 0
    plan_solution_valid = self.solution_invalid_cnt < 2

    plan_send = messaging.new_message()
    plan_send.init('pathPlan')
    plan_send.valid = sm.all_alive_and_valid(service_list=['carState', 'controlsState', 'liveParameters', 'model'])
    plan_send.pathPlan.laneWidth = float(self.MP.lane_width)
    plan_send.pathPlan.dPoly = [float(x) for x in self.MP.d_poly]
    plan_send.pathPlan.cPoly = [float(x) for x in self.MP.c_poly]
    plan_send.pathPlan.cProb = float(self.MP.c_prob)
    plan_send.pathPlan.lPoly = [float(x) for x in self.l_poly]
    plan_send.pathPlan.lProb = float(self.MP.l_prob)
    plan_send.pathPlan.rPoly = [float(x) for x in self.r_poly]
    plan_send.pathPlan.rProb = float(self.MP.r_prob)
    plan_send.pathPlan.angleSteers = float(self.angle_steers_des_mpc)
    plan_send.pathPlan.rateSteers = float(rate_desired)
    plan_send.pathPlan.angleOffset = float(angle_offset_average)
    plan_send.pathPlan.mpcAngles = [float(x) for x in self.mpc_angles]
    plan_send.pathPlan.mpcTimes = [float(x) for x in self.mpc_times]
    plan_send.pathPlan.mpcRates = [float(x) for x in self.mpc_rates]
    plan_send.pathPlan.mpcSolutionValid = bool(plan_solution_valid)
    plan_send.pathPlan.paramsValid = bool(sm['liveParameters'].valid)
    plan_send.pathPlan.sensorValid = bool(sm['liveParameters'].sensorValid)

    self.plan.send(plan_send.to_bytes())

    if LOG_MPC:
      dat = messaging.new_message()
      dat.init('liveMpc')
      dat.liveMpc.x = list(self.mpc_solution[0].x)
      dat.liveMpc.y = list(self.mpc_solution[0].y)
      dat.liveMpc.psi = list(self.mpc_solution[0].psi)
      dat.liveMpc.delta = list(self.mpc_solution[0].delta)
      dat.liveMpc.cost = self.mpc_solution[0].cost
      self.livempc.send(dat.to_bytes())
