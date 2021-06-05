#!/usr/bin/env python3
import math
import numpy as np
from common.numpy_fast import interp

import cereal.messaging as messaging
from common.realtime import DT_MDL
from selfdrive.swaglog import cloudlog
from selfdrive.config import Conversions as CV
from selfdrive.controls.lib.longcontrol import LongCtrlState
from selfdrive.controls.lib.long_mpc import LongitudinalMpc
from selfdrive.controls.lib.long_mpc_model import LongitudinalMpcModel
from selfdrive.controls.lib.drive_helpers import V_CRUISE_MAX, MPC_N

LON_MPC_STEP = 0.2  # first step is 0.2s
AWARENESS_DECEL = -0.2     # car smoothly decel at .2m/s^2 when user is distracted

# lookup tables VS speed to determine min and max accels in cruise
# make sure these accelerations are smaller than mpc limits
_A_CRUISE_MIN_V = [-1.0, -1.0, -1.0, -1.0, -1.0]
_A_CRUISE_MIN_BP = [  0.,  5.,  10., 20.,  40.]

# need fast accel at very low speed for stop and go
# make sure these accelerations are smaller than mpc limits
_A_CRUISE_MAX_V = [1.2, 1.2, 0.80, .80]
_A_CRUISE_MAX_V_FOLLOWING = [1.6, 1.6, 0.80, .80]
_A_CRUISE_MAX_BP = [0.,  6.4, 22.5, 40.]

# Lookup table for turns
_A_TOTAL_MAX_V = [1.7, 3.2]
_A_TOTAL_MAX_BP = [20., 40.]


def calc_cruise_accel_limits(v_ego, following):
  a_cruise_min = interp(v_ego, _A_CRUISE_MIN_BP, _A_CRUISE_MIN_V)

  if following:
    a_cruise_max = interp(v_ego, _A_CRUISE_MAX_BP, _A_CRUISE_MAX_V_FOLLOWING)
  else:
    a_cruise_max = interp(v_ego, _A_CRUISE_MAX_BP, _A_CRUISE_MAX_V)
  return np.vstack([a_cruise_min, a_cruise_max])


def limit_accel_in_turns(v_ego, angle_steers, a_target, CP):
  """
  This function returns a limited long acceleration allowed, depending on the existing lateral acceleration
  this should avoid accelerating when losing the target in turns
  """

  a_total_max = interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
  a_y = v_ego**2 * angle_steers * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)
  a_x_allowed = math.sqrt(max(a_total_max**2 - a_y**2, 0.))

  return [a_target[0], min(a_target[1], a_x_allowed)]


class Planner():
  def __init__(self, CP):
    self.CP = CP

    self.lead_mpc1 = LongitudinalMpc(1)
    self.lead_mpc2 = LongitudinalMpc(2)
    self.cruise_mpc = LongitudinalMpcModel()
    self.fcw = False

    self.v_desired = 0.0
    self.a_desired = 0.0
    self.longitudinalPlanSource = 'cruise'
    self.alpha = np.exp(-DT_MDL/2.0)


  def update(self, sm, CP):
    v_ego = sm['carState'].vEgo
    a_ego = sm['carState'].aEgo

    # Prevent divergence, smooth in current v_ego
    self.v_desired = self.alpha * self.v_desired + (1 - self.alpha) * v_ego

    v_cruise_kph = sm['controlsState'].vCruise
    v_cruise_kph = min(v_cruise_kph, V_CRUISE_MAX)
    v_cruise = v_cruise_kph * CV.KPH_TO_MS

    long_control_state = sm['controlsState'].longControlState
    force_slow_decel = sm['controlsState'].forceDecel


    lead_0 = sm['modelV2'].leads[0]
    lead_1 = sm['modelV2'].leads[1]
    t_idxs = np.array(sm['modelV2'].position.t)
    mpc_t = [0.0, .2, .4, .6, .8] + list(np.arange(1.0, 10.1, .6))

    enabled = (long_control_state == LongCtrlState.pid) or (long_control_state == LongCtrlState.stopping)
    following = lead_0.prob > .5 and lead_0.x[0] < 45.0 and lead_0.v[0] > v_ego and lead_0.a[0] > 0.0

    # Calculate speed for normal cruise control
    if not enabled or sm['carState'].gasPressed:
      starting = long_control_state == LongCtrlState.starting
      self.v_desired = self.CP.minSpeedCan if starting else v_ego
      self.a_desired = self.CP.startAccel if starting else min(0.0, a_ego)

    accel_limits = [float(x) for x in calc_cruise_accel_limits(v_ego, following)]
    accel_limits_turns = limit_accel_in_turns(v_ego, sm['carState'].steeringAngleDeg, accel_limits, self.CP)
    if force_slow_decel:
      # if required so, force a smooth deceleration
      accel_limits_turns[1] = min(accel_limits_turns[1], AWARENESS_DECEL)
      accel_limits_turns[0] = min(accel_limits_turns[0], accel_limits_turns[1])

    self.lead_mpc1.set_cur_state(self.v_desired, self.a_desired)
    self.lead_mpc2.set_cur_state(self.v_desired, self.a_desired)
    self.cruise_mpc.set_cur_state(self.v_desired, self.a_desired)
    self.lead_mpc1.update(sm['carState'], lead_0)
    self.lead_mpc2.update(sm['carState'], lead_1)
    v_cruise_clipped = np.clip(v_cruise, v_ego - 10.0, v_ego + 5.0)
    self.cruise_mpc.update(v_ego, a_ego,
                           v_cruise_clipped * np.arange(0.,10.,1.0),
                           v_cruise_clipped * np.ones(10),
                           np.zeros(10))

    next_a = self.cruise_mpc.mpc_solution.a_ego[1]
    self.longitudinalPlanSource = 'cruise'
    self.a_desired_trajectory = np.interp(t_idxs[:MPC_N+1], mpc_t, list(self.cruise_mpc.mpc_solution.a_ego))
    if self.lead_mpc1.lead_status and self.lead_mpc1.mpc_solution.a_ego[1] < next_a:
      self.longitudinalPlanSource = 'mpc1'
      next_a = self.lead_mpc1.mpc_solution.a_ego[1]
      self.a_desired_trajectory = np.interp(t_idxs[:MPC_N+1], mpc_t, list(self.lead_mpc1.mpc_solution.a_ego))
    if self.lead_mpc2.lead_status and self.lead_mpc2.mpc_solution.a_ego[1] < next_a:
      self.longitudinalPlanSource = 'mpc2'
      next_a = self.lead_mpc2.mpc_solution.a_ego[1]
      self.a_desired_trajectory = np.interp(t_idxs[:MPC_N+1], mpc_t, list(self.lead_mpc2.mpc_solution.a_ego))


    # TODO throw FCW if brake predictions exceed capability
    self.fcw = False
    if self.fcw:
      cloudlog.info("FCW triggered")

    # Interpolate 0.05 seconds and save as starting point for next iteration
    a_prev = self.a_desired
    self.a_desired = np.interp(DT_MDL, t_idxs[:MPC_N+1], self.a_desired_trajectory)
    self.a_desired = np.clip(self.a_desired, accel_limits_turns[0], accel_limits_turns[1])
    self.v_desired = self.v_desired + DT_MDL * (self.a_desired + a_prev)/2.0


  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')

    plan_send.valid = sm.all_alive_and_valid(service_list=['carState', 'controlsState'])

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.mdMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.radarStateMonoTime = 0

    longitudinalPlan.vStart = float(self.v_desired)
    longitudinalPlan.aStart = float(self.a_desired)
    longitudinalPlan.vTargetFuture = float(self.v_desired + self.a_desired*3.0)
    longitudinalPlan.hasLead = self.lead_mpc1.lead_status
    longitudinalPlan.longitudinalPlanSource = self.longitudinalPlanSource
    longitudinalPlan.fcw = self.fcw

    longitudinalPlan.processingDelay = (plan_send.logMonoTime / 1e9) - sm.logMonoTime['modelV2']

    pm.send('longitudinalPlan', plan_send)
