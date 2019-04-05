#!/usr/bin/env python
import zmq
import math
import numpy as np
from common.params import Params
from common.numpy_fast import interp

import selfdrive.messaging as messaging
from selfdrive.swaglog import cloudlog
from selfdrive.config import Conversions as CV
from selfdrive.services import service_list
from selfdrive.controls.lib.drive_helpers import create_event, EventTypes as ET
from selfdrive.controls.lib.speed_smoother import speed_smoother
from selfdrive.controls.lib.longcontrol import LongCtrlState, MIN_CAN_SPEED
from selfdrive.controls.lib.fcw import FCWChecker
from selfdrive.controls.lib.long_mpc import LongitudinalMpc

NO_CURVATURE_SPEED = 200. * CV.MPH_TO_MS

_DT_MPC = 0.2  # 5Hz
MAX_SPEED_ERROR = 2.0
AWARENESS_DECEL = -0.2     # car smoothly decel at .2m/s^2 when user is distracted

# lookup tables VS speed to determine min and max accels in cruise
# make sure these accelerations are smaller than mpc limits
_A_CRUISE_MIN_V  = [-1.0, -.8, -.67, -.5, -.30]
_A_CRUISE_MIN_BP = [   0., 5.,  10., 20.,  40.]

# need fast accel at very low speed for stop and go
# make sure these accelerations are smaller than mpc limits
#_A_CRUISE_MAX_V = [1.1, 1.1, .8, .5, .3] comma default 
#_A_CRUISE_MAX_V = [1.6, 1.6, 1.2, .7, .3] kegman
_A_CRUISE_MAX_V = [1.6, 1.6, 1.5, .7, .3] #better (regain speed faster)
_A_CRUISE_MAX_V_FOLLOWING = [1.6, 1.6, 1.2, .7, .3] #comma default
#_A_CRUISE_MAX_V_FOLLOWING = [1.1, 1.6, 1.3, .7, .3] #better (less agressive accel on jams)
_A_CRUISE_MAX_BP = [0.,  5., 10., 20., 40.]

# Lookup table for turns
_A_TOTAL_MAX_V = [1.5, 1.9, 3.2]
_A_TOTAL_MAX_BP = [0., 20., 40.]


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

  a_target[1] = min(a_target[1], a_x_allowed)
  return a_target


class Planner(object):
  def __init__(self, CP, fcw_enabled):
    context = zmq.Context()
    self.CP = CP
    self.poller = zmq.Poller()

    self.plan = messaging.pub_sock(context, service_list['plan'].port)
    self.live_longitudinal_mpc = messaging.pub_sock(context, service_list['liveLongitudinalMpc'].port)

    self.mpc1 = LongitudinalMpc(1, self.live_longitudinal_mpc)
    self.mpc2 = LongitudinalMpc(2, self.live_longitudinal_mpc)

    self.v_acc_start = 0.0
    self.a_acc_start = 0.0

    self.v_acc = 0.0
    self.v_acc_future = 0.0
    self.a_acc = 0.0
    self.v_cruise = 0.0
    self.a_cruise = 0.0

    self.longitudinalPlanSource = 'cruise'
    self.fcw_checker = FCWChecker()
    self.fcw_enabled = fcw_enabled

    self.params = Params()

  def choose_solution(self, v_cruise_setpoint, enabled):
    if enabled:
      solutions = {'cruise': self.v_cruise}
      if self.mpc1.prev_lead_status:
        solutions['mpc1'] = self.mpc1.v_mpc
      if self.mpc2.prev_lead_status:
        solutions['mpc2'] = self.mpc2.v_mpc

      slowest = min(solutions, key=solutions.get)

      self.longitudinalPlanSource = slowest

      # Choose lowest of MPC and cruise
      if slowest == 'mpc1':
        self.v_acc = self.mpc1.v_mpc
        self.a_acc = self.mpc1.a_mpc
      elif slowest == 'mpc2':
        self.v_acc = self.mpc2.v_mpc
        self.a_acc = self.mpc2.a_mpc
      elif slowest == 'cruise':
        self.v_acc = self.v_cruise
        self.a_acc = self.a_cruise

    self.v_acc_future = min([self.mpc1.v_mpc_future, self.mpc2.v_mpc_future, v_cruise_setpoint])

  def update(self, CS, CP, VM, PP, live20, live100, md, live_map_data):
    """Gets called when new live20 is available"""
    cur_time = live20.logMonoTime / 1e9
    v_ego = CS.carState.vEgo

    long_control_state = live100.live100.longControlState
    v_cruise_kph = live100.live100.vCruise
    force_slow_decel = live100.live100.forceDecel
    v_cruise_setpoint = v_cruise_kph * CV.KPH_TO_MS

    lead_1 = live20.live20.leadOne
    lead_2 = live20.live20.leadTwo

    enabled = (long_control_state == LongCtrlState.pid) or (long_control_state == LongCtrlState.stopping)
    following = lead_1.status and lead_1.dRel < 45.0 and lead_1.vLeadK > v_ego and lead_1.aLeadK > 0.0

    v_speedlimit = NO_CURVATURE_SPEED
    v_curvature = NO_CURVATURE_SPEED
    map_valid = live_map_data.liveMapData.mapValid

    # Speed limit and curvature
    set_speed_limit_active = self.params.get("LimitSetSpeed") == "1" and self.params.get("SpeedLimitOffset") is not None
    if set_speed_limit_active:
      if live_map_data.liveMapData.speedLimitValid:
        speed_limit = live_map_data.liveMapData.speedLimit
        offset = float(self.params.get("SpeedLimitOffset"))
        v_speedlimit = speed_limit + offset

      if live_map_data.liveMapData.curvatureValid:
        curvature = abs(live_map_data.liveMapData.curvature)
        a_y_max = 2.975 - v_ego * 0.0375  # ~1.85 @ 75mph, ~2.6 @ 25mph
        v_curvature = math.sqrt(a_y_max / max(1e-4, curvature))
        v_curvature = min(NO_CURVATURE_SPEED, v_curvature)

    decel_for_turn = bool(v_curvature < min([v_cruise_setpoint, v_speedlimit, v_ego + 1.]))
    v_cruise_setpoint = min([v_cruise_setpoint, v_curvature, v_speedlimit])

    # Calculate speed for normal cruise control
    if enabled:
      accel_limits = map(float, calc_cruise_accel_limits(v_ego, following))
      jerk_limits = [min(-0.1, accel_limits[0]), max(0.1, accel_limits[1])]  # TODO: make a separate lookup for jerk tuning
      accel_limits = limit_accel_in_turns(v_ego, CS.carState.steeringAngle, accel_limits, self.CP)

      if force_slow_decel:
        # if required so, force a smooth deceleration
        accel_limits[1] = min(accel_limits[1], AWARENESS_DECEL)
        accel_limits[0] = min(accel_limits[0], accel_limits[1])

      # Change accel limits based on time remaining to turn
      if decel_for_turn:
        time_to_turn = max(1.0, live_map_data.liveMapData.distToTurn / max(self.v_cruise, 1.))
        required_decel = min(0, (v_curvature - self.v_cruise) / time_to_turn)
        accel_limits[0] = max(accel_limits[0], required_decel)

      self.v_cruise, self.a_cruise = speed_smoother(self.v_acc_start, self.a_acc_start,
                                                    v_cruise_setpoint,
                                                    accel_limits[1], accel_limits[0],
                                                    jerk_limits[1], jerk_limits[0],
                                                    _DT_MPC)
      # cruise speed can't be negative even is user is distracted
      self.v_cruise = max(self.v_cruise, 0.)
    else:
      starting = long_control_state == LongCtrlState.starting
      a_ego = min(CS.carState.aEgo, 0.0)
      reset_speed = MIN_CAN_SPEED if starting else v_ego
      reset_accel = self.CP.startAccel if starting else a_ego
      self.v_acc = reset_speed
      self.a_acc = reset_accel
      self.v_acc_start = reset_speed
      self.a_acc_start = reset_accel
      self.v_cruise = reset_speed
      self.a_cruise = reset_accel

    self.mpc1.set_cur_state(self.v_acc_start, self.a_acc_start)
    self.mpc2.set_cur_state(self.v_acc_start, self.a_acc_start)

    self.mpc1.update(CS, lead_1, v_cruise_setpoint)
    self.mpc2.update(CS, lead_2, v_cruise_setpoint)

    self.choose_solution(v_cruise_setpoint, enabled)

    # determine fcw
    if self.mpc1.new_lead:
      self.fcw_checker.reset_lead(cur_time)

    blinkers = CS.carState.leftBlinker or CS.carState.rightBlinker
    fcw = self.fcw_checker.update(self.mpc1.mpc_solution, cur_time, v_ego, CS.carState.aEgo,
                                  lead_1.dRel, lead_1.vLead, lead_1.aLeadK,
                                  lead_1.yRel, lead_1.vLat,
                                  lead_1.fcw, blinkers) and not CS.carState.brakePressed
    if fcw:
      cloudlog.info("FCW triggered %s", self.fcw_checker.counters)

    model_dead = cur_time - (md.logMonoTime / 1e9) > 0.5

    # **** send the plan ****
    plan_send = messaging.new_message()
    plan_send.init('plan')

    # TODO: Move all these events to controlsd. This has nothing to do with planning
    events = []
    if model_dead:
      events.append(create_event('modelCommIssue', [ET.NO_ENTRY, ET.SOFT_DISABLE]))

    radar_errors = list(live20.live20.radarErrors)
    if 'commIssue' in radar_errors:
      events.append(create_event('radarCommIssue', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
    if 'fault' in radar_errors:
      events.append(create_event('radarFault', [ET.NO_ENTRY, ET.SOFT_DISABLE]))

    plan_send.plan.events = events
    plan_send.plan.mdMonoTime = md.logMonoTime
    plan_send.plan.l20MonoTime = live20.logMonoTime

    # longitudal plan
    plan_send.plan.vCruise = self.v_cruise
    plan_send.plan.aCruise = self.a_cruise
    plan_send.plan.vStart = self.v_acc_start
    plan_send.plan.aStart = self.a_acc_start
    plan_send.plan.vTarget = self.v_acc
    plan_send.plan.aTarget = self.a_acc
    plan_send.plan.vTargetFuture = self.v_acc_future
    plan_send.plan.hasLead = self.mpc1.prev_lead_status
    plan_send.plan.longitudinalPlanSource = self.longitudinalPlanSource

    plan_send.plan.vCurvature = v_curvature
    plan_send.plan.decelForTurn = decel_for_turn
    plan_send.plan.mapValid = map_valid

    # Send out fcw
    fcw = fcw and (self.fcw_enabled or long_control_state != LongCtrlState.off)
    plan_send.plan.fcw = fcw

    self.plan.send(plan_send.to_bytes())

    # Interpolate 0.05 seconds and save as starting point for next iteration
    dt = 0.05  # s
    a_acc_sol = self.a_acc_start + (dt / _DT_MPC) * (self.a_acc - self.a_acc_start)
    v_acc_sol = self.v_acc_start + dt * (a_acc_sol + self.a_acc_start) / 2.0
    self.v_acc_start = v_acc_sol
    self.a_acc_start = a_acc_sol
