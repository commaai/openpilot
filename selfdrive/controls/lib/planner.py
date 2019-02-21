#!/usr/bin/env python
import zmq
import math
import numpy as np
from collections import defaultdict
from common.params import Params
from common.realtime import sec_since_boot
from common.numpy_fast import interp
import selfdrive.messaging as messaging
from selfdrive.swaglog import cloudlog
from selfdrive.config import Conversions as CV
from selfdrive.services import service_list
from selfdrive.controls.lib.drive_helpers import create_event, MPC_COST_LONG, EventTypes as ET
from selfdrive.controls.lib.longitudinal_mpc import libmpc_py
from selfdrive.controls.lib.speed_smoother import speed_smoother
from selfdrive.controls.lib.longcontrol import LongCtrlState, MIN_CAN_SPEED
from selfdrive.controls.lib.radar_helpers import _LEAD_ACCEL_TAU

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
_A_CRUISE_MAX_V = [1.1, 1.1, .8, .5, .3]
_A_CRUISE_MAX_V_FOLLOWING = [1.6, 1.6, 1.2, .7, .3]
_A_CRUISE_MAX_BP = [0.,  5., 10., 20., 40.]

# Lookup table for turns
_A_TOTAL_MAX_V = [1.5, 1.9, 3.2]
_A_TOTAL_MAX_BP = [0., 20., 40.]

_FCW_A_ACT_V = [-3., -2.]
_FCW_A_ACT_BP = [0., 30.]


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


class FCWChecker(object):
  def __init__(self):
    self.reset_lead(0.0)

  def reset_lead(self, cur_time):
    self.last_fcw_a = 0.0
    self.v_lead_max = 0.0
    self.lead_seen_t = cur_time
    self.last_fcw_time = 0.0
    self.last_min_a = 0.0

    self.counters = defaultdict(lambda: 0)

  @staticmethod
  def calc_ttc(v_ego, a_ego, x_lead, v_lead, a_lead):
    max_ttc = 5.0

    v_rel = v_ego - v_lead
    a_rel = a_ego - a_lead

    # assuming that closing gap ARel comes from lead vehicle decel,
    # then limit ARel so that v_lead will get to zero in no sooner than t_decel.
    # This helps underweighting ARel when v_lead is close to zero.
    t_decel = 2.
    a_rel = np.minimum(a_rel, v_lead / t_decel)

    # delta of the quadratic equation to solve for ttc
    delta = v_rel**2 + 2 * x_lead * a_rel

    # assign an arbitrary high ttc value if there is no solution to ttc
    if delta < 0.1 or (np.sqrt(delta) + v_rel < 0.1):
      ttc = max_ttc
    else:
      ttc = np.minimum(2 * x_lead / (np.sqrt(delta) + v_rel), max_ttc)
    return ttc

  def update(self, mpc_solution, cur_time, v_ego, a_ego, x_lead, v_lead, a_lead, y_lead, vlat_lead, fcw_lead, blinkers):
    mpc_solution_a = list(mpc_solution[0].a_ego)
    self.last_min_a = min(mpc_solution_a)
    self.v_lead_max = max(self.v_lead_max, v_lead)

    if (fcw_lead > 0.99):
      ttc = self.calc_ttc(v_ego, a_ego, x_lead, v_lead, a_lead)
      self.counters['v_ego'] = self.counters['v_ego'] + 1 if v_ego > 5.0 else 0
      self.counters['ttc'] = self.counters['ttc'] + 1 if ttc < 2.5 else 0
      self.counters['v_lead_max'] = self.counters['v_lead_max'] + 1 if self.v_lead_max > 2.5 else 0
      self.counters['v_ego_lead'] = self.counters['v_ego_lead'] + 1 if v_ego > v_lead else 0
      self.counters['lead_seen'] = self.counters['lead_seen'] + 0.33
      self.counters['y_lead'] = self.counters['y_lead'] + 1 if abs(y_lead) < 1.0 else 0
      self.counters['vlat_lead'] = self.counters['vlat_lead'] + 1 if abs(vlat_lead) < 0.4 else 0
      self.counters['blinkers'] = self.counters['blinkers'] + 10.0 / (20 * 3.0) if not blinkers else 0

      a_thr = interp(v_lead, _FCW_A_ACT_BP, _FCW_A_ACT_V)
      a_delta = min(mpc_solution_a[:15]) - min(0.0, a_ego)

      fcw_allowed = all(c >= 10 for c in self.counters.values())
      if (self.last_min_a < -3.0 or a_delta < a_thr) and fcw_allowed and self.last_fcw_time + 5.0 < cur_time:
        self.last_fcw_time = cur_time
        self.last_fcw_a = self.last_min_a
        return True

    return False


class LongitudinalMpc(object):
  def __init__(self, mpc_id, live_longitudinal_mpc):
    self.live_longitudinal_mpc = live_longitudinal_mpc
    self.mpc_id = mpc_id

    self.setup_mpc()
    self.v_mpc = 0.0
    self.v_mpc_future = 0.0
    self.a_mpc = 0.0
    self.v_cruise = 0.0
    self.prev_lead_status = False
    self.prev_lead_x = 0.0
    self.new_lead = False

    self.last_cloudlog_t = 0.0

  def send_mpc_solution(self, qp_iterations, calculation_time):
    qp_iterations = max(0, qp_iterations)
    dat = messaging.new_message()
    dat.init('liveLongitudinalMpc')
    dat.liveLongitudinalMpc.xEgo = list(self.mpc_solution[0].x_ego)
    dat.liveLongitudinalMpc.vEgo = list(self.mpc_solution[0].v_ego)
    dat.liveLongitudinalMpc.aEgo = list(self.mpc_solution[0].a_ego)
    dat.liveLongitudinalMpc.xLead = list(self.mpc_solution[0].x_l)
    dat.liveLongitudinalMpc.vLead = list(self.mpc_solution[0].v_l)
    dat.liveLongitudinalMpc.cost = self.mpc_solution[0].cost
    dat.liveLongitudinalMpc.aLeadTau = self.a_lead_tau
    dat.liveLongitudinalMpc.qpIterations = qp_iterations
    dat.liveLongitudinalMpc.mpcId = self.mpc_id
    dat.liveLongitudinalMpc.calculationTime = calculation_time
    self.live_longitudinal_mpc.send(dat.to_bytes())

  def setup_mpc(self):
    ffi, self.libmpc = libmpc_py.get_libmpc(self.mpc_id)
    self.libmpc.init(MPC_COST_LONG.TTC, MPC_COST_LONG.DISTANCE,
                     MPC_COST_LONG.ACCELERATION, MPC_COST_LONG.JERK)

    self.mpc_solution = ffi.new("log_t *")
    self.cur_state = ffi.new("state_t *")
    self.cur_state[0].v_ego = 0
    self.cur_state[0].a_ego = 0
    self.a_lead_tau = _LEAD_ACCEL_TAU

  def set_cur_state(self, v, a):
    self.cur_state[0].v_ego = v
    self.cur_state[0].a_ego = a

  def update(self, CS, lead, v_cruise_setpoint):
    v_ego = CS.carState.vEgo

    # Setup current mpc state
    self.cur_state[0].x_ego = 0.0

    if lead is not None and lead.status:
      x_lead = lead.dRel
      v_lead = max(0.0, lead.vLead)
      a_lead = lead.aLeadK

      if (v_lead < 0.1 or -a_lead / 2.0 > v_lead):
        v_lead = 0.0
        a_lead = 0.0

      self.a_lead_tau = max(lead.aLeadTau, (a_lead**2 * math.pi) / (2 * (v_lead + 0.01)**2))
      self.new_lead = False
      if not self.prev_lead_status or abs(x_lead - self.prev_lead_x) > 2.5:
        self.libmpc.init_with_simulation(self.v_mpc, x_lead, v_lead, a_lead, self.a_lead_tau)
        self.new_lead = True

      self.prev_lead_status = True
      self.prev_lead_x = x_lead
      self.cur_state[0].x_l = x_lead
      self.cur_state[0].v_l = v_lead
    else:
      self.prev_lead_status = False
      # Fake a fast lead car, so mpc keeps running
      self.cur_state[0].x_l = 50.0
      self.cur_state[0].v_l = v_ego + 10.0
      a_lead = 0.0
      self.a_lead_tau = _LEAD_ACCEL_TAU

    # Calculate mpc
    t = sec_since_boot()
    n_its = self.libmpc.run_mpc(self.cur_state, self.mpc_solution, self.a_lead_tau, a_lead)
    duration = int((sec_since_boot() - t) * 1e9)
    self.send_mpc_solution(n_its, duration)

    # Get solution. MPC timestep is 0.2 s, so interpolation to 0.05 s is needed
    self.v_mpc = self.mpc_solution[0].v_ego[1]
    self.a_mpc = self.mpc_solution[0].a_ego[1]
    self.v_mpc_future = self.mpc_solution[0].v_ego[10]

    # Reset if NaN or goes through lead car
    dls = np.array(list(self.mpc_solution[0].x_l)) - np.array(list(self.mpc_solution[0].x_ego))
    crashing = min(dls) < -50.0
    nans = np.any(np.isnan(list(self.mpc_solution[0].v_ego)))
    backwards = min(list(self.mpc_solution[0].v_ego)) < -0.01

    if ((backwards or crashing) and self.prev_lead_status) or nans:
      if t > self.last_cloudlog_t + 5.0:
        self.last_cloudlog_t = t
        cloudlog.warning("Longitudinal mpc %d reset - backwards: %s crashing: %s nan: %s" % (
                          self.mpc_id, backwards, crashing, nans))

      self.libmpc.init(MPC_COST_LONG.TTC, MPC_COST_LONG.DISTANCE,
                       MPC_COST_LONG.ACCELERATION, MPC_COST_LONG.JERK)
      self.cur_state[0].v_ego = v_ego
      self.cur_state[0].a_ego = 0.0
      self.v_mpc = v_ego
      self.a_mpc = CS.carState.aEgo
      self.prev_lead_status = False


class Planner(object):
  def __init__(self, CP, fcw_enabled):
    context = zmq.Context()
    self.CP = CP
    self.poller = zmq.Poller()

    self.plan = messaging.pub_sock(context, service_list['plan'].port)
    self.live_longitudinal_mpc = messaging.pub_sock(context, service_list['liveLongitudinalMpc'].port)

    self.radar_errors = []

    self.mpc1 = LongitudinalMpc(1, self.live_longitudinal_mpc)
    self.mpc2 = LongitudinalMpc(2, self.live_longitudinal_mpc)

    self.v_acc_start = 0.0
    self.a_acc_start = 0.0

    self.v_acc = 0.0
    self.v_acc_future = 0.0
    self.a_acc = 0.0
    self.v_cruise = 0.0
    self.a_cruise = 0.0

    self.lead_1 = None
    self.lead_2 = None

    self.longitudinalPlanSource = 'cruise'
    self.fcw = False
    self.fcw_checker = FCWChecker()
    self.fcw_enabled = fcw_enabled

    self.params = Params()
    self.v_curvature = NO_CURVATURE_SPEED
    self.v_speedlimit = NO_CURVATURE_SPEED
    self.decel_for_turn = False
    self.map_valid = False

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

    self.last_md_ts = md.logMonoTime

    self.radar_errors = list(live20.live20.radarErrors)

    self.lead_1 = live20.live20.leadOne
    self.lead_2 = live20.live20.leadTwo

    enabled = (long_control_state == LongCtrlState.pid) or (long_control_state == LongCtrlState.stopping)
    following = self.lead_1.status and self.lead_1.dRel < 45.0 and self.lead_1.vLeadK > v_ego and self.lead_1.aLeadK > 0.0

    self.v_speedlimit = NO_CURVATURE_SPEED
    self.v_curvature = NO_CURVATURE_SPEED
    self.map_valid = live_map_data.liveMapData.mapValid

    # Speed limit and curvature
    set_speed_limit_active = self.params.get("LimitSetSpeed") == "1" and self.params.get("SpeedLimitOffset") is not None
    if set_speed_limit_active:
      if live_map_data.liveMapData.speedLimitValid:
        speed_limit = live_map_data.liveMapData.speedLimit
        offset = float(self.params.get("SpeedLimitOffset"))
        self.v_speedlimit = speed_limit + offset

      if live_map_data.liveMapData.curvatureValid:
        curvature = abs(live_map_data.liveMapData.curvature)
        a_y_max = 2.975 - v_ego * 0.0375  # ~1.85 @ 75mph, ~2.6 @ 25mph
        v_curvature = math.sqrt(a_y_max / max(1e-4, curvature))
        self.v_curvature = min(NO_CURVATURE_SPEED, v_curvature)

    self.decel_for_turn = bool(self.v_curvature < min([v_cruise_setpoint, self.v_speedlimit, v_ego + 1.]))
    v_cruise_setpoint = min([v_cruise_setpoint, self.v_curvature, self.v_speedlimit])

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
      if self.decel_for_turn:
        time_to_turn = max(1.0, live_map_data.liveMapData.distToTurn / max(self.v_cruise, 1.))
        required_decel = min(0, (self.v_curvature - self.v_cruise) / time_to_turn)
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

    self.mpc1.update(CS, self.lead_1, v_cruise_setpoint)
    self.mpc2.update(CS, self.lead_2, v_cruise_setpoint)

    self.choose_solution(v_cruise_setpoint, enabled)

    # determine fcw
    if self.mpc1.new_lead:
      self.fcw_checker.reset_lead(cur_time)

    blinkers = CS.carState.leftBlinker or CS.carState.rightBlinker
    self.fcw = self.fcw_checker.update(self.mpc1.mpc_solution, cur_time, v_ego, CS.carState.aEgo,
                                       self.lead_1.dRel, self.lead_1.vLead, self.lead_1.aLeadK,
                                       self.lead_1.yRel, self.lead_1.vLat,
                                       self.lead_1.fcw, blinkers) and not CS.carState.brakePressed
    if self.fcw:
      cloudlog.info("FCW triggered %s", self.fcw_checker.counters)

    model_dead = cur_time - (md.logMonoTime / 1e9) > 0.5

    # **** send the plan ****
    plan_send = messaging.new_message()
    plan_send.init('plan')

    # TODO: Move all these events to controlsd. This has nothing to do with planning
    events = []
    if model_dead:
      events.append(create_event('modelCommIssue', [ET.NO_ENTRY, ET.IMMEDIATE_DISABLE]))
    if 'fault' in self.radar_errors:
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

    plan_send.plan.vCurvature = self.v_curvature
    plan_send.plan.decelForTurn = self.decel_for_turn
    plan_send.plan.mapValid = self.map_valid

    # Send out fcw
    fcw = self.fcw and (self.fcw_enabled or long_control_state != LongCtrlState.off)
    plan_send.plan.fcw = fcw

    self.plan.send(plan_send.to_bytes())

    # Interpolate 0.05 seconds and save as starting point for next iteration
    dt = 0.05  # s
    a_acc_sol = self.a_acc_start + (dt / _DT_MPC) * (self.a_acc - self.a_acc_start)
    v_acc_sol = self.v_acc_start + dt * (a_acc_sol + self.a_acc_start) / 2.0
    self.v_acc_start = v_acc_sol
    self.a_acc_start = a_acc_sol
