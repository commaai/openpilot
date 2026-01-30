#!/usr/bin/env python3
import math
import numpy as np

from enum import Enum
from cereal import log
import cereal.messaging as messaging
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_accel_from_plan, smooth_value
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.controls.radard import _LEAD_ACCEL_TAU

CRUISE_MIN_ACCEL = -1.2
A_CRUISE_MAX_VALS = [1.6, 1.2, 0.8, 0.6]
A_CRUISE_MAX_BP = [0., 10.0, 25., 40.]
ALLOW_THROTTLE_THRESHOLD = 0.4
MIN_ALLOW_THROTTLE_SPEED = 2.5

K_CRUISE = 0.75
TAU_CRUISE = 0.4

COMFORT_BRAKE = 2.5
STOP_DISTANCE = 6.0
CRASH_DISTANCE = .25
MIN_X_LEAD_FACTOR = 0.5

# Lookup table for turns
_A_TOTAL_MAX_V = [1.7, 3.2]
_A_TOTAL_MAX_BP = [20., 40.]

T_IDXS = np.array(ModelConstants.T_IDXS)
FCW_IDXS = T_IDXS < 5.0
T_IDXS_LEAD = T_IDXS[FCW_IDXS]
T_DIFFS_LEAD = np.diff(T_IDXS_LEAD, prepend=[0.])


class Source(Enum):
  LEAD0 = 0
  LEAD1 = 1
  CRUISE = 2
  E2E = 3


def get_T_FOLLOW(personality=log.LongitudinalPersonality.standard):
  if personality==log.LongitudinalPersonality.relaxed:
    return 1.75
  elif personality==log.LongitudinalPersonality.standard:
    return 1.45
  elif personality==log.LongitudinalPersonality.aggressive:
    return 1.25
  else:
    raise NotImplementedError("Longitudinal personality not supported")

def get_max_accel(v_ego):
  return np.interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS)

def get_coast_accel(pitch):
  return np.sin(pitch) * -5.65 - 0.3  # fitted from data using xx/projects/allow_throttle/compute_coast_accel.py

def extrapolate_state(x, v, a, a_tau):
  a_traj = a * np.exp(-a_tau * (T_IDXS_LEAD**2)/2.)
  v_traj = np.clip(v + np.cumsum(T_DIFFS_LEAD * a_traj), 0.0, 1e8)
  x_traj = x + np.cumsum(T_DIFFS_LEAD * v_traj)
  xv_traj = np.column_stack((x_traj, v_traj))
  return xv_traj

def process_lead(lead):
  if lead is None or not lead.status:
    return None

  x_lead = lead.dRel
  v_lead = lead.vLead
  a_lead = lead.aLeadK
  a_lead_tau = lead.aLeadTau

  x_lead = np.clip(x_lead, 0.01, 1e8)
  v_lead = np.clip(v_lead, 0.0, 1e8)
  a_lead = np.clip(a_lead, -10., 5.)
  lead_xv = extrapolate_state(x_lead, v_lead, a_lead, a_lead_tau)
  return lead_xv

def limit_accel_in_turns(v_ego, angle_steers, a_target, CP):
  """
  This function returns a limited long acceleration allowed, depending on the existing lateral acceleration
  this should avoid accelerating when losing the target in turns
  """
  # FIXME: This function to calculate lateral accel is incorrect and should use the VehicleModel
  # The lookup table for turns should also be updated if we do this
  a_total_max = np.interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
  a_y = v_ego ** 2 * angle_steers * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)
  a_x_allowed = math.sqrt(max(a_total_max ** 2 - a_y ** 2, 0.))

  return [a_target[0], min(a_target[1], a_x_allowed)]

def parse_model(model_msg):
  if len(model_msg.meta.disengagePredictions.gasPressProbs) > 1:
    throttle_prob = model_msg.meta.disengagePredictions.gasPressProbs[1]
  else:
    throttle_prob = 1.0
  return throttle_prob

# TODO maybe use predicted values at delay instead
def lead_controller(v_ego, v_lead, gap_0, v_cruise, t_follow, accel_clip):
  dv = v_lead - v_ego
  closing_speed = max(0.0, -dv)
  # TODO might need to be just COMFORT_BRAKE if it follows lead to closely
  dynamic_term = (v_ego * closing_speed) / (2.0 * np.sqrt(accel_clip[1] * COMFORT_BRAKE))
  desired_gap = STOP_DISTANCE + v_ego * t_follow + dynamic_term

  velocity_term = (v_ego / max(v_cruise, 0.1)) ** 4
  gap_term = (desired_gap / gap_0) ** 2

  return accel_clip[1] * (1.0 - velocity_term - gap_term)

def simulate_trajectory(v_ego, lead_xv, v_cruise, t_follow, accel_clip):
  a_traj = []
  v_traj = []
  x_ego = 0.0
  collision = False
  
  for idx, dt in enumerate(T_DIFFS_LEAD):
    x_lead, v_lead = lead_xv[idx]
    gap = x_lead - x_ego
    
    if gap < CRASH_DISTANCE:
      collision = True
    
    a_target = lead_controller(v_ego, v_lead, gap, v_cruise, t_follow, accel_clip)
    
    a_traj.append(a_target)
    v_traj.append(v_ego)
    
    v_ego += a_target * dt
    v_ego = max(0.0, v_ego)
    x_ego += v_ego * dt
  
  return np.array(v_traj), np.array(a_traj), collision


class LongitudinalPlanner:
  def __init__(self, CP, init_v=0.0, dt=DT_MDL):
    self.CP = CP
    self.source = Source.CRUISE
    self.crash_cnt = 0
    self.fcw = False
    self.dt = dt
    self.allow_throttle = True

    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, self.dt)
    self.output_a_target = 0.0
    self.output_should_stop = False

  def update(self, sm):
    if len(sm['carControl'].orientationNED) == 3:
      accel_coast = get_coast_accel(sm['carControl'].orientationNED[1])
    else:
      accel_coast = ACCEL_MAX

    t_follow = get_T_FOLLOW(sm['selfdriveState'].personality)
    v_ego = sm['carState'].vEgo
    v_cruise_kph = min(sm['carState'].vCruise, V_CRUISE_MAX)
    v_cruise = v_cruise_kph * CV.KPH_TO_MS
    v_cruise_initialized = sm['carState'].vCruise != V_CRUISE_UNSET

    long_control_off = sm['controlsState'].longControlState == LongCtrlState.off
    force_slow_decel = sm['controlsState'].forceDecel

    # Reset current state when not engaged, or user is controlling the speed
    reset_state = long_control_off if self.CP.openpilotLongitudinalControl else not sm['selfdriveState'].enabled
    # PCM cruise speed may be updated a few cycles later, check if initialized
    reset_state = reset_state or not v_cruise_initialized

    accel_clip = [ACCEL_MIN, get_max_accel(v_ego)]
    steer_angle_without_offset = sm['carState'].steeringAngleDeg - sm['liveParameters'].angleOffsetDeg
    accel_clip = limit_accel_in_turns(v_ego, steer_angle_without_offset, accel_clip, self.CP)

    if reset_state:
      self.v_desired_filter.x = v_ego
      # Clip aEgo to cruise limits to prevent large accelerations when becoming active
      self.output_a_target = np.clip(sm['carState'].aEgo, accel_clip[0], accel_clip[1])

    # Prevent divergence, smooth in current v_ego
    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))
    throttle_prob = parse_model(sm['modelV2'])
    # Don't clip at low speeds since throttle_prob doesn't account for creep
    self.allow_throttle = throttle_prob > ALLOW_THROTTLE_THRESHOLD or v_ego <= MIN_ALLOW_THROTTLE_SPEED

    if not self.allow_throttle:
      clipped_accel_coast = max(accel_coast, accel_clip[0])
      clipped_accel_coast_interp = np.interp(v_ego, [MIN_ALLOW_THROTTLE_SPEED, MIN_ALLOW_THROTTLE_SPEED*2], [accel_clip[1], clipped_accel_coast])
      accel_clip[1] = min(accel_clip[1], clipped_accel_coast_interp)

    if force_slow_decel:
      v_cruise = 0.0

    out_accels = {}
    if sm['selfdriveState'].experimentalMode:
      out_accels[Source.E2E] = (sm['modelV2'].action.desiredAcceleration, sm['modelV2'].action.shouldStop)

    cruise_accel = K_CRUISE * (v_cruise - v_ego)
    cruise_accel = np.clip(cruise_accel, CRUISE_MIN_ACCEL, accel_clip[1])
    cruise_accel = smooth_value(cruise_accel, self.output_a_target, TAU_CRUISE)
    out_accels[Source.CRUISE] = (cruise_accel, False)

    lead_info = {Source.LEAD0: sm['radarState'].leadOne, Source.LEAD1: sm['radarState'].leadTwo}
    for key in lead_info.keys():
      lead_xv = process_lead(lead_info[key])
      if lead_xv is None:
        continue
      v_traj, a_traj, collision = simulate_trajectory(v_ego, lead_xv, v_cruise, t_follow, accel_clip)
      if key == Source.LEAD0:
        if lead_info[key].fcw and collision:
          self.crash_cnt += 1
        else:
          self.crash_cnt = 0

      action_t =  self.CP.longitudinalActuatorDelay + DT_MDL
      out_accels[key] = get_accel_from_plan(v_traj, a_traj, T_IDXS_LEAD, action_t, self.CP.vEgoStopping)


    # TODO counter is only needed because radar is glitchy, remove once radar is gone
    self.fcw = self.crash_cnt > 2 and not sm['carState'].standstill
    if self.fcw:
      cloudlog.info("FCW triggered")

    source, (output_a_target, _) = min(out_accels.items(), key=lambda x: x[1][0])
    self.source = source
    self.output_should_stop = any(should_stop for _, should_stop in out_accels.values())

    self.output_a_target = np.clip(output_a_target, accel_clip[0], accel_clip[1])

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')

    plan_send.valid = sm.all_checks(service_list=['carState', 'controlsState', 'selfdriveState', 'radarState'])

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']

    longitudinalPlan.hasLead = sm['radarState'].leadOne.status
    longitudinalPlan.longitudinalPlanSource = self.source.name
    longitudinalPlan.fcw = self.fcw

    longitudinalPlan.aTarget = float(self.output_a_target)
    longitudinalPlan.shouldStop = bool(self.output_should_stop)
    longitudinalPlan.allowBrake = True
    longitudinalPlan.allowThrottle = bool(self.allow_throttle)

    pm.send('longitudinalPlan', plan_send)
