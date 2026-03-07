#!/usr/bin/env python3
import math
import numpy as np

from cereal import log
import cereal.messaging as messaging
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.constants import CV
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.drive_helpers import smooth_value
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.common.swaglog import cloudlog

LongitudinalPlanSource = log.LongitudinalPlan.LongitudinalPlanSource

CRUISE_MIN_ACCEL = -1.2
A_CRUISE_MAX_VALS = [1.6, 1.2, 0.8, 0.6]
A_CRUISE_MAX_BP = [0., 10.0, 25., 40.]
ALLOW_THROTTLE_THRESHOLD = 0.4
MIN_ALLOW_THROTTLE_SPEED = 2.5
ACCEL_CLIP_JERK_MAX = 1.0

K_CRUISE = 0.27
TAU_CRUISE_SMOOTH = 1.0

K_P_LEAD = 2.0
K_D_LEAD = 5.5
TAU_LEAD_SMOOTH = 0.25

COMFORT_BRAKE = 2.5
STOP_DISTANCE = 6.0
CRASH_DISTANCE = .25

# Lookup table for turns
_A_TOTAL_MAX_V = [1.7, 3.2]
_A_TOTAL_MAX_BP = [20., 40.]

T_IDXS = np.array(ModelConstants.T_IDXS)
FCW_IDXS = T_IDXS < 5.0
T_IDXS_LEAD = T_IDXS[FCW_IDXS]
T_DIFFS_LEAD = np.diff(T_IDXS_LEAD, prepend=[0.])

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
  x_lead = np.clip(lead.dRel, 0.01, 1e8)
  v_lead = np.clip(lead.vLead, 0.0, 1e8)
  a_lead = np.clip(lead.aLeadK, -10., 5.)
  a_lead_tau = lead.aLeadTau
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

def get_throttle_prob(model_msg):
  gas_press_probs = model_msg.meta.disengagePredictions.gasPressProbs
  return gas_press_probs[1] if len(gas_press_probs) > 1 else 1.0

# TODO should we assume larger deceleration for ego to break harsher
def get_follow_distance(v_ego, v_lead, t_follow):
  return (v_ego**2 - v_lead**2) / (2 * COMFORT_BRAKE) + t_follow * v_ego + STOP_DISTANCE

def lead_controller(v_ego, v_lead, actual_distance, accel_clip, output_a_target, t_follow):
  desired_distance = get_follow_distance(v_ego, v_lead, t_follow)
  e_d = (actual_distance - desired_distance) / (v_ego + 5.0)
  e_v = (v_lead - v_ego) / (v_ego + 10.0)
  lead_accel = K_P_LEAD * e_d + K_D_LEAD * e_v
  lead_accel = np.clip(lead_accel, accel_clip[0], accel_clip[1])
  lead_accel = smooth_value(lead_accel, output_a_target, TAU_LEAD_SMOOTH)
  return lead_accel

def check_collision(v_ego, lead_xv, output_a_target, accel_clip, t_follow):
  x_ego = 0.0
  for idx, dt in enumerate(T_DIFFS_LEAD):
    x_lead, v_lead = lead_xv[idx]
    actual_distance = x_lead - x_ego
    if actual_distance < CRASH_DISTANCE:
      return True
    a = lead_controller(v_ego, v_lead, actual_distance, accel_clip, output_a_target, t_follow)
    output_a_target = a
    v_ego = max(0.0, v_ego + a * dt)
    x_ego += v_ego * dt
  return False

class LongitudinalPlanner:
  def __init__(self, CP, dt=DT_MDL):
    self.CP = CP
    self.source = LongitudinalPlanSource.cruise
    self.crash_cnt = 0
    self.fcw = False
    self.dt = dt
    self.allow_throttle = True

    self.prev_accel_clip = [ACCEL_MIN, ACCEL_MAX]
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
      # Clip aEgo to cruise limits to prevent large accelerations when becoming active
      self.output_a_target = np.clip(sm['carState'].aEgo, accel_clip[0], accel_clip[1])
      self.prev_accel_clip = accel_clip

    # Prevent divergence, smooth in current v_ego
    throttle_prob = get_throttle_prob(sm['modelV2'])
    # Don't clip at low speeds since throttle_prob doesn't account for creep
    self.allow_throttle = throttle_prob > ALLOW_THROTTLE_THRESHOLD or v_ego <= MIN_ALLOW_THROTTLE_SPEED

    if not self.allow_throttle:
      clipped_accel_coast = max(accel_coast, accel_clip[0])
      clipped_accel_coast_interp = np.interp(v_ego, [MIN_ALLOW_THROTTLE_SPEED, MIN_ALLOW_THROTTLE_SPEED*2], [accel_clip[1], clipped_accel_coast])
      accel_clip[1] = min(accel_clip[1], clipped_accel_coast_interp)

    if force_slow_decel:
      v_cruise = 0.0

    accel_clip[1] = np.clip(accel_clip[1], self.prev_accel_clip[1] - self.dt*ACCEL_CLIP_JERK_MAX, self.prev_accel_clip[1] + self.dt*ACCEL_CLIP_JERK_MAX)
    self.prev_accel_clip = accel_clip

    out_accels = {}
    if sm['selfdriveState'].experimentalMode:
      out_accels[LongitudinalPlanSource.e2e] = (sm['modelV2'].action.desiredAcceleration, sm['modelV2'].action.shouldStop)

    cruise_accel = K_CRUISE * (v_cruise - v_ego)
    cruise_accel = np.clip(cruise_accel, CRUISE_MIN_ACCEL, accel_clip[1])
    cruise_accel = smooth_value(cruise_accel, self.output_a_target, TAU_CRUISE_SMOOTH)
    out_accels[LongitudinalPlanSource.cruise] = (cruise_accel, False)

    lead_0, lead_1 = sm['radarState'].leadOne, sm['radarState'].leadTwo
    lead_info = {LongitudinalPlanSource.lead0: lead_0, LongitudinalPlanSource.lead1: lead_1}
    for key in lead_info.keys():
      lead_xv = process_lead(lead_info[key])
      if lead_xv is None:
        continue

      actual_distance, v_lead = lead_xv[0]
      lead_accel = lead_controller(v_ego, v_lead, actual_distance, accel_clip, self.output_a_target, t_follow)

      v_lead_1sec = np.interp(1.0, T_IDXS_LEAD, lead_xv[:, 1])
      should_stop_lead = (v_lead < 0.5 and v_lead_1sec < 0.5 and actual_distance < get_follow_distance(v_ego, 0.0, t_follow))
      out_accels[key] = (lead_accel, should_stop_lead)

      if key == LongitudinalPlanSource.lead0:
        if lead_info[key].fcw and check_collision(v_ego, lead_xv, self.output_a_target, accel_clip, t_follow):
          self.crash_cnt += 1
        else:
          self.crash_cnt = 0

    if not lead_0.status:
      self.crash_cnt = 0

    # TODO counter is only needed because radar is glitchy, remove once radar is gone
    self.fcw = self.crash_cnt > 2 and not sm['carState'].standstill
    if self.fcw:
      cloudlog.info("FCW triggered")

    source, (output_a_target, _) = min(out_accels.items(), key=lambda kv: kv[1][0])
    self.source = source
    self.output_should_stop = any(should_stop for _, should_stop in out_accels.values())
    self.output_a_target = np.clip(output_a_target, accel_clip[0], accel_clip[1])

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')

    plan_send.valid = sm.all_checks(service_list=['carState', 'controlsState', 'selfdriveState', 'radarState'])

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']

    longitudinalPlan.hasLead = sm['radarState'].leadOne.status
    longitudinalPlan.longitudinalPlanSource = self.source
    longitudinalPlan.fcw = self.fcw

    longitudinalPlan.aTarget = float(self.output_a_target)
    longitudinalPlan.shouldStop = bool(self.output_should_stop)
    longitudinalPlan.allowBrake = True
    longitudinalPlan.allowThrottle = bool(self.allow_throttle)

    pm.send('longitudinalPlan', plan_send)
