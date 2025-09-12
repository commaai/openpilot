#!/usr/bin/env python3
import math
import numpy as np

from cereal import log
import cereal.messaging as messaging
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
#from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.controls.radard import _LEAD_ACCEL_TAU

A_CRUISE_MAX_VALS = [1.6, 1.2, 0.8, 0.6]
A_CRUISE_MAX_BP = [0., 10.0, 25., 40.]
A_CRUISE_MIN_VALS = [-1.2,]
A_CRUISE_MIN_BP = [0., ]

ALLOW_THROTTLE_THRESHOLD = 0.4
MIN_ALLOW_THROTTLE_SPEED = 2.5

#FCW_IDXS = ModelConstants.T_IDXS < 5.0
#T_DIFFS = np.diff(T_IDXS, prepend=[0.])
COMFORT_BRAKE = 2.5
STOP_DISTANCE = 6.0


# Lookup table for turns
_A_TOTAL_MAX_V = [1.7, 3.2]
_A_TOTAL_MAX_BP = [20., 40.]

T_IDXS = np.array(ModelConstants.T_IDXS)
FCW_IDXS = T_IDXS < 5.0
T_DIFFS = np.diff(T_IDXS, prepend=[0.])


def get_jerk_factor(personality=log.LongitudinalPersonality.standard):
  if personality==log.LongitudinalPersonality.relaxed:
    return 1.0
  elif personality==log.LongitudinalPersonality.standard:
    return 1.0
  elif personality==log.LongitudinalPersonality.aggressive:
    return 0.5
  else:
    raise NotImplementedError("Longitudinal personality not supported")


def get_T_FOLLOW(personality=log.LongitudinalPersonality.standard):
  if personality==log.LongitudinalPersonality.relaxed:
    return 1.75
  elif personality==log.LongitudinalPersonality.standard:
    return 1.45
  elif personality==log.LongitudinalPersonality.aggressive:
    return 1.25
  else:
    raise NotImplementedError("Longitudinal personality not supported")

def get_stopped_equivalence_factor(v_lead):
  return (v_lead**2) / (2 * COMFORT_BRAKE)

def get_safe_obstacle_distance(v_ego, t_follow):
  return (v_ego**2) / (2 * COMFORT_BRAKE) + t_follow * v_ego + STOP_DISTANCE

def get_desired_follow_distance(v_ego, v_lead, t_follow=None):
  if t_follow is None:
    t_follow = get_T_FOLLOW()
  return get_safe_obstacle_distance(v_ego, t_follow) - get_stopped_equivalence_factor(v_lead)


def get_max_accel(v_ego):
  return np.interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS)

def get_coast_accel(pitch):
  return np.sin(pitch) * -5.65 - 0.3  # fitted from data using xx/projects/allow_throttle/compute_coast_accel.py

def extrapolate_lead(x_lead, v_lead, a_lead, a_lead_tau):
  a_lead_traj = a_lead * np.exp(-a_lead_tau * (T_IDXS**2)/2.)
  v_lead_traj = np.clip(v_lead + np.cumsum(T_DIFFS * a_lead_traj), 0.0, 1e8)
  x_lead_traj = x_lead + np.cumsum(T_DIFFS * v_lead_traj)
  lead_xv = np.column_stack((x_lead_traj, v_lead_traj))
  return lead_xv

def process_lead(v_ego, lead):
  if lead is not None and lead.status:
    x_lead = lead.dRel
    v_lead = lead.vLead
    a_lead = lead.aLeadK
    a_lead_tau = lead.aLeadTau
  else:
    # Fake a fast lead car, so mpc can keep running in the same mode
    x_lead = 50.0
    v_lead = v_ego + 10.0
    a_lead = 0.0
    a_lead_tau = _LEAD_ACCEL_TAU

  # MPC will not converge if immediate crash is expected
  # Clip lead distance to what is still possible to brake for
  min_x_lead = ((v_ego + v_lead)/2) * (v_ego - v_lead) / (-ACCEL_MIN * 2)
  x_lead = np.clip(x_lead, min_x_lead, 1e8)
  v_lead = np.clip(v_lead, 0.0, 1e8)
  a_lead = np.clip(a_lead, -10., 5.)
  lead_xv = extrapolate_lead(x_lead, v_lead, a_lead, a_lead_tau)
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


class LongitudinalPlanner:
  def __init__(self, CP, init_v=0.0, init_a=0.0, dt=DT_MDL):
    self.CP = CP
    # TODO remove mpc modes when TR released
    self.fcw = False
    self.dt = dt
    self.allow_throttle = True

    self.a_desired = init_a
    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, self.dt)
    self.prev_accel_clip = [ACCEL_MIN, ACCEL_MAX]
    self.output_a_target = 0.0
    self.output_should_stop = False

    self.solverExecutionTime = 0.0

  @staticmethod
  def parse_model(model_msg):
    if len(model_msg.meta.disengagePredictions.gasPressProbs) > 1:
      throttle_prob = model_msg.meta.disengagePredictions.gasPressProbs[1]
    else:
      throttle_prob = 1.0
    return throttle_prob

  def update(self, sm):
    mode = 'blended' if sm['selfdriveState'].experimentalMode else 'acc'

    if len(sm['carControl'].orientationNED) == 3:
      accel_coast = get_coast_accel(sm['carControl'].orientationNED[1])
    else:
      accel_coast = ACCEL_MAX

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

    # No change cost when user is controlling the speed, or when standstill
    #prev_accel_constraint = not (reset_state or sm['carState'].standstill)

    if mode == 'acc':
      accel_clip = [ACCEL_MIN, get_max_accel(v_ego)]
      steer_angle_without_offset = sm['carState'].steeringAngleDeg - sm['liveParameters'].angleOffsetDeg
      accel_clip = limit_accel_in_turns(v_ego, steer_angle_without_offset, accel_clip, self.CP)
    else:
      accel_clip = [ACCEL_MIN, ACCEL_MAX]

    if reset_state:
      self.v_desired_filter.x = v_ego
      # Clip aEgo to cruise limits to prevent large accelerations when becoming active
      self.a_desired = np.clip(sm['carState'].aEgo, accel_clip[0], accel_clip[1])

    # Prevent divergence, smooth in current v_ego
    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))
    throttle_prob = self.parse_model(sm['modelV2'])
    # Don't clip at low speeds since throttle_prob doesn't account for creep
    self.allow_throttle = throttle_prob > ALLOW_THROTTLE_THRESHOLD or v_ego <= MIN_ALLOW_THROTTLE_SPEED

    if not self.allow_throttle:
      clipped_accel_coast = max(accel_coast, accel_clip[0])
      clipped_accel_coast_interp = np.interp(v_ego, [MIN_ALLOW_THROTTLE_SPEED, MIN_ALLOW_THROTTLE_SPEED*2], [accel_clip[1], clipped_accel_coast])
      accel_clip[1] = min(accel_clip[1], clipped_accel_coast_interp)

    if force_slow_decel:
      v_cruise = 0.0


    # TODO counter is only needed because radar is glitchy, remove once radar is gone
    #self.fcw = self.mpc.crash_cnt > 2 and not sm['carState'].standstill
    #if self.fcw:
    #  cloudlog.info("FCW triggered")

    # Interpolate 0.05 seconds and save as starting point for next iteration
    #a_prev = self.a_desired
    #self.a_desired = float(np.interp(self.dt, CONTROL_N_T_IDX, self.a_desired_trajectory))
    #self.v_desired_filter.x = self.v_desired_filter.x + self.dt * (self.a_desired + a_prev) / 2.0

    #action_t =  self.CP.longitudinalActuatorDelay + DT_MDL
    out_accels = {}
    cruise_min = np.interp(v_ego, A_CRUISE_MIN_BP, A_CRUISE_MIN_VALS)
    cruise_max = np.interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS)
    cruise_accel = 0.5*(v_cruise - v_ego)
    cruise_accel = np.clip(cruise_accel, cruise_min, cruise_max)
    out_accels['cruise'] = (cruise_accel, False)



    lead_xv_0 = process_lead(v_ego, sm['radarState'].leadOne)
    #lead_xv_1 = process_lead(v_ego, sm['radarState'].leadTwo)


    get_safe_obstacle_distance(v_ego, get_T_FOLLOW())

    ## To estimate a safe distance from a moving lead, we calculate how much stopping
    ## distance that lead needs as a minimum. We can add that to the current distance
    ## and then treat that as a stopped car/obstacle at this new distance.
    #lead_0_obstacle = lead_xv_0[:,0] + get_stopped_equivalence_factor(lead_xv_0[:,1])
    #lead_1_obstacle = lead_xv_1[:,0] + get_stopped_equivalence_factor(lead_xv_1[:,1])

    desired_follow_distance = get_desired_follow_distance(v_ego, lead_xv_0[0,1], get_T_FOLLOW())
    true_follow_distance = lead_xv_0[0,0]

    follow_distance_error = true_follow_distance - desired_follow_distance
    lead_accel = np.clip(0.2 * follow_distance_error, -3.5, 2.0)

    out_accels['lead'] = (lead_accel, False)


    if mode == 'blended':
      out_accels['model']= (sm['modelV2'].action.desiredAcceleration, sm['modelV2'].action.shouldStop)


    # smoothing opn the clip #TODO do better?
    for idx in range(2):
      accel_clip[idx] = np.clip(accel_clip[idx], self.prev_accel_clip[idx] - 0.05, self.prev_accel_clip[idx] + 0.05)
    self.prev_accel_clip = accel_clip

    output_a_target = np.min([x for x, _ in out_accels.values()])
    self.output_should_stop = np.all([x for _, x in out_accels.values()])
    self.output_a_target = np.clip(output_a_target, accel_clip[0], accel_clip[1])

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')

    plan_send.valid = sm.all_checks(service_list=['carState', 'controlsState', 'selfdriveState', 'radarState'])

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']

    longitudinalPlan.hasLead = sm['radarState'].leadOne.status
    #longitudinalPlan.longitudinalPlanSource = self.mpc.source
    longitudinalPlan.fcw = self.fcw

    longitudinalPlan.aTarget = float(self.output_a_target)
    longitudinalPlan.shouldStop = bool(self.output_should_stop)
    longitudinalPlan.allowBrake = True
    longitudinalPlan.allowThrottle = bool(self.allow_throttle)

    pm.send('longitudinalPlan', plan_send)
