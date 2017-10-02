import math
import numpy as np
from common.numpy_fast import clip, interp
import selfdrive.messaging as messaging

# TODO: we compute a_pcm but we don't use it, as accelOverride is hardcoded in controlsd

# lookup tables VS speed to determine min and max accels in cruise
_A_CRUISE_MIN_V  = [-1.0, -.8, -.67, -.5, -.30]
_A_CRUISE_MIN_BP = [   0., 5.,  10., 20.,  40.]

# need fast accel at very low speed for stop and go
_A_CRUISE_MAX_V  = [1., 1., .8, .5, .30]
_A_CRUISE_MAX_BP = [0.,  5., 10., 20., 40.]

def calc_cruise_accel_limits(v_ego):
  a_cruise_min = interp(v_ego, _A_CRUISE_MIN_BP, _A_CRUISE_MIN_V)
  a_cruise_max = interp(v_ego, _A_CRUISE_MAX_BP, _A_CRUISE_MAX_V)
  return np.vstack([a_cruise_min, a_cruise_max])

_A_TOTAL_MAX_V = [1.5, 1.9, 3.2]
_A_TOTAL_MAX_BP = [0., 20., 40.]

def limit_accel_in_turns(v_ego, angle_steers, a_target, a_pcm, CP):
  #*** this function returns a limited long acceleration allowed, depending on the existing lateral acceleration
  # this should avoid accelerating when losing the target in turns
  deg_to_rad = np.pi / 180.  # from can reading to rad

  a_total_max = interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
  a_y = v_ego**2 * angle_steers * deg_to_rad / (CP.sR * CP.l)
  a_x_allowed = math.sqrt(max(a_total_max**2 - a_y**2, 0.))

  a_target[1] = min(a_target[1], a_x_allowed)
  a_pcm = min(a_pcm, a_x_allowed)
  return a_target, a_pcm

def process_a_lead(a_lead):
  # soft threshold of 0.5m/s^2 applied to a_lead to reject noise, also not considered positive a_lead
  a_lead_threshold = 0.5
  a_lead = min(a_lead + a_lead_threshold, 0)
  return a_lead

def calc_desired_distance(v_lead):
  #*** compute desired distance ***
  t_gap = 1.7  # good to be far away
  d_offset = 4 # distance when at zero speed
  return d_offset + v_lead * t_gap


#linear slope
_L_SLOPE_V = [0.40, 0.10]
_L_SLOPE_BP = [0.,  40]

# parabola slope
_P_SLOPE_V = [1.0, 0.25]
_P_SLOPE_BP = [0., 40]

def calc_desired_speed(d_lead, d_des, v_lead, a_lead):
  #*** compute desired speed ***
  # the desired speed curve is divided in 4 portions: 
  # 1-constant
  # 2-linear to regain distance
  # 3-linear to shorten distance
  # 4-parabolic (constant decel)

  max_runaway_speed = -2. # no slower than 2m/s over the lead

  # interpolate the lookups to find the slopes for a give lead speed
  l_slope = interp(v_lead, _L_SLOPE_BP, _L_SLOPE_V)
  p_slope = interp(v_lead, _P_SLOPE_BP, _P_SLOPE_V)

  # this is where parabola and linear curves are tangents  
  x_linear_to_parabola = p_slope / l_slope**2

  # parabola offset to have the parabola being tangent to the linear curve
  x_parabola_offset = p_slope / (2 * l_slope**2)

  if d_lead < d_des:
    # calculate v_rel_des on the line that connects 0m at max_runaway_speed to d_des
    v_rel_des_1 = (- max_runaway_speed) / d_des * (d_lead - d_des)
    # calculate v_rel_des on one third of the linear slope
    v_rel_des_2 = (d_lead - d_des) * l_slope / 3.
    # take the min of the 2 above
    v_rel_des = min(v_rel_des_1, v_rel_des_2)
    v_rel_des = max(v_rel_des, max_runaway_speed)
  elif d_lead < d_des + x_linear_to_parabola:
    v_rel_des = (d_lead - d_des) * l_slope
    v_rel_des = max(v_rel_des, max_runaway_speed)
  else:
    v_rel_des = math.sqrt(2 * (d_lead - d_des - x_parabola_offset) * p_slope)

  # compute desired speed
  v_target = v_rel_des + v_lead

  # compute v_coast: above this speed we want to coast
  t_lookahead = 1.   # how far in time we consider a_lead to anticipate the coast region
  v_coast_shift = max(a_lead * t_lookahead, - v_lead)   # don't consider projections that would make v_lead<0
  v_coast = (v_lead + v_target)/2 + v_coast_shift              # no accel allowed above this line
  v_coast = min(v_coast, v_target)

  return v_target, v_coast

def calc_critical_decel(d_lead, v_rel, d_offset, v_offset):
  # this function computes the required decel to avoid crashing, given safety offsets
  a_critical =  - max(0., v_rel + v_offset)**2/max(2*(d_lead - d_offset), 0.5)
  return a_critical


# maximum acceleration adjustment
_A_CORR_BY_SPEED_V = [0.4, 0.4, 0]
# speeds
_A_CORR_BY_SPEED_BP = [0., 2., 10.]

# max acceleration allowed in acc, which happens in restart
A_ACC_MAX = max(_A_CORR_BY_SPEED_V) + max(_A_CRUISE_MAX_V)

def calc_positive_accel_limit(d_lead, d_des, v_ego, v_rel, v_ref, v_rel_ref, v_coast, v_target, a_lead_contr, a_max):
  a_coast_min = -1.0   # never coast faster then -1m/s^2
  # coasting behavior above v_coast. Forcing a_max to be negative will force the pid_speed to decrease,
  # regardless v_target
  if v_ref > min(v_coast, v_target):
    # for smooth coast we can be aggressive and target a point where car would actually crash
    v_offset_coast = 0.
    d_offset_coast = d_des/2. - 4.

    # acceleration value to smoothly coast until we hit v_target
    if d_lead > d_offset_coast + 0.1:
      a_coast = calc_critical_decel(d_lead, v_rel_ref, d_offset_coast, v_offset_coast)
      # if lead is decelerating, then offset the coast decel
      a_coast += a_lead_contr
      a_max = max(a_coast, a_coast_min)
    else:
      a_max = a_coast_min
  else:
    # same as cruise accel, plus add a small correction based on relative lead speed
    # if the lead car is faster, we can accelerate more, if the car is slower, then we can reduce acceleration
    a_max = a_max + interp(v_ego, _A_CORR_BY_SPEED_BP, _A_CORR_BY_SPEED_V) \
                  * clip(-v_rel / 4., -.5, 1)
  return a_max

# arbitrary limits to avoid too high accel being computed
_A_SAT = [-10., 5.]

# do not consider a_lead at 0m/s, fully consider it at 10m/s
_A_LEAD_LOW_SPEED_V = [0., 1.]

# speed break points
_A_LEAD_LOW_SPEED_BP = [0., 10.]

# add a small offset to the desired decel, just for safety margin
_DECEL_OFFSET_V = [-0.3, -0.5, -0.5, -0.4, -0.3]

# speed bp: different offset based on the likelyhood that lead decels abruptly
_DECEL_OFFSET_BP = [0., 4., 15., 30, 40.]


def calc_acc_accel_limits(d_lead, d_des, v_ego, v_pid, v_lead, v_rel, a_lead,
                          v_target, v_coast, a_target, a_pcm):
  #*** compute max accel ***
  # v_rel is now your velocity in lead car frame
  v_rel *= -1  # this simplifies things when thinking in d_rel-v_rel diagram

  v_rel_pid = v_pid - v_lead

  # this is how much lead accel we consider in assigning the desired decel
  a_lead_contr = a_lead * interp(v_lead, _A_LEAD_LOW_SPEED_BP,
                                 _A_LEAD_LOW_SPEED_V) * 0.8

  # first call of calc_positive_accel_limit is used to shape v_pid
  a_target[1] = calc_positive_accel_limit(d_lead, d_des, v_ego, v_rel, v_pid,
                                          v_rel_pid, v_coast, v_target,
                                          a_lead_contr, a_target[1])
  # second call of calc_positive_accel_limit is used to limit the pcm throttle
  # control (only useful when we don't control throttle directly)
  a_pcm = calc_positive_accel_limit(d_lead, d_des, v_ego, v_rel, v_ego,
                                    v_rel, v_coast, v_target,
                                    a_lead_contr, a_pcm)

  #*** compute max decel ***
  v_offset = 1.  # assume the car is 1m/s slower
  d_offset = 1.  # assume the distance is 1m lower
  if v_target - v_ego > 0.5:
    pass  # acc target speed is above vehicle speed, so we can use the cruise limits
  elif d_lead > d_offset + 0.01:  # add small value to avoid by zero divisions
    # compute needed accel to get to 1m distance with -1m/s rel speed
    decel_offset = interp(v_lead, _DECEL_OFFSET_BP, _DECEL_OFFSET_V)

    critical_decel = calc_critical_decel(d_lead, v_rel, d_offset, v_offset)
    a_target[0] = min(decel_offset + critical_decel + a_lead_contr,
                      a_target[0])
  else:
    a_target[0] = _A_SAT[0]
  # a_min can't be higher than a_max
  a_target[0] = min(a_target[0], a_target[1])
  # final check on limits
  a_target = np.clip(a_target, _A_SAT[0], _A_SAT[1])
  a_target = a_target.tolist()
  return a_target, a_pcm

def calc_jerk_factor(d_lead, v_rel):
  # we don't have an explicit jerk limit, so this function calculates a factor
  # that is used by the PID controller to scale the gains. Not the cleanest solution 
  # but we need this for the demo.
  # TODO: Calculate Kp and Ki directly in this function.

  # the higher is the decel required to avoid a crash, the higher is the PI factor scaling
  d_offset = 0.5
  v_offset = 2.
  a_offset = 1.
  jerk_factor_max = 1.0 # can't increase Kp and Ki more than double.
  if d_lead < d_offset + 0.1: # add small value to avoid by zero divisions
    jerk_factor = jerk_factor_max
  else:
    a_critical = - calc_critical_decel(d_lead, -v_rel, d_offset, v_offset)
    # increase Kp and Ki by 20% for every 1m/s2 of decel required above 1m/s2
    jerk_factor = max(a_critical - a_offset, 0.)/5.
    jerk_factor = min(jerk_factor, jerk_factor_max)
  return jerk_factor



MAX_SPEED_POSSIBLE = 55.

def compute_speed_with_leads(v_ego, angle_steers, v_pid, l1, l2, CP):
  # drive limits
  # TODO: Make lims function of speed (more aggressive at low speed).
  a_lim = [-3., 1.5]

  #*** set target speed pretty high, as lead hasn't been considered yet
  v_target_lead = MAX_SPEED_POSSIBLE

  #*** set accel limits as cruise accel/decel limits ***
  a_target = calc_cruise_accel_limits(v_ego)

  # start with 1
  a_pcm = 1.

  #*** limit max accel in sharp turns
  a_target, a_pcm = limit_accel_in_turns(v_ego, angle_steers, a_target, a_pcm, CP)
  jerk_factor = 0.

  if l1 is not None and l1.status:
    #*** process noisy a_lead signal from radar processing ***
    a_lead_p = process_a_lead(l1.aLeadK)

    #*** compute desired distance ***
    d_des = calc_desired_distance(l1.vLead)

    #*** compute desired speed ***
    v_target_lead, v_coast = calc_desired_speed(l1.dRel, d_des, l1.vLead, a_lead_p)

    if l2 is not None and l2.status:
      #*** process noisy a_lead signal from radar processing ***
      a_lead_p2 = process_a_lead(l2.aLeadK)

      #*** compute desired distance ***
      d_des2 = calc_desired_distance(l2.vLead)

      #*** compute desired speed ***
      v_target_lead2, v_coast2 = calc_desired_speed(l2.dRel, d_des2, l2.vLead, a_lead_p2)

      # listen to lead that makes you go slower
      if v_target_lead2 < v_target_lead:
        l1 = l2
        d_des, a_lead_p, v_target_lead, v_coast = d_des2, a_lead_p2, v_target_lead2, v_coast2

    # l1 is the main lead now

    #*** compute accel limits ***
    a_target1, a_pcm1 = calc_acc_accel_limits(l1.dRel, d_des, v_ego, v_pid, l1.vLead,
                                     l1.vRel, a_lead_p, v_target_lead, v_coast, a_target, a_pcm)

    # we can now limit a_target to a_lim
    a_target = np.clip(a_target1, a_lim[0], a_lim[1])
    a_pcm = np.clip(a_pcm1, a_lim[0], a_lim[1]).tolist()

    #*** compute max factor ***
    jerk_factor = calc_jerk_factor(l1.dRel, l1.vRel)

  # force coasting decel if driver hasn't been controlling car in a while
  return v_target_lead, a_target, a_pcm, jerk_factor


class AdaptiveCruise(object):
  def __init__(self):
    self.l1, self.l2 = None, None
  def update(self, v_ego, angle_steers, v_pid, CP, l20):
    if l20 is not None:
      self.l1 = l20.live20.leadOne
      self.l2 = l20.live20.leadTwo

    self.v_target_lead, self.a_target, self.a_pcm, self.jerk_factor = \
      compute_speed_with_leads(v_ego, angle_steers, v_pid, self.l1, self.l2, CP)
    self.has_lead = self.v_target_lead != MAX_SPEED_POSSIBLE
