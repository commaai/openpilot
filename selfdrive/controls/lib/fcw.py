import numpy as np
from common.realtime import sec_since_boot

#Time to collisions greater than 5s are iognored
MAX_TTC = 5.

def calc_ttc(l1):
  # if l1 is None, return max ttc immediately
  if not l1:
    return MAX_TTC
  # this function returns the time to collision (ttc), assuming that 
  # ARel will stay constant TODO: review this assumptions
  # change sign to rel quantities as it's going to be easier for calculations
  vRel = -l1.vRel
  aRel = -l1.aRel

  # assuming that closing gap ARel comes from lead vehicle decel, 
  # then limit ARel so that v_lead will get to zero in no sooner than t_decel.
  # This helps underweighting ARel when v_lead is close to zero.
  t_decel = 2.
  aRel = np.minimum(aRel, l1.vLead/t_decel)

  # delta of the quadratic equation to solve for ttc
  delta = vRel**2 + 2 * l1.dRel * aRel

  # assign an arbitrary high ttc value if there is no solution to ttc
  if delta < 0.1 or (np.sqrt(delta) + vRel < 0.1):
    ttc = MAX_TTC
  else:
    ttc = np.minimum(2 * l1.dRel / (np.sqrt(delta) + vRel), MAX_TTC)
  return ttc

class ForwardCollisionWarning(object):
  def __init__(self, dt):
    self.last_active = 0.
    self.violation_time = 0.
    self.active = False
    self.dt = dt   # time step

  def process(self, CS, AC):
    # send an fcw alert if the violation time > violation_thrs
    violation_thrs = 0.3  # fcw turns on after a continuous violation for this time
    fcw_t_delta = 5.         # no more than one fcw alert within this time
    a_acc_on  = -2.0         # with system on, above this limit of desired decel, we should trigger fcw
    a_acc_off = -2.5         # with system off, above this limit of desired decel, we should trigger fcw
    ttc_thrs = 2.5           # ttc threshold for fcw
    v_fcw_min = 2.           # no fcw below 2m/s
    steer_angle_th = 40.     # deg, no fcw above this steer angle
    cur_time = sec_since_boot()
 
    ttc = calc_ttc(AC.l1)
    a_fcw = a_acc_on if CS.cruiseState.enabled else a_acc_off

    # increase violation time if we want to decelerate quite fast
    if AC.l1 and ( \
        (CS.vEgo > v_fcw_min) and (CS.vEgo > AC.v_target_lead) and (AC.a_target[0] < a_fcw) \
        and not CS.brakePressed and ttc < ttc_thrs and abs(CS.steeringAngle) < steer_angle_th \
        and AC.l1.fcw):
      self.violation_time = np.minimum(self.violation_time + self.dt, violation_thrs)
    else:
      self.violation_time = np.maximum(self.violation_time - 2*self.dt, 0)
 
    # fire FCW  
    self.active = self.violation_time >= violation_thrs and cur_time > (self.last_active + fcw_t_delta)
    if self.active:
      self.last_active = cur_time
