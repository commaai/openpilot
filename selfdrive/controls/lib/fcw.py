import math
from collections import defaultdict

from common.numpy_fast import interp

_FCW_A_ACT_V = [-3., -2.]
_FCW_A_ACT_BP = [0., 30.]


class FCWChecker():
  def __init__(self):
    self.reset_lead(0.0)
    self.common_counters = defaultdict(lambda: 0)

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
    a_rel = min(a_rel, v_lead / t_decel)

    # delta of the quadratic equation to solve for ttc
    delta = v_rel**2 + 2 * x_lead * a_rel

    # assign an arbitrary high ttc value if there is no solution to ttc
    if delta < 0.1 or (math.sqrt(delta) + v_rel < 0.1):
      ttc = max_ttc
    else:
      ttc = min(2 * x_lead / (math.sqrt(delta) + v_rel), max_ttc)
    return ttc

  def update(self, mpc_solution_a, cur_time, active, v_ego, a_ego, x_lead, v_lead, a_lead, y_lead, vlat_lead, fcw_lead, blinkers):

    self.last_min_a = min(mpc_solution_a)
    self.v_lead_max = max(self.v_lead_max, v_lead)

    self.common_counters['blinkers'] = self.common_counters['blinkers'] + 10.0 / (20 * 3.0) if not blinkers else 0
    self.common_counters['v_ego'] = self.common_counters['v_ego'] + 1 if v_ego > 5.0 else 0

    if (fcw_lead > 0.99):
      ttc = self.calc_ttc(v_ego, a_ego, x_lead, v_lead, a_lead)
      self.counters['ttc'] = self.counters['ttc'] + 1 if ttc < 2.5 else 0
      self.counters['v_lead_max'] = self.counters['v_lead_max'] + 1 if self.v_lead_max > 2.5 else 0
      self.counters['v_ego_lead'] = self.counters['v_ego_lead'] + 1 if v_ego > v_lead else 0
      self.counters['lead_seen'] = self.counters['lead_seen'] + 0.33
      self.counters['y_lead'] = self.counters['y_lead'] + 1 if abs(y_lead) < 1.0 else 0
      self.counters['vlat_lead'] = self.counters['vlat_lead'] + 1 if abs(vlat_lead) < 0.4 else 0

      a_thr = interp(v_lead, _FCW_A_ACT_BP, _FCW_A_ACT_V)
      a_delta = min(mpc_solution_a[:15]) - min(0.0, a_ego)

      future_fcw_allowed = all(c >= 10 for c in self.counters.values())
      future_fcw_allowed = future_fcw_allowed and all(c >= 10 for c in self.common_counters.values())
      future_fcw = (self.last_min_a < -3.0 or a_delta < a_thr) and future_fcw_allowed

      if future_fcw and (self.last_fcw_time + 5.0 < cur_time):
        self.last_fcw_time = cur_time
        self.last_fcw_a = self.last_min_a
        return True

    return False
