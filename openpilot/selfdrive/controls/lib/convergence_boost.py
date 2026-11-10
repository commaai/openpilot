import numpy as np
from openpilot.common.constants import CV
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants

# e2e long tends to settle below the target speed on open road; pull it up when the model isn't planning to slow
K_CONV = 0.3                       # 1/s: accel per m/s below target
A_CONV_MAX = 0.8
DEADBAND = 1.0 * CV.MPH_TO_MS
PLAN_HORIZON_S = 5.0
PLAN_SLOWDOWN_BP = [0.5, 1.5]      # m/s the model plans to lose within the horizon -> boost fades out
E2E_BRAKING_BP = [-0.2, 0.0]       # model braking at all -> boost fades out
SPEED_BP = [10 * CV.MPH_TO_MS, 20 * CV.MPH_TO_MS]
HORIZON = np.array(ModelConstants.T_IDXS) <= PLAN_HORIZON_S
RISE_TAU_S = 1.0                   # boost ramps in slowly, drops instantly


class ConvergenceBoost:
  def __init__(self):
    self.boost = 0.0

  def update(self, sm, a_e2e, v_ego, v_target):
    target = self._target(sm, a_e2e, v_ego, v_target)
    self.boost = target if target < self.boost else self.boost + (target - self.boost) * DT_MDL / RISE_TAU_S
    return a_e2e + self.boost

  def _target(self, sm, a_e2e, v_ego, v_target):
    model = sm['modelV2']
    v_plan = np.asarray(model.velocity.x)
    if not np.isfinite(v_target) or model.action.shouldStop or len(v_plan) != len(HORIZON):
      return 0.0
    deficit = v_target - v_ego - DEADBAND
    if deficit <= 0:
      return 0.0
    plan_slowdown = v_ego - v_plan[HORIZON].min()
    gate = (np.interp(plan_slowdown, PLAN_SLOWDOWN_BP, [1.0, 0.0]) * np.interp(a_e2e, E2E_BRAKING_BP, [0.0, 1.0]) *
            np.interp(v_ego, SPEED_BP, [0.0, 1.0]))
    a_conv = min(K_CONV * deficit, A_CONV_MAX)
    return float(gate * max(a_conv - a_e2e, 0.0))
