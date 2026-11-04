import numpy as np

# Pulling away from a stop: the model asks for a gentle start. Make it go faster, but only for a real launch (not a
# creep), and not toward a close lead that does not pull away. The boost is gone by LAUNCH_SPEED_BP[-1].
LAUNCH_SPEED_BP = [0.0, 5.0]   # m/s
LAUNCH_GAIN_V = [1.6, 1.0]
LAUNCH_MIN_ACCEL = 0.2         # m/s², the model must ask for at least this much accel

# Plan speed 3 s ahead: at the first value it is a creep (no boost), at the second a real launch (full boost).
# Fleet data (1,162 starts from a stop): 1 s into a start, a plan speed above 3.5 m/s is true for 91% of launches
# and 20% of creeps. When the model first asks for accel, the plan does not show the difference yet.
LAUNCH_PLAN_T = 3.0            # s
LAUNCH_PLAN_V_BP = [2.5, 4.0]  # m/s

# no boost behind a close lead, unless it pulls away
LEAD_DIST_BP = [5.0, 20.0]     # m
LEAD_VREL_BP = [0.0, 1.5]      # m/s


def get_launch_accel(a_e2e: float, v_ego: float, plan_t, plan_v, lead) -> float:
  """The model's accel, with more of it when the car pulls away for real. Braking and creeping are as they are."""
  if a_e2e <= LAUNCH_MIN_ACCEL or v_ego >= LAUNCH_SPEED_BP[-1] or len(plan_t) == 0:
    return a_e2e

  plan_scale = float(np.interp(np.interp(LAUNCH_PLAN_T, plan_t, plan_v), LAUNCH_PLAN_V_BP, [0.0, 1.0]))
  lead_scale = 1.0
  if lead.present:
    lead_scale = max(float(np.interp(lead.dRel, LEAD_DIST_BP, [0.0, 1.0])), float(np.interp(lead.vRel, LEAD_VREL_BP, [0.0, 1.0])))

  gain = float(np.interp(v_ego, LAUNCH_SPEED_BP, LAUNCH_GAIN_V))
  return a_e2e * (1.0 + (gain - 1.0) * plan_scale * lead_scale)
