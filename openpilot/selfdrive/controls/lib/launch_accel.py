import numpy as np

from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import get_safe_obstacle_distance, get_stopped_equivalence_factor

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

# Following a lead that pulls away (e.g. stop-and-go): the model is often slow to close the gap. Boost its accel while
# the lead moves away and the gap is more than the MPC follow distance. The MPC still limits the accel for the lead.
PULL_SPEED_BP = [4.0, 6.0, 18.0, 22.0]  # m/s
PULL_SPEED_V = [0.0, 1.0, 1.0, 0.0]
PULL_GAIN = 1.3
PULL_MIN_ACCEL = 0.1           # m/s², the model must ask for at least this much accel
PULL_VREL_BP = [0.5, 1.5]      # m/s, how fast the lead moves away
PULL_GAP_BP = [1.0, 1.3]       # gap / MPC follow distance


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


def get_pull_away_accel(a_e2e: float, v_ego: float, lead, t_follow: float) -> float:
  """The model's accel, with more of it when a lead in front pulls away and the gap is large. Braking is as it is."""
  if a_e2e <= PULL_MIN_ACCEL or not lead.present:
    return a_e2e

  # the same follow distance as the MPC: the obstacle distance at this speed, minus what the lead needs to stop
  follow_dist = get_safe_obstacle_distance(v_ego, t_follow) - get_stopped_equivalence_factor(max(lead.vLead, 0.0))
  gap_ratio = lead.dRel / max(follow_dist, 1.0)
  scale = (float(np.interp(v_ego, PULL_SPEED_BP, PULL_SPEED_V)) *
           float(np.interp(lead.vRel, PULL_VREL_BP, [0.0, 1.0])) *
           float(np.interp(gap_ratio, PULL_GAP_BP, [0.0, 1.0])))
  return a_e2e * (1.0 + (PULL_GAIN - 1.0) * scale)
