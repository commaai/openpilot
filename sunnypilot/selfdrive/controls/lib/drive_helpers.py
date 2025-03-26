from numpy import clip, interp
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, MIN_SPEED, MAX_LATERAL_JERK
from openpilot.sunnypilot.modeld.constants import ModelConstants


def get_lag_adjusted_curvature(steer_delay, v_ego, psis, curvatures):
  if len(psis) != CONTROL_N:
    psis = [0.0]*CONTROL_N
    curvatures = [0.0]*CONTROL_N
  v_ego = max(MIN_SPEED, v_ego)

  # MPC can plan to turn the wheel and turn back before t_delay. This means
  # in high delay cases some corrections never even get commanded. So just use
  # psi to calculate a simple linearization of desired curvature
  current_curvature_desired = curvatures[0]
  psi = interp(steer_delay, ModelConstants.T_IDXS[:CONTROL_N], psis)
  average_curvature_desired = psi / (v_ego * steer_delay)
  desired_curvature = 2 * average_curvature_desired - current_curvature_desired

  # This is the "desired rate of the setpoint" not an actual desired rate
  max_curvature_rate = MAX_LATERAL_JERK / (v_ego**2) # inexact calculation, check https://github.com/commaai/openpilot/pull/24755
  safe_desired_curvature = clip(desired_curvature,
                                current_curvature_desired - max_curvature_rate * DT_MDL,
                                current_curvature_desired + max_curvature_rate * DT_MDL)

  return float(safe_desired_curvature)
