import numpy as np
from cereal import log
from openpilot.common.realtime import DT_CTRL

MIN_SPEED = 1.0
CONTROL_N = 17
CAR_ROTATION_RADIUS = 0.0
# This is a turn radius smaller than most cars can achieve
MAX_CURVATURE = 0.2
MAX_VEL_ERR = 5.0  # m/s

# EU guidelines
MAX_LATERAL_JERK = 5.0  # m/s^3
MAX_LATERAL_ACCEL = 3.0  # m/s^2


def clip_curvature(v_ego, prev_curvature, new_curvature):
  v_ego = max(MIN_SPEED, v_ego)
  max_curvature = min(MAX_LATERAL_ACCEL / v_ego ** 2, MAX_CURVATURE)
  new_curvature = np.clip(new_curvature, -max_curvature, max_curvature)
  max_curvature_rate = MAX_LATERAL_JERK / (v_ego ** 2)  # inexact calculation, check https://github.com/commaai/openpilot/pull/24755
  safe_desired_curvature = np.clip(new_curvature,
                                   prev_curvature - max_curvature_rate * DT_CTRL,
                                   prev_curvature + max_curvature_rate * DT_CTRL)

  return safe_desired_curvature


def get_speed_error(modelV2: log.ModelDataV2, v_ego: float) -> float:
  # ToDo: Try relative error, and absolute speed
  if len(modelV2.temporalPose.trans):
    vel_err = np.clip(modelV2.temporalPose.trans[0] - v_ego, -MAX_VEL_ERR, MAX_VEL_ERR)
    return float(vel_err)
  return 0.0
