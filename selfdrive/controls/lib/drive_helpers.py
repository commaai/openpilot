import numpy as np
from cereal import log
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.vehicle_model import ACCELERATION_DUE_TO_GRAVITY

MIN_SPEED = 1.0
CONTROL_N = 17
CAR_ROTATION_RADIUS = 0.0
# This is a turn radius smaller than most cars can achieve
MAX_CURVATURE = 0.2
MAX_VEL_ERR = 5.0  # m/s

# EU guidelines
MAX_LATERAL_JERK = 5.0  # m/s^3
MAX_LATERAL_ACCEL_NOROLL = 3.0  # m/s^2


def clamp(val, min_val, max_val):
    clamped_val = min(max(val, min_val), max_val)
    was_clamped = clamped_val != val
    return clamped_val, was_clamped


def clip_curvature(v_ego, prev_curvature, new_curvature, roll):
  v_ego = max(MIN_SPEED, v_ego)
  roll_compensation = roll * ACCELERATION_DUE_TO_GRAVITY
  max_lat_accel = MAX_LATERAL_ACCEL_NOROLL + roll_compensation
  min_lat_accel = -MAX_LATERAL_ACCEL_NOROLL + roll_compensation
  clipped_curv, was_clipped_iso = clamp(new_curvature, min_lat_accel / v_ego ** 2, max_lat_accel / v_ego ** 2)
  new_curvature, was_clipped_max_curv = clamp(clipped_curv, -MAX_CURVATURE, MAX_CURVATURE)
  was_clipped = was_clipped_iso or was_clipped_max_curv
  max_curvature_rate = MAX_LATERAL_JERK / (v_ego ** 2)  # inexact calculation, check https://github.com/commaai/openpilot/pull/24755
  # Don't flag curv_rate clips, not important for experience
  safe_desired_curvature, _ = clamp(new_curvature,
                                   prev_curvature - max_curvature_rate * DT_CTRL,
                                   prev_curvature + max_curvature_rate * DT_CTRL)
  return safe_desired_curvature, was_clipped


def get_speed_error(modelV2: log.ModelDataV2, v_ego: float) -> float:
  # ToDo: Try relative error, and absolute speed
  if len(modelV2.temporalPose.trans):
    vel_err = np.clip(modelV2.temporalPose.trans[0] - v_ego, -MAX_VEL_ERR, MAX_VEL_ERR)
    return float(vel_err)
  return 0.0
