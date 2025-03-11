import math
import numpy as np

from cereal import log
from openpilot.selfdrive.controls.lib.latcontrol import LatControl

STEER_ANGLE_SATURATION_THRESHOLD = 2.5  # Degrees

# Speed-based scaling configuration for steering responsiveness
#  As speed increases, we apply more of the desired steering angle change
SPEED_BREAKPOINTS_MS = [0, 3, 6, 10]  # Speed breakpoints in m/s
STEERING_FACTOR_AT_SPEED = [0, 0.2, 0.45, 1.0]  # Corresponding steering influence factors
# 0 = ignore desired angle (keep current steering), 1.0 = fully apply desired angle change

class LatControlAngle(LatControl):
  def __init__(self, CP, CI):
    super().__init__(CP, CI)
    self.sat_check_min_speed = 5.

  def update(self, active, CS, VM, params, steer_limited_by_controls, desired_curvature, calibrated_pose, curvature_limited):
    angle_log = log.ControlsState.LateralAngleState.new_message()

    if not active:
      angle_log.active = False
      angle_steers_des = float(CS.steeringAngleDeg)
    else:
      angle_log.active = True

      # Compute the fully desired steering angle based on curvature
      raw_angle_steers_des = math.degrees(VM.get_steer_from_curvature(-desired_curvature, CS.vEgo, params.roll))
      raw_angle_steers_des += params.angleOffsetDeg  # Apply calibration offset

      # Calculate the scaling factor. At lower speeds, we apply less of the desired steering change to avoid jerky movements
      speed_scaling_factor = np.interp(CS.vEgo, SPEED_BREAKPOINTS_MS, STEERING_FACTOR_AT_SPEED)
      angle_steers_des = CS.steeringAngleDeg + speed_scaling_factor * (raw_angle_steers_des - CS.steeringAngleDeg)

    angle_control_saturated = abs(angle_steers_des - CS.steeringAngleDeg) > STEER_ANGLE_SATURATION_THRESHOLD
    angle_log.saturated = bool(self._check_saturation(angle_control_saturated, CS, False, curvature_limited))
    angle_log.steeringAngleDeg = float(CS.steeringAngleDeg)
    angle_log.steeringAngleDesiredDeg = float(angle_steers_des)

    return 0, float(angle_steers_des), angle_log
