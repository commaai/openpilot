import math
import numpy as np
from dataclasses import dataclass
from opendbc.car import structs, rate_limit, DT_CTRL, ACCELERATION_DUE_TO_GRAVITY
from opendbc.car.vehicle_model import VehicleModel

FRICTION_THRESHOLD = 0.2

# - ISO 11270
ISO_LATERAL_ACCEL = 3.0  # m/s^2
ISO_LATERAL_JERK = 5.0  # m/s^3

# - Common angle/curvature safety limits
# Add extra tolerance for average banked road since safety doesn't have the roll
AVERAGE_ROAD_ROLL = 0.06  # ~3.4 degrees, 6% superelevation. higher actual roll lowers lateral acceleration
MAX_LATERAL_ACCEL = ISO_LATERAL_ACCEL + (ACCELERATION_DUE_TO_GRAVITY * AVERAGE_ROAD_ROLL)  # ~3.6 m/s^2
# Lower than ISO 11270 lateral jerk limit (5.0 m/s^3) with bank tolerance, matches safety MAX_LATERAL_JERK
MAX_LATERAL_JERK = 3.0 + (ACCELERATION_DUE_TO_GRAVITY * AVERAGE_ROAD_ROLL)  # ~3.6 m/s^3


# TODO: deprecate in favor of vehicle-model-based limiting
#  (need to solve cars having different steering ratios, etc.)
@dataclass
class AngleSteeringLimits:
  # uses apply_std_steer_angle_limits
  STEER_ANGLE_MAX: float
  ANGLE_RATE_LIMIT_UP: tuple[list[float], list[float]]
  ANGLE_RATE_LIMIT_DOWN: tuple[list[float], list[float]]


@dataclass
class AngleSteeringLimitsVM:
  # uses apply_steer_angle_limits_vm
  # Max accepted by the EPS
  STEER_ANGLE_MAX: float
  MAX_LATERAL_ACCEL: float = MAX_LATERAL_ACCEL
  MAX_LATERAL_JERK: float = MAX_LATERAL_JERK
  # Used for comfort or to prevent faults at low speed
  MAX_ANGLE_RATE: float = math.inf


@dataclass
class CurvatureSteeringLimits:
  # Max accepted by the EPS
  CURVATURE_MAX: float
  MAX_LATERAL_ACCEL: float = MAX_LATERAL_ACCEL
  MAX_LATERAL_JERK: float = MAX_LATERAL_JERK

  def apply_limits(self, apply_curvature: float, apply_curvature_last: float, v_ego: float, curvature: float,
                   lat_active: bool, steer_step: int) -> float:
    """Apply accel and jerk constraints to curvature."""
    v_ego = max(v_ego, 1)

    # *** max lateral accel limit ***
    max_curvature = self.MAX_LATERAL_ACCEL / (v_ego ** 2)
    new_apply_curvature = float(np.clip(apply_curvature, -max_curvature, max_curvature))

    # *** max lateral jerk limit ***
    max_jerk = (self.MAX_LATERAL_JERK / (v_ego ** 2)) * (steer_step * DT_CTRL)
    new_apply_curvature = float(np.clip(new_apply_curvature, apply_curvature_last - max_jerk, apply_curvature_last + max_jerk))

    # curvature is current curvature when inactive
    if not lat_active:
      new_apply_curvature = curvature

    # prevent fault
    return float(np.clip(new_apply_curvature, -self.CURVATURE_MAX, self.CURVATURE_MAX))


def apply_driver_steer_torque_limits(apply_torque: int, apply_torque_last: int, driver_torque: float, LIMITS, steer_max: int | None = None):
  # some safety modes utilize a dynamic max steer
  if steer_max is None:
    steer_max = LIMITS.STEER_MAX

  # limits due to driver torque
  driver_max_torque = steer_max + (LIMITS.STEER_DRIVER_ALLOWANCE + driver_torque * LIMITS.STEER_DRIVER_FACTOR) * LIMITS.STEER_DRIVER_MULTIPLIER
  driver_min_torque = -steer_max + (-LIMITS.STEER_DRIVER_ALLOWANCE + driver_torque * LIMITS.STEER_DRIVER_FACTOR) * LIMITS.STEER_DRIVER_MULTIPLIER
  max_steer_allowed = max(min(steer_max, driver_max_torque), 0)
  min_steer_allowed = min(max(-steer_max, driver_min_torque), 0)
  apply_torque = np.clip(apply_torque, min_steer_allowed, max_steer_allowed)

  # slow rate if steer torque increases in magnitude
  if apply_torque_last > 0:
    apply_torque = np.clip(apply_torque, max(apply_torque_last - LIMITS.STEER_DELTA_DOWN, -LIMITS.STEER_DELTA_UP),
                           apply_torque_last + LIMITS.STEER_DELTA_UP)
  else:
    apply_torque = np.clip(apply_torque, apply_torque_last - LIMITS.STEER_DELTA_UP,
                           min(apply_torque_last + LIMITS.STEER_DELTA_DOWN, LIMITS.STEER_DELTA_UP))

  return int(round(float(apply_torque)))


def apply_dist_to_meas_limits(val, val_last, val_meas,
                              STEER_DELTA_UP, STEER_DELTA_DOWN,
                              STEER_ERROR_MAX, STEER_MAX):
  # limits due to comparison of commanded val VS measured val (torque/angle/curvature)
  max_lim = min(max(val_meas + STEER_ERROR_MAX, STEER_ERROR_MAX), STEER_MAX)
  min_lim = max(min(val_meas - STEER_ERROR_MAX, -STEER_ERROR_MAX), -STEER_MAX)

  val = np.clip(val, min_lim, max_lim)

  # slow rate if val increases in magnitude
  if val_last > 0:
    val = np.clip(val,
                  max(val_last - STEER_DELTA_DOWN, -STEER_DELTA_UP),
                  val_last + STEER_DELTA_UP)
  else:
    val = np.clip(val,
                  val_last - STEER_DELTA_UP,
                  min(val_last + STEER_DELTA_DOWN, STEER_DELTA_UP))

  return float(val)


def apply_meas_steer_torque_limits(apply_torque, apply_torque_last, motor_torque, LIMITS):
  return int(round(apply_dist_to_meas_limits(apply_torque, apply_torque_last, motor_torque,
                                             LIMITS.STEER_DELTA_UP, LIMITS.STEER_DELTA_DOWN,
                                             LIMITS.STEER_ERROR_MAX, LIMITS.STEER_MAX)))


def apply_std_steer_angle_limits(apply_angle: float, apply_angle_last: float, v_ego: float, steering_angle: float,
                                 lat_active: bool, limits: AngleSteeringLimits) -> float:
  # pick angle rate limits based on wind up/down
  steer_up = apply_angle_last * apply_angle >= 0. and abs(apply_angle) > abs(apply_angle_last)
  rate_limits = limits.ANGLE_RATE_LIMIT_UP if steer_up else limits.ANGLE_RATE_LIMIT_DOWN

  angle_rate_lim = np.interp(v_ego, rate_limits[0], rate_limits[1])
  new_apply_angle = np.clip(apply_angle, apply_angle_last - angle_rate_lim, apply_angle_last + angle_rate_lim)

  # angle is current steering wheel angle when inactive on all angle cars
  if not lat_active:
    new_apply_angle = steering_angle

  return float(np.clip(new_apply_angle, -limits.STEER_ANGLE_MAX, limits.STEER_ANGLE_MAX))


def get_max_angle_delta_vm(v_ego_raw: float, VM: VehicleModel, limits):
  """Calculate the maximum steering angle rate based on lateral jerk limits."""
  max_curvature_rate_sec = limits.ANGLE_LIMITS.MAX_LATERAL_JERK / (v_ego_raw ** 2)  # (1/m)/s
  max_angle_rate_sec = math.degrees(VM.get_steer_from_curvature(max_curvature_rate_sec, v_ego_raw, 0))  # deg/s
  return max_angle_rate_sec * (DT_CTRL * limits.STEER_STEP)


def get_max_angle_vm(v_ego_raw: float, VM: VehicleModel, limits):
  """Calculate the maximum steering angle based on lateral acceleration limits."""
  max_curvature = limits.ANGLE_LIMITS.MAX_LATERAL_ACCEL / (v_ego_raw ** 2)  # 1/m
  return math.degrees(VM.get_steer_from_curvature(max_curvature, v_ego_raw, 0))  # deg


def apply_steer_angle_limits_vm(apply_angle: float, apply_angle_last: float, v_ego_raw: float, steering_angle: float,
                                lat_active: bool, limits, VM: VehicleModel) -> float:
  """Apply jerk, accel, and safety limit constraints to steering angle."""
  v_ego_raw = max(v_ego_raw, 1)

  # *** max lateral jerk limit ***
  max_angle_delta = get_max_angle_delta_vm(v_ego_raw, VM, limits)

  # prevent fault/low speed comfort
  max_angle_delta = min(max_angle_delta, limits.ANGLE_LIMITS.MAX_ANGLE_RATE)
  new_apply_angle = rate_limit(apply_angle, apply_angle_last, -max_angle_delta, max_angle_delta)

  # *** max lateral accel limit ***
  max_angle = get_max_angle_vm(v_ego_raw, VM, limits)
  new_apply_angle = np.clip(new_apply_angle, -max_angle, max_angle)

  # angle is current angle when inactive
  if not lat_active:
    new_apply_angle = steering_angle

  # prevent fault
  return float(np.clip(new_apply_angle, -limits.ANGLE_LIMITS.STEER_ANGLE_MAX, limits.ANGLE_LIMITS.STEER_ANGLE_MAX))


def common_fault_avoidance(fault_condition: bool, request: bool, above_limit_frames: int,
                           max_above_limit_frames: int, max_mismatching_frames: int = 1):
  """
  Several cars have the ability to work around their EPS limits by cutting the
  request bit of their LKAS message after a certain number of frames above the limit.
  """

  # Count up to max_above_limit_frames, at which point we need to cut the request for above_limit_frames to avoid a fault
  if request and fault_condition:
    above_limit_frames += 1
  else:
    above_limit_frames = 0

  # Once we cut the request bit, count additionally to max_mismatching_frames before setting the request bit high again.
  # Some brands do not respect our workaround without multiple messages on the bus, for example
  if above_limit_frames > max_above_limit_frames:
    request = False

  if above_limit_frames >= max_above_limit_frames + max_mismatching_frames:
    above_limit_frames = 0

  return above_limit_frames, request


def apply_center_deadzone(error, deadzone):
  if (error > - deadzone) and (error < deadzone):
    error = 0.
  return error


def get_friction(lateral_accel_error: float, lateral_accel_deadzone: float, friction_threshold: float,
                 torque_params: structs.CarParams.LateralTorqueTuning) -> float:
  # TODO torque params' friction should be in lat accel space, not torque space
  friction_interp = np.interp(
    apply_center_deadzone(lateral_accel_error, lateral_accel_deadzone),
    [-friction_threshold, friction_threshold],
    [-torque_params.friction * torque_params.latAccelFactor, torque_params.friction * torque_params.latAccelFactor]
  )
  return float(friction_interp)
