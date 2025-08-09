import math
import numpy as np
from dataclasses import dataclass
from opendbc.car import structs, rate_limit, DT_CTRL
from opendbc.car.vehicle_model import VehicleModel

FRICTION_THRESHOLD = 0.3

# Add extra tolerance for average banked road since safety doesn't have the roll
AVERAGE_ROAD_ROLL = 0.06  # ~3.4 degrees, 6% superelevation. higher actual roll lowers lateral acceleration

# ISO 11270
ISO_LATERAL_ACCEL = 3.0  # m/s^2
ISO_LATERAL_JERK = 5.0  # m/s^3


@dataclass
class AngleSteeringLimits:
  STEER_ANGLE_MAX: float
  ANGLE_RATE_LIMIT_UP: tuple[list[float], list[float]]
  ANGLE_RATE_LIMIT_DOWN: tuple[list[float], list[float]]
  MAX_LATERAL_ACCEL: float = ISO_LATERAL_ACCEL * .7  # ~2.1m/s^2, 70% ISO limits as default, but ideally this is set manually for comfortable limits?
  MAX_LATERAL_JERK: float = ISO_LATERAL_JERK * .5  # ~2.5m/s^3, 50% of ISO limits as default, but ideally this is set manually for comfortable limits?


def apply_driver_steer_torque_limits(apply_torque: int, apply_torque_last: int, driver_torque: float, LIMITS, steer_max: int = None):
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
  friction_interp = np.interp(
    apply_center_deadzone(lateral_accel_error, lateral_accel_deadzone),
    [-friction_threshold, friction_threshold],
    [-torque_params.friction, torque_params.friction]
  )
  return float(friction_interp)


def get_max_angle_delta_vm(v_ego_raw: float, steer_step: int, VM: VehicleModel, max_lateral_jerk: float):
  """ Calculate the maximum steering angle rate based on lateral jerk limits. """
  max_curvature_rate_sec = max_lateral_jerk / (v_ego_raw ** 2)  # (1/m)/s
  max_angle_rate_sec = math.degrees(VM.get_steer_from_curvature(max_curvature_rate_sec, v_ego_raw, 0))  # deg/s
  return max_angle_rate_sec * (DT_CTRL * steer_step)


def get_max_angle_vm(v_ego_raw: float, VM: VehicleModel, max_lateral_accel: float):
  """ Calculate the maximum steering angle based on lateral acceleration limits. """
  max_curvature = max_lateral_accel / (v_ego_raw ** 2)  # 1/m
  return math.degrees(VM.get_steer_from_curvature(max_curvature, v_ego_raw, 0))  # deg


def apply_vm_steer_angle_limits(apply_angle: float, apply_angle_last: float, v_ego_raw: float, steering_angle: float, lat_active: bool,
                                limits, VM: VehicleModel, MAX_ANGLE_RATE: float = None) -> float:
  """Apply rate, accel, and safety limit constraints to steering angle."""
  v_ego_safe = max(v_ego_raw, 1.)

  # max lateral jerk limit
  max_angle_delta = get_max_angle_delta_vm(v_ego_safe, limits.STEER_STEP, VM, limits.ANGLE_LIMITS.MAX_LATERAL_JERK)

  # this is to prevent a fault on cars that need it, like Tesla
  if MAX_ANGLE_RATE is not None:
    max_angle_delta = min(max_angle_delta, MAX_ANGLE_RATE)

  new_apply_angle = rate_limit(apply_angle, apply_angle_last, -max_angle_delta, max_angle_delta)

  # max lateral accel limit
  max_angle = get_max_angle_vm(v_ego_safe, VM, limits.ANGLE_LIMITS.MAX_LATERAL_ACCEL)
  new_apply_angle = np.clip(new_apply_angle, -max_angle, max_angle)

  # angle is current angle when inactive
  if not lat_active:
    new_apply_angle = steering_angle

  # final hard limit to prevent fault
  return float(np.clip(new_apply_angle, -limits.ANGLE_LIMITS.STEER_ANGLE_MAX, limits.ANGLE_LIMITS.STEER_ANGLE_MAX))
