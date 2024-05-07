# functions common among cars
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from enum import IntFlag, ReprEnum, EnumType
from dataclasses import replace

import capnp

from cereal import car
from openpilot.common.numpy_fast import clip, interp
from openpilot.common.utils import Freezable
from openpilot.selfdrive.car.docs_definitions import CarDocs


# kg of standard extra cargo to count for drive, gas, etc...
STD_CARGO_KG = 136.

ButtonType = car.CarState.ButtonEvent.Type
EventName = car.CarEvent.EventName
AngleRateLimit = namedtuple('AngleRateLimit', ['speed_bp', 'angle_v'])


def apply_hysteresis(val: float, val_steady: float, hyst_gap: float) -> float:
  if val > val_steady + hyst_gap:
    val_steady = val - hyst_gap
  elif val < val_steady - hyst_gap:
    val_steady = val + hyst_gap
  return val_steady


def create_button_events(cur_btn: int, prev_btn: int, buttons_dict: dict[int, capnp.lib.capnp._EnumModule],
                         unpressed_btn: int = 0) -> list[capnp.lib.capnp._DynamicStructBuilder]:
  events: list[capnp.lib.capnp._DynamicStructBuilder] = []

  if cur_btn == prev_btn:
    return events

  # Add events for button presses, multiple when a button switches without going to unpressed
  for pressed, btn in ((False, prev_btn), (True, cur_btn)):
    if btn != unpressed_btn:
      events.append(car.CarState.ButtonEvent(pressed=pressed,
                                             type=buttons_dict.get(btn, ButtonType.unknown)))
  return events


def gen_empty_fingerprint():
  return {i: {} for i in range(8)}


# these params were derived for the Civic and used to calculate params for other cars
class VehicleDynamicsParams:
  MASS = 1326. + STD_CARGO_KG
  WHEELBASE = 2.70
  CENTER_TO_FRONT = WHEELBASE * 0.4
  CENTER_TO_REAR = WHEELBASE - CENTER_TO_FRONT
  ROTATIONAL_INERTIA = 2500
  TIRE_STIFFNESS_FRONT = 192150
  TIRE_STIFFNESS_REAR = 202500


# TODO: get actual value, for now starting with reasonable value for
# civic and scaling by mass and wheelbase
def scale_rot_inertia(mass, wheelbase):
  return VehicleDynamicsParams.ROTATIONAL_INERTIA * mass * wheelbase ** 2 / (VehicleDynamicsParams.MASS * VehicleDynamicsParams.WHEELBASE ** 2)


# TODO: start from empirically derived lateral slip stiffness for the civic and scale by
# mass and CG position, so all cars will have approximately similar dyn behaviors
def scale_tire_stiffness(mass, wheelbase, center_to_front, tire_stiffness_factor):
  center_to_rear = wheelbase - center_to_front
  tire_stiffness_front = (VehicleDynamicsParams.TIRE_STIFFNESS_FRONT * tire_stiffness_factor) * mass / VehicleDynamicsParams.MASS * \
                         (center_to_rear / wheelbase) / (VehicleDynamicsParams.CENTER_TO_REAR / VehicleDynamicsParams.WHEELBASE)

  tire_stiffness_rear = (VehicleDynamicsParams.TIRE_STIFFNESS_REAR * tire_stiffness_factor) * mass / VehicleDynamicsParams.MASS * \
                        (center_to_front / wheelbase) / (VehicleDynamicsParams.CENTER_TO_FRONT / VehicleDynamicsParams.WHEELBASE)

  return tire_stiffness_front, tire_stiffness_rear


DbcDict = dict[str, str]


def dbc_dict(pt_dbc, radar_dbc, chassis_dbc=None, body_dbc=None) -> DbcDict:
  return {'pt': pt_dbc, 'radar': radar_dbc, 'chassis': chassis_dbc, 'body': body_dbc}


def apply_driver_steer_torque_limits(apply_torque, apply_torque_last, driver_torque, LIMITS):

  # limits due to driver torque
  driver_max_torque = LIMITS.STEER_MAX + (LIMITS.STEER_DRIVER_ALLOWANCE + driver_torque * LIMITS.STEER_DRIVER_FACTOR) * LIMITS.STEER_DRIVER_MULTIPLIER
  driver_min_torque = -LIMITS.STEER_MAX + (-LIMITS.STEER_DRIVER_ALLOWANCE + driver_torque * LIMITS.STEER_DRIVER_FACTOR) * LIMITS.STEER_DRIVER_MULTIPLIER
  max_steer_allowed = max(min(LIMITS.STEER_MAX, driver_max_torque), 0)
  min_steer_allowed = min(max(-LIMITS.STEER_MAX, driver_min_torque), 0)
  apply_torque = clip(apply_torque, min_steer_allowed, max_steer_allowed)

  # slow rate if steer torque increases in magnitude
  if apply_torque_last > 0:
    apply_torque = clip(apply_torque, max(apply_torque_last - LIMITS.STEER_DELTA_DOWN, -LIMITS.STEER_DELTA_UP),
                        apply_torque_last + LIMITS.STEER_DELTA_UP)
  else:
    apply_torque = clip(apply_torque, apply_torque_last - LIMITS.STEER_DELTA_UP,
                        min(apply_torque_last + LIMITS.STEER_DELTA_DOWN, LIMITS.STEER_DELTA_UP))

  return int(round(float(apply_torque)))


def apply_dist_to_meas_limits(val, val_last, val_meas,
                              STEER_DELTA_UP, STEER_DELTA_DOWN,
                              STEER_ERROR_MAX, STEER_MAX):
  # limits due to comparison of commanded val VS measured val (torque/angle/curvature)
  max_lim = min(max(val_meas + STEER_ERROR_MAX, STEER_ERROR_MAX), STEER_MAX)
  min_lim = max(min(val_meas - STEER_ERROR_MAX, -STEER_ERROR_MAX), -STEER_MAX)

  val = clip(val, min_lim, max_lim)

  # slow rate if val increases in magnitude
  if val_last > 0:
    val = clip(val,
               max(val_last - STEER_DELTA_DOWN, -STEER_DELTA_UP),
               val_last + STEER_DELTA_UP)
  else:
    val = clip(val,
               val_last - STEER_DELTA_UP,
               min(val_last + STEER_DELTA_DOWN, STEER_DELTA_UP))

  return float(val)


def apply_meas_steer_torque_limits(apply_torque, apply_torque_last, motor_torque, LIMITS):
  return int(round(apply_dist_to_meas_limits(apply_torque, apply_torque_last, motor_torque,
                                             LIMITS.STEER_DELTA_UP, LIMITS.STEER_DELTA_DOWN,
                                             LIMITS.STEER_ERROR_MAX, LIMITS.STEER_MAX)))


def apply_std_steer_angle_limits(apply_angle, apply_angle_last, v_ego, LIMITS):
  # pick angle rate limits based on wind up/down
  steer_up = apply_angle_last * apply_angle >= 0. and abs(apply_angle) > abs(apply_angle_last)
  rate_limits = LIMITS.ANGLE_RATE_LIMIT_UP if steer_up else LIMITS.ANGLE_RATE_LIMIT_DOWN

  angle_rate_lim = interp(v_ego, rate_limits.speed_bp, rate_limits.angle_v)
  return clip(apply_angle, apply_angle_last - angle_rate_lim, apply_angle_last + angle_rate_lim)


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


def make_can_msg(addr, dat, bus):
  return [addr, 0, dat, bus]


def get_safety_config(safety_model, safety_param = None):
  ret = car.CarParams.SafetyConfig.new_message()
  ret.safetyModel = safety_model
  if safety_param is not None:
    ret.safetyParam = safety_param
  return ret


class CanBusBase:
  offset: int

  def __init__(self, CP, fingerprint: dict[int, dict[int, int]] | None) -> None:
    if CP is None:
      assert fingerprint is not None
      num = max([k for k, v in fingerprint.items() if len(v)], default=0) // 4 + 1
    else:
      num = len(CP.safetyConfigs)
    self.offset = 4 * (num - 1)


class CanSignalRateCalculator:
  """
  Calculates the instantaneous rate of a CAN signal by using the counter
  variable and the known frequency of the CAN message that contains it.
  """
  def __init__(self, frequency):
    self.frequency = frequency
    self.previous_counter = 0
    self.previous_value = 0
    self.rate = 0

  def update(self, current_value, current_counter):
    if current_counter != self.previous_counter:
      self.rate = (current_value - self.previous_value) * self.frequency

    self.previous_counter = current_counter
    self.previous_value = current_value

    return self.rate


@dataclass(frozen=True, kw_only=True)
class CarSpecs:
  mass: float  # kg, curb weight
  wheelbase: float  # meters
  steerRatio: float
  centerToFrontRatio: float = 0.5
  minSteerSpeed: float = 0.0  # m/s
  minEnableSpeed: float = -1.0  # m/s
  tireStiffnessFactor: float = 1.0

  def override(self, **kwargs):
    return replace(self, **kwargs)


@dataclass(order=True)
class PlatformConfig(Freezable):
  car_docs: list[CarDocs]
  specs: CarSpecs

  dbc_dict: DbcDict

  flags: int = 0

  platform_str: str | None = None

  def __hash__(self) -> int:
    return hash(self.platform_str)

  def override(self, **kwargs):
    return replace(self, **kwargs)

  def init(self):
    pass

  def __post_init__(self):
    self.init()


class PlatformsType(EnumType):
  def __new__(metacls, cls, bases, classdict, *, boundary=None, _simple=False, **kwds):
    for key in classdict._member_names.keys():
      cfg: PlatformConfig = classdict[key]
      cfg.platform_str = key
      cfg.freeze()
    return super().__new__(metacls, cls, bases, classdict, boundary=boundary, _simple=_simple, **kwds)


class Platforms(str, ReprEnum, metaclass=PlatformsType):
  config: PlatformConfig

  def __new__(cls, platform_config: PlatformConfig):
    member = str.__new__(cls, platform_config.platform_str)
    member.config = platform_config
    member._value_ = platform_config.platform_str
    return member

  def __repr__(self):
    return f"<{self.__class__.__name__}.{self.name}>"

  @classmethod
  def create_dbc_map(cls) -> dict[str, DbcDict]:
    return {p: p.config.dbc_dict for p in cls}

  @classmethod
  def with_flags(cls, flags: IntFlag) -> set['Platforms']:
    return {p for p in cls if p.config.flags & flags}

  @classmethod
  def without_flags(cls, flags: IntFlag) -> set['Platforms']:
    return {p for p in cls if not (p.config.flags & flags)}

  @classmethod
  def print_debug(cls, flags):
    platforms_with_flag = defaultdict(list)
    for flag in flags:
      for platform in cls:
        if platform.config.flags & flag:
          assert flag.name is not None
          platforms_with_flag[flag.name].append(platform)

    for flag, platforms in platforms_with_flag.items():
      print(f"{flag:32s}: {', '.join(p.name for p in platforms)}")
