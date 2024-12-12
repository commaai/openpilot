# functions common among cars
import logging
from collections import namedtuple
from dataclasses import dataclass, field
from enum import IntFlag, ReprEnum, StrEnum, EnumType, auto
from dataclasses import replace

from panda import uds
from opendbc.car import structs
from opendbc.car.can_definitions import CanData
from opendbc.car.docs_definitions import CarDocs, ExtraCarDocs
from opendbc.car.common.numpy_fast import clip, interp

# set up logging
carlog = logging.getLogger('carlog')
carlog.setLevel(logging.INFO)
carlog.propagate = False

DT_CTRL = 0.01  # car state and control loop timestep (s)

# kg of standard extra cargo to count for drive, gas, etc...
STD_CARGO_KG = 136.

ACCELERATION_DUE_TO_GRAVITY = 9.81  # m/s^2

ButtonType = structs.CarState.ButtonEvent.Type
AngleRateLimit = namedtuple('AngleRateLimit', ['speed_bp', 'angle_v'])


def apply_hysteresis(val: float, val_steady: float, hyst_gap: float) -> float:
  if val > val_steady + hyst_gap:
    val_steady = val - hyst_gap
  elif val < val_steady - hyst_gap:
    val_steady = val + hyst_gap
  return val_steady


def create_button_events(cur_btn: int, prev_btn: int, buttons_dict: dict[int, structs.CarState.ButtonEvent.Type],
                         unpressed_btn: int = 0) -> list[structs.CarState.ButtonEvent]:
  events: list[structs.CarState.ButtonEvent] = []

  if cur_btn == prev_btn:
    return events

  # Add events for button presses, multiple when a button switches without going to unpressed
  for pressed, btn in ((False, prev_btn), (True, cur_btn)):
    if btn != unpressed_btn:
      events.append(structs.CarState.ButtonEvent(pressed=pressed,
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


DbcDict = dict[StrEnum, str]

class Bus(StrEnum):
  pt = auto()
  cam = auto()
  radar = auto()
  adas = auto()
  alt = auto()
  body = auto()
  chassis = auto()
  loopback = auto()
  main = auto()
  party = auto()
  ap_party = auto()


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


def apply_center_deadzone(error, deadzone):
  if (error > - deadzone) and (error < deadzone):
    error = 0.
  return error


def rate_limit(new_value, last_value, dw_step, up_step):
  return clip(new_value, last_value + dw_step, last_value + up_step)


def get_friction(lateral_accel_error: float, lateral_accel_deadzone: float, friction_threshold: float,
                 torque_params: structs.CarParams.LateralTorqueTuning, friction_compensation: bool) -> float:
  friction_interp = interp(
    apply_center_deadzone(lateral_accel_error, lateral_accel_deadzone),
    [-friction_threshold, friction_threshold],
    [-torque_params.friction, torque_params.friction]
  )
  friction = float(friction_interp) if friction_compensation else 0.0
  return friction


def make_tester_present_msg(addr, bus, subaddr=None, suppress_response=False):
  dat = [0x02, uds.SERVICE_TYPE.TESTER_PRESENT]
  if subaddr is not None:
    dat.insert(0, subaddr)
  dat.append(0x80 if suppress_response else 0x0)  # sub-function

  dat.extend([0x0] * (8 - len(dat)))
  return CanData(addr, bytes(dat), bus)


def get_safety_config(safety_model: structs.CarParams.SafetyModel, safety_param: int = None) -> structs.CarParams.SafetyConfig:
  ret = structs.CarParams.SafetyConfig()
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


class Freezable:
  _frozen: bool = False

  def freeze(self):
    if not self._frozen:
      self._frozen = True

  def __setattr__(self, *args, **kwargs):
    if self._frozen:
      raise Exception("cannot modify frozen object")
    super().__setattr__(*args, **kwargs)


@dataclass(order=True)
class PlatformConfigBase(Freezable):
  car_docs: list[CarDocs] | list[ExtraCarDocs]
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


@dataclass(order=True)
class PlatformConfig(PlatformConfigBase):
  car_docs: list[CarDocs]
  specs: CarSpecs
  dbc_dict: DbcDict


@dataclass(order=True)
class ExtraPlatformConfig(PlatformConfigBase):
  car_docs: list[ExtraCarDocs]
  specs: CarSpecs = CarSpecs(mass=0., wheelbase=0., steerRatio=0.)
  dbc_dict: DbcDict = field(default_factory=lambda: dict())


class PlatformsType(EnumType):
  def __new__(metacls, cls, bases, classdict, *, boundary=None, _simple=False, **kwds):
    for key in classdict._member_names.keys():
      cfg: PlatformConfig = classdict[key]
      cfg.platform_str = key
      cfg.freeze()
    return super().__new__(metacls, cls, bases, classdict, boundary=boundary, _simple=_simple, **kwds)


class Platforms(str, ReprEnum, metaclass=PlatformsType):
  config: PlatformConfigBase

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
