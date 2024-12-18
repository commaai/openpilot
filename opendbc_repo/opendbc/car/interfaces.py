import json
import os
import numpy as np
import time
import tomllib
from abc import abstractmethod, ABC
from enum import StrEnum
from typing import Any, NamedTuple
from collections.abc import Callable
from functools import cache

from opendbc.car import DT_CTRL, apply_hysteresis, gen_empty_fingerprint, scale_rot_inertia, scale_tire_stiffness, get_friction, STD_CARGO_KG
from opendbc.car import structs
from opendbc.car.can_definitions import CanData, CanRecvCallable, CanSendCallable
from opendbc.car.common.basedir import BASEDIR
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.common.simple_kalman import KF1D, get_kalman_gain
from opendbc.car.common.numpy_fast import clip
from opendbc.car.values import PLATFORMS
from opendbc.can.parser import CANParser

GearShifter = structs.CarState.GearShifter

V_CRUISE_MAX = 145
MAX_CTRL_SPEED = (V_CRUISE_MAX + 4) * CV.KPH_TO_MS
ACCEL_MAX = 2.0
ACCEL_MIN = -3.5
FRICTION_THRESHOLD = 0.3

TORQUE_PARAMS_PATH = os.path.join(BASEDIR, 'torque_data/params.toml')
TORQUE_OVERRIDE_PATH = os.path.join(BASEDIR, 'torque_data/override.toml')
TORQUE_SUBSTITUTE_PATH = os.path.join(BASEDIR, 'torque_data/substitute.toml')

GEAR_SHIFTER_MAP: dict[str, structs.CarState.GearShifter] = {
  'P': GearShifter.park, 'PARK': GearShifter.park,
  'R': GearShifter.reverse, 'REVERSE': GearShifter.reverse,
  'N': GearShifter.neutral, 'NEUTRAL': GearShifter.neutral,
  'E': GearShifter.eco, 'ECO': GearShifter.eco,
  'T': GearShifter.manumatic, 'MANUAL': GearShifter.manumatic,
  'D': GearShifter.drive, 'DRIVE': GearShifter.drive,
  'S': GearShifter.sport, 'SPORT': GearShifter.sport,
  'L': GearShifter.low, 'LOW': GearShifter.low,
  'B': GearShifter.brake, 'BRAKE': GearShifter.brake,
}


class LatControlInputs(NamedTuple):
  lateral_acceleration: float
  roll_compensation: float
  vego: float
  aego: float


TorqueFromLateralAccelCallbackType = Callable[[LatControlInputs, structs.CarParams.LateralTorqueTuning, float, float, bool, bool], float]


@cache
def get_torque_params():
  with open(TORQUE_SUBSTITUTE_PATH, 'rb') as f:
    sub = tomllib.load(f)
  with open(TORQUE_PARAMS_PATH, 'rb') as f:
    params = tomllib.load(f)
  with open(TORQUE_OVERRIDE_PATH, 'rb') as f:
    override = tomllib.load(f)

  torque_params = {}
  for candidate in (sub.keys() | params.keys() | override.keys()) - {'legend'}:
    if sum([candidate in x for x in [sub, params, override]]) > 1:
      raise RuntimeError(f'{candidate} is defined twice in torque config')

    sub_candidate = sub.get(candidate, candidate)

    if sub_candidate in override:
      out = override[sub_candidate]
    elif sub_candidate in params:
      out = params[sub_candidate]
    else:
      raise NotImplementedError(f"Did not find torque params for {sub_candidate}")

    torque_params[sub_candidate] = {key: out[i] for i, key in enumerate(params['legend'])}
    if candidate in sub:
      torque_params[candidate] = torque_params[sub_candidate]

  return torque_params

# generic car and radar interfaces

class CarInterfaceBase(ABC):
  def __init__(self, CP: structs.CarParams, CarController, CarState):
    self.CP = CP

    self.frame = 0
    self.v_ego_cluster_seen = False

    self.CS: CarStateBase = CarState(CP)
    self.can_parsers: dict[StrEnum, CANParser] = self.CS.get_can_parsers(CP)

    dbc_names = {bus: cp.dbc_name for bus, cp in self.can_parsers.items()}
    self.CC: CarControllerBase = CarController(dbc_names, CP)

  def apply(self, c: structs.CarControl, now_nanos: int | None = None) -> tuple[structs.CarControl.Actuators, list[CanData]]:
    if now_nanos is None:
      now_nanos = int(time.monotonic() * 1e9)
    return self.CC.update(c, self.CS, now_nanos)

  @staticmethod
  def get_pid_accel_limits(CP, current_speed, cruise_speed):
    return ACCEL_MIN, ACCEL_MAX

  @classmethod
  def get_non_essential_params(cls, candidate: str) -> structs.CarParams:
    """
    Parameters essential to controlling the car may be incomplete or wrong without FW versions or fingerprints.
    """
    return cls.get_params(candidate, gen_empty_fingerprint(), list(), False, False)

  @classmethod
  def get_params(cls, candidate: str, fingerprint: dict[int, dict[int, int]], car_fw: list[structs.CarParams.CarFw],
                 experimental_long: bool, docs: bool) -> structs.CarParams:
    ret = CarInterfaceBase.get_std_params(candidate)

    platform = PLATFORMS[candidate]
    ret.mass = platform.config.specs.mass
    ret.wheelbase = platform.config.specs.wheelbase
    ret.steerRatio = platform.config.specs.steerRatio
    ret.centerToFront = ret.wheelbase * platform.config.specs.centerToFrontRatio
    ret.minEnableSpeed = platform.config.specs.minEnableSpeed
    ret.minSteerSpeed = platform.config.specs.minSteerSpeed
    ret.tireStiffnessFactor = platform.config.specs.tireStiffnessFactor
    ret.flags |= int(platform.config.flags)

    ret = cls._get_params(ret, candidate, fingerprint, car_fw, experimental_long, docs)

    # Vehicle mass is published curb weight plus assumed payload such as a human driver; notCars have no assumed payload
    if not ret.notCar:
      ret.mass = ret.mass + STD_CARGO_KG

    # Set params dependent on values set by the car interface
    ret.rotationalInertia = scale_rot_inertia(ret.mass, ret.wheelbase)
    ret.tireStiffnessFront, ret.tireStiffnessRear = scale_tire_stiffness(ret.mass, ret.wheelbase, ret.centerToFront, ret.tireStiffnessFactor)

    return ret

  @staticmethod
  @abstractmethod
  def _get_params(ret: structs.CarParams, candidate, fingerprint: dict[int, dict[int, int]],
                  car_fw: list[structs.CarParams.CarFw], experimental_long: bool, docs: bool) -> structs.CarParams:
    raise NotImplementedError

  @staticmethod
  def init(CP: structs.CarParams, can_recv: CanRecvCallable, can_send: CanSendCallable):
    pass

  @staticmethod
  def get_steer_feedforward_default(desired_angle, v_ego):
    # Proportional to realigning tire momentum: lateral acceleration.
    return desired_angle * (v_ego**2)

  def get_steer_feedforward_function(self):
    return self.get_steer_feedforward_default

  def torque_from_lateral_accel_linear(self, latcontrol_inputs: LatControlInputs, torque_params: structs.CarParams.LateralTorqueTuning,
                                       lateral_accel_error: float, lateral_accel_deadzone: float, friction_compensation: bool, gravity_adjusted: bool) -> float:
    # The default is a linear relationship between torque and lateral acceleration (accounting for road roll and steering friction)
    friction = get_friction(lateral_accel_error, lateral_accel_deadzone, FRICTION_THRESHOLD, torque_params, friction_compensation)
    return (latcontrol_inputs.lateral_acceleration / float(torque_params.latAccelFactor)) + friction

  def torque_from_lateral_accel(self) -> TorqueFromLateralAccelCallbackType:
    return self.torque_from_lateral_accel_linear

  # returns a set of default params to avoid repetition in car specific params
  @staticmethod
  def get_std_params(candidate: str) -> structs.CarParams:
    ret = structs.CarParams()
    ret.carFingerprint = candidate

    # Car docs fields
    ret.maxLateralAccel = get_torque_params()[candidate]['MAX_LAT_ACCEL_MEASURED']
    ret.autoResumeSng = True  # describes whether car can resume from a stop automatically

    # standard ALC params
    ret.tireStiffnessFactor = 1.0
    ret.steerControlType = structs.CarParams.SteerControlType.torque
    ret.minSteerSpeed = 0.
    ret.wheelSpeedFactor = 1.0

    ret.pcmCruise = True     # openpilot's state is tied to the PCM's cruise state on most cars
    ret.minEnableSpeed = -1. # enable is done by stock ACC, so ignore this
    ret.steerRatioRear = 0.  # no rear steering, at least on the listed cars aboveA
    ret.openpilotLongitudinalControl = False
    ret.stopAccel = -2.0
    ret.stoppingDecelRate = 0.8 # brake_travel/s while trying to stop
    ret.vEgoStopping = 0.5
    ret.vEgoStarting = 0.5
    ret.longitudinalTuning.kf = 1.
    ret.longitudinalTuning.kpBP = [0.]
    ret.longitudinalTuning.kpV = [0.]
    ret.longitudinalTuning.kiBP = [0.]
    ret.longitudinalTuning.kiV = [0.]
    # TODO estimate car specific lag, use .15s for now
    ret.longitudinalActuatorDelay = 0.15
    ret.steerLimitTimer = 1.0
    return ret

  @staticmethod
  def configure_torque_tune(candidate: str, tune: structs.CarParams.LateralTuning, steering_angle_deadzone_deg: float = 0.0, use_steering_angle: bool = True):
    params = get_torque_params()[candidate]

    tune.init('torque')
    tune.torque.useSteeringAngle = use_steering_angle
    tune.torque.kp = 1.0
    tune.torque.kf = 1.0
    tune.torque.ki = 0.1
    tune.torque.friction = params['FRICTION']
    tune.torque.latAccelFactor = params['LAT_ACCEL_FACTOR']
    tune.torque.latAccelOffset = 0.0
    tune.torque.steeringAngleDeadzoneDeg = steering_angle_deadzone_deg

  def _update(self) -> structs.CarState:
    return self.CS.update(self.can_parsers)

  def update(self, can_packets: list[tuple[int, list[CanData]]]) -> structs.CarState:
    # parse can
    for cp in self.can_parsers.values():
      if cp is not None:
        cp.update_strings(can_packets)

    # get CarState
    ret = self._update()

    ret.canValid = all(cp.can_valid for cp in self.can_parsers.values())
    ret.canTimeout = any(cp.bus_timeout for cp in self.can_parsers.values())

    if ret.vEgoCluster == 0.0 and not self.v_ego_cluster_seen:
      ret.vEgoCluster = ret.vEgo
    else:
      self.v_ego_cluster_seen = True

    # Many cars apply hysteresis to the ego dash speed
    ret.vEgoCluster = apply_hysteresis(ret.vEgoCluster, self.CS.out.vEgoCluster, self.CS.cluster_speed_hyst_gap)
    if abs(ret.vEgo) < self.CS.cluster_min_speed:
      ret.vEgoCluster = 0.0

    if ret.cruiseState.speedCluster == 0:
      ret.cruiseState.speedCluster = ret.cruiseState.speed

    # save for next iteration
    self.CS.out = ret

    return ret


class RadarInterfaceBase(ABC):
  def __init__(self, CP: structs.CarParams):
    self.CP = CP
    self.rcp = None
    self.pts: dict[int, structs.RadarData.RadarPoint] = {}
    self.frame = 0

  def update(self, can_packets: list[tuple[int, list[CanData]]]) -> structs.RadarDataT | None:
    self.frame += 1
    if (self.frame % 5) == 0:  # 20 Hz is very standard
      return structs.RadarData()
    return None


class CarStateBase(ABC):
  def __init__(self, CP: structs.CarParams):
    self.CP = CP
    self.car_fingerprint = CP.carFingerprint
    self.out = structs.CarState()

    self.cruise_buttons = 0
    self.left_blinker_cnt = 0
    self.right_blinker_cnt = 0
    self.steering_pressed_cnt = 0
    self.left_blinker_prev = False
    self.right_blinker_prev = False
    self.cluster_speed_hyst_gap = 0.0
    self.cluster_min_speed = 0.0  # min speed before dropping to 0
    self.secoc_key: bytes = b"00" * 16

    Q = [[0.0, 0.0], [0.0, 100.0]]
    R = 0.3
    A = [[1.0, DT_CTRL], [0.0, 1.0]]
    C = [[1.0, 0.0]]
    x0=[[0.0], [0.0]]
    K = get_kalman_gain(DT_CTRL, np.array(A), np.array(C), np.array(Q), R)
    self.v_ego_kf = KF1D(x0=x0, A=A, C=C[0], K=K)

  @abstractmethod
  def update(self, can_parsers) -> structs.CarState:
    pass

  def update_speed_kf(self, v_ego_raw):
    if abs(v_ego_raw - self.v_ego_kf.x[0][0]) > 2.0:  # Prevent large accelerations when car starts at non zero speed
      self.v_ego_kf.set_x([[v_ego_raw], [0.0]])

    v_ego_x = self.v_ego_kf.update(v_ego_raw)
    return float(v_ego_x[0]), float(v_ego_x[1])

  def get_wheel_speeds(self, fl, fr, rl, rr, unit=CV.KPH_TO_MS):
    factor = unit * self.CP.wheelSpeedFactor

    wheelSpeeds = structs.CarState.WheelSpeeds()
    wheelSpeeds.fl = fl * factor
    wheelSpeeds.fr = fr * factor
    wheelSpeeds.rl = rl * factor
    wheelSpeeds.rr = rr * factor
    return wheelSpeeds

  def update_blinker_from_lamp(self, blinker_time: int, left_blinker_lamp: bool, right_blinker_lamp: bool):
    """Update blinkers from lights. Enable output when light was seen within the last `blinker_time`
    iterations"""
    # TODO: Handle case when switching direction. Now both blinkers can be on at the same time
    self.left_blinker_cnt = blinker_time if left_blinker_lamp else max(self.left_blinker_cnt - 1, 0)
    self.right_blinker_cnt = blinker_time if right_blinker_lamp else max(self.right_blinker_cnt - 1, 0)
    return self.left_blinker_cnt > 0, self.right_blinker_cnt > 0

  def update_steering_pressed(self, steering_pressed, steering_pressed_min_count):
    """Applies filtering on steering pressed for noisy driver torque signals."""
    self.steering_pressed_cnt += 1 if steering_pressed else -1
    self.steering_pressed_cnt = clip(self.steering_pressed_cnt, 0, steering_pressed_min_count * 2)
    return self.steering_pressed_cnt > steering_pressed_min_count

  def update_blinker_from_stalk(self, blinker_time: int, left_blinker_stalk: bool, right_blinker_stalk: bool):
    """Update blinkers from stalk position. When stalk is seen the blinker will be on for at least blinker_time,
    or until the stalk is turned off, whichever is longer. If the opposite stalk direction is seen the blinker
    is forced to the other side. On a rising edge of the stalk the timeout is reset."""

    if left_blinker_stalk:
      self.right_blinker_cnt = 0
      if not self.left_blinker_prev:
        self.left_blinker_cnt = blinker_time

    if right_blinker_stalk:
      self.left_blinker_cnt = 0
      if not self.right_blinker_prev:
        self.right_blinker_cnt = blinker_time

    self.left_blinker_cnt = max(self.left_blinker_cnt - 1, 0)
    self.right_blinker_cnt = max(self.right_blinker_cnt - 1, 0)

    self.left_blinker_prev = left_blinker_stalk
    self.right_blinker_prev = right_blinker_stalk

    return bool(left_blinker_stalk or self.left_blinker_cnt > 0), bool(right_blinker_stalk or self.right_blinker_cnt > 0)

  @staticmethod
  def parse_gear_shifter(gear: str | None) -> structs.CarState.GearShifter:
    if gear is None:
      return GearShifter.unknown
    return GEAR_SHIFTER_MAP.get(gear.upper(), GearShifter.unknown)

  @staticmethod
  def get_can_parsers(CP) -> dict[StrEnum, CANParser]:
    return {}


class CarControllerBase(ABC):
  def __init__(self, dbc_names: dict[StrEnum, str], CP: structs.CarParams):
    self.CP = CP
    self.frame = 0
    self.secoc_key: bytes = b"00" * 16

  @abstractmethod
  def update(self, CC: structs.CarControl, CS: CarStateBase, now_nanos: int) -> tuple[structs.CarControl.Actuators, list[CanData]]:
    pass


INTERFACE_ATTR_FILE = {
  "FINGERPRINTS": "fingerprints",
  "FW_VERSIONS": "fingerprints",
}

# interface-specific helpers

def get_interface_attr(attr: str, combine_brands: bool = False, ignore_none: bool = False) -> dict[str | StrEnum, Any]:
  # read all the folders in opendbc/car and return a dict where:
  # - keys are all the car models or brand names
  # - values are attr values from all car folders
  result = {}
  for car_folder in sorted([x[0] for x in os.walk(BASEDIR)]):
    try:
      brand_name = car_folder.split('/')[-1]
      brand_values = __import__(f'opendbc.car.{brand_name}.{INTERFACE_ATTR_FILE.get(attr, "values")}', fromlist=[attr])
      if hasattr(brand_values, attr) or not ignore_none:
        attr_data = getattr(brand_values, attr, None)
      else:
        continue

      if combine_brands:
        if isinstance(attr_data, dict):
          for f, v in attr_data.items():
            result[f] = v
      else:
        result[brand_name] = attr_data
    except (ImportError, OSError):
      pass

  return result


class NanoFFModel:
  def __init__(self, weights_loc: str, platform: str):
    self.weights_loc = weights_loc
    self.platform = platform
    self.load_weights(platform)

  def load_weights(self, platform: str):
    with open(self.weights_loc) as fob:
      self.weights = {k: np.array(v) for k, v in json.load(fob)[platform].items()}

  def relu(self, x: np.ndarray):
    return np.maximum(0.0, x)

  def forward(self, x: np.ndarray):
    assert x.ndim == 1
    x = (x - self.weights['input_norm_mat'][:, 0]) / (self.weights['input_norm_mat'][:, 1] - self.weights['input_norm_mat'][:, 0])
    x = self.relu(np.dot(x, self.weights['w_1']) + self.weights['b_1'])
    x = self.relu(np.dot(x, self.weights['w_2']) + self.weights['b_2'])
    x = self.relu(np.dot(x, self.weights['w_3']) + self.weights['b_3'])
    x = np.dot(x, self.weights['w_4']) + self.weights['b_4']
    return x

  def predict(self, x: list[float], do_sample: bool = False):
    x = self.forward(np.array(x))
    if do_sample:
      pred = np.random.laplace(x[0], np.exp(x[1]) / self.weights['temperature'])
    else:
      pred = x[0]
    pred = pred * (self.weights['output_norm_mat'][1] - self.weights['output_norm_mat'][0]) + self.weights['output_norm_mat'][0]
    return pred
