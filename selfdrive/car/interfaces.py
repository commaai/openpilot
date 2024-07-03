import json
import os
import numpy as np
import tomllib
from abc import abstractmethod, ABC
from enum import StrEnum
from typing import Any, NamedTuple
from collections.abc import Callable
from functools import cache

from cereal import car
from openpilot.common.basedir import BASEDIR
from openpilot.common.conversions import Conversions as CV
from openpilot.common.simple_kalman import KF1D, get_kalman_gain
from openpilot.common.numpy_fast import clip
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.car import apply_hysteresis, gen_empty_fingerprint, scale_rot_inertia, scale_tire_stiffness, STD_CARGO_KG
from openpilot.selfdrive.car.values import PLATFORMS
from openpilot.selfdrive.controls.lib.drive_helpers import V_CRUISE_MAX, get_friction
from openpilot.selfdrive.controls.lib.events import Events
from openpilot.selfdrive.controls.lib.vehicle_model import VehicleModel

ButtonType = car.CarState.ButtonEvent.Type
GearShifter = car.CarState.GearShifter
EventName = car.CarEvent.EventName

MAX_CTRL_SPEED = (V_CRUISE_MAX + 4) * CV.KPH_TO_MS
ACCEL_MAX = 2.0
ACCEL_MIN = -3.5
FRICTION_THRESHOLD = 0.3

TORQUE_PARAMS_PATH = os.path.join(BASEDIR, 'selfdrive/car/torque_data/params.toml')
TORQUE_OVERRIDE_PATH = os.path.join(BASEDIR, 'selfdrive/car/torque_data/override.toml')
TORQUE_SUBSTITUTE_PATH = os.path.join(BASEDIR, 'selfdrive/car/torque_data/substitute.toml')

GEAR_SHIFTER_MAP: dict[str, car.CarState.GearShifter] = {
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


TorqueFromLateralAccelCallbackType = Callable[[LatControlInputs, car.CarParams.LateralTorqueTuning, float, float, bool, bool], float]


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
  def __init__(self, CP, CarController, CarState):
    self.CP = CP
    self.VM = VehicleModel(CP)

    self.frame = 0
    self.steering_unpressed = 0
    self.low_speed_alert = False
    self.no_steer_warning = False
    self.silent_steer_warning = True
    self.v_ego_cluster_seen = False

    self.CS = CarState(CP)
    self.cp = self.CS.get_can_parser(CP)
    self.cp_cam = self.CS.get_cam_can_parser(CP)
    self.cp_adas = self.CS.get_adas_can_parser(CP)
    self.cp_body = self.CS.get_body_can_parser(CP)
    self.cp_loopback = self.CS.get_loopback_can_parser(CP)
    self.can_parsers = [self.cp, self.cp_cam, self.cp_adas, self.cp_body, self.cp_loopback]

    dbc_name = "" if self.cp is None else self.cp.dbc_name
    self.CC: CarControllerBase = CarController(dbc_name, CP, self.VM)

  def apply(self, c: car.CarControl, now_nanos: int) -> tuple[car.CarControl.Actuators, list[tuple[int, int, bytes, int]]]:
    return self.CC.update(c, self.CS, now_nanos)

  @staticmethod
  def get_pid_accel_limits(CP, current_speed, cruise_speed):
    return ACCEL_MIN, ACCEL_MAX

  @classmethod
  def get_non_essential_params(cls, candidate: str):
    """
    Parameters essential to controlling the car may be incomplete or wrong without FW versions or fingerprints.
    """
    return cls.get_params(candidate, gen_empty_fingerprint(), list(), False, False)

  @classmethod
  def get_params(cls, candidate: str, fingerprint: dict[int, dict[int, int]], car_fw: list[car.CarParams.CarFw], experimental_long: bool, docs: bool):
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
  def _get_params(ret: car.CarParams, candidate, fingerprint: dict[int, dict[int, int]],
                  car_fw: list[car.CarParams.CarFw], experimental_long: bool, docs: bool):
    raise NotImplementedError

  @staticmethod
  def init(CP, logcan, sendcan):
    pass

  @staticmethod
  def get_steer_feedforward_default(desired_angle, v_ego):
    # Proportional to realigning tire momentum: lateral acceleration.
    return desired_angle * (v_ego**2)

  def get_steer_feedforward_function(self):
    return self.get_steer_feedforward_default

  def torque_from_lateral_accel_linear(self, latcontrol_inputs: LatControlInputs, torque_params: car.CarParams.LateralTorqueTuning,
                                       lateral_accel_error: float, lateral_accel_deadzone: float, friction_compensation: bool, gravity_adjusted: bool) -> float:
    # The default is a linear relationship between torque and lateral acceleration (accounting for road roll and steering friction)
    friction = get_friction(lateral_accel_error, lateral_accel_deadzone, FRICTION_THRESHOLD, torque_params, friction_compensation)
    return (latcontrol_inputs.lateral_acceleration / float(torque_params.latAccelFactor)) + friction

  def torque_from_lateral_accel(self) -> TorqueFromLateralAccelCallbackType:
    return self.torque_from_lateral_accel_linear

  # returns a set of default params to avoid repetition in car specific params
  @staticmethod
  def get_std_params(candidate):
    ret = car.CarParams.new_message()
    ret.carFingerprint = candidate

    # Car docs fields
    ret.maxLateralAccel = get_torque_params()[candidate]['MAX_LAT_ACCEL_MEASURED']
    ret.autoResumeSng = True  # describes whether car can resume from a stop automatically

    # standard ALC params
    ret.tireStiffnessFactor = 1.0
    ret.steerControlType = car.CarParams.SteerControlType.torque
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
    ret.stoppingControl = True
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
  def configure_torque_tune(candidate, tune, steering_angle_deadzone_deg=0.0, use_steering_angle=True):
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

  @abstractmethod
  def _update(self, c: car.CarControl) -> car.CarState:
    pass

  def update(self, c: car.CarControl, can_strings: list[bytes]) -> car.CarState:
    # parse can
    for cp in self.can_parsers:
      if cp is not None:
        cp.update_strings(can_strings)

    # get CarState
    ret = self._update(c)

    ret.canValid = all(cp.can_valid for cp in self.can_parsers if cp is not None)
    ret.canTimeout = any(cp.bus_timeout for cp in self.can_parsers if cp is not None)

    if ret.vEgoCluster == 0.0 and not self.v_ego_cluster_seen:
      ret.vEgoCluster = ret.vEgo
    else:
      self.v_ego_cluster_seen = True

    # Many cars apply hysteresis to the ego dash speed
    if self.CS is not None:
      ret.vEgoCluster = apply_hysteresis(ret.vEgoCluster, self.CS.out.vEgoCluster, self.CS.cluster_speed_hyst_gap)
      if abs(ret.vEgo) < self.CS.cluster_min_speed:
        ret.vEgoCluster = 0.0

    if ret.cruiseState.speedCluster == 0:
      ret.cruiseState.speedCluster = ret.cruiseState.speed

    # copy back for next iteration
    if self.CS is not None:
      self.CS.out = ret.as_reader()

    return ret


  def create_common_events(self, cs_out, extra_gears=None, pcm_enable=True, allow_enable=True,
                           enable_buttons=(ButtonType.accelCruise, ButtonType.decelCruise)):
    events = Events()

    if cs_out.doorOpen:
      events.add(EventName.doorOpen)
    if cs_out.seatbeltUnlatched:
      events.add(EventName.seatbeltNotLatched)
    if cs_out.gearShifter != GearShifter.drive and (extra_gears is None or
       cs_out.gearShifter not in extra_gears):
      events.add(EventName.wrongGear)
    if cs_out.gearShifter == GearShifter.reverse:
      events.add(EventName.reverseGear)
    if not cs_out.cruiseState.available:
      events.add(EventName.wrongCarMode)
    if cs_out.espDisabled:
      events.add(EventName.espDisabled)
    if cs_out.espActive:
      events.add(EventName.espActive)
    if cs_out.stockFcw:
      events.add(EventName.stockFcw)
    if cs_out.stockAeb:
      events.add(EventName.stockAeb)
    if cs_out.vEgo > MAX_CTRL_SPEED:
      events.add(EventName.speedTooHigh)
    if cs_out.cruiseState.nonAdaptive:
      events.add(EventName.wrongCruiseMode)
    if cs_out.brakeHoldActive and self.CP.openpilotLongitudinalControl:
      events.add(EventName.brakeHold)
    if cs_out.parkingBrake:
      events.add(EventName.parkBrake)
    if cs_out.accFaulted:
      events.add(EventName.accFaulted)
    if cs_out.steeringPressed:
      events.add(EventName.steerOverride)
    if cs_out.brakePressed and cs_out.standstill:
      events.add(EventName.preEnableStandstill)
    if cs_out.gasPressed:
      events.add(EventName.gasPressedOverride)

    # Handle button presses
    for b in cs_out.buttonEvents:
      # Enable OP long on falling edge of enable buttons (defaults to accelCruise and decelCruise, overridable per-port)
      if not self.CP.pcmCruise and (b.type in enable_buttons and not b.pressed):
        events.add(EventName.buttonEnable)
      # Disable on rising and falling edge of cancel for both stock and OP long
      if b.type == ButtonType.cancel:
        events.add(EventName.buttonCancel)

    # Handle permanent and temporary steering faults
    self.steering_unpressed = 0 if cs_out.steeringPressed else self.steering_unpressed + 1
    if cs_out.steerFaultTemporary:
      if cs_out.steeringPressed and (not self.CS.out.steerFaultTemporary or self.no_steer_warning):
        self.no_steer_warning = True
      else:
        self.no_steer_warning = False

        # if the user overrode recently, show a less harsh alert
        if self.silent_steer_warning or cs_out.standstill or self.steering_unpressed < int(1.5 / DT_CTRL):
          self.silent_steer_warning = True
          events.add(EventName.steerTempUnavailableSilent)
        else:
          events.add(EventName.steerTempUnavailable)
    else:
      self.no_steer_warning = False
      self.silent_steer_warning = False
    if cs_out.steerFaultPermanent:
      events.add(EventName.steerUnavailable)

    # we engage when pcm is active (rising edge)
    # enabling can optionally be blocked by the car interface
    if pcm_enable:
      if cs_out.cruiseState.enabled and not self.CS.out.cruiseState.enabled and allow_enable:
        events.add(EventName.pcmEnable)
      elif not cs_out.cruiseState.enabled:
        events.add(EventName.pcmDisable)

    return events


class RadarInterfaceBase(ABC):
  def __init__(self, CP):
    self.rcp = None
    self.pts = {}
    self.delay = 0
    self.radar_ts = CP.radarTimeStep
    self.frame = 0

  def update(self, can_strings):
    self.frame += 1
    if (self.frame % int(100 * self.radar_ts)) == 0:
      return car.RadarData.new_message()
    return None


class CarStateBase(ABC):
  def __init__(self, CP):
    self.CP = CP
    self.car_fingerprint = CP.carFingerprint
    self.out = car.CarState.new_message()

    self.cruise_buttons = 0
    self.left_blinker_cnt = 0
    self.right_blinker_cnt = 0
    self.steering_pressed_cnt = 0
    self.left_blinker_prev = False
    self.right_blinker_prev = False
    self.cluster_speed_hyst_gap = 0.0
    self.cluster_min_speed = 0.0  # min speed before dropping to 0

    Q = [[0.0, 0.0], [0.0, 100.0]]
    R = 0.3
    A = [[1.0, DT_CTRL], [0.0, 1.0]]
    C = [[1.0, 0.0]]
    x0=[[0.0], [0.0]]
    K = get_kalman_gain(DT_CTRL, np.array(A), np.array(C), np.array(Q), R)
    self.v_ego_kf = KF1D(x0=x0, A=A, C=C[0], K=K)

  def update_speed_kf(self, v_ego_raw):
    if abs(v_ego_raw - self.v_ego_kf.x[0][0]) > 2.0:  # Prevent large accelerations when car starts at non zero speed
      self.v_ego_kf.set_x([[v_ego_raw], [0.0]])

    v_ego_x = self.v_ego_kf.update(v_ego_raw)
    return float(v_ego_x[0]), float(v_ego_x[1])

  def get_wheel_speeds(self, fl, fr, rl, rr, unit=CV.KPH_TO_MS):
    factor = unit * self.CP.wheelSpeedFactor

    wheelSpeeds = car.CarState.WheelSpeeds.new_message()
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
  def parse_gear_shifter(gear: str | None) -> car.CarState.GearShifter:
    if gear is None:
      return GearShifter.unknown
    return GEAR_SHIFTER_MAP.get(gear.upper(), GearShifter.unknown)

  @staticmethod
  def get_can_parser(CP):
    return None

  @staticmethod
  def get_cam_can_parser(CP):
    return None

  @staticmethod
  def get_adas_can_parser(CP):
    return None

  @staticmethod
  def get_body_can_parser(CP):
    return None

  @staticmethod
  def get_loopback_can_parser(CP):
    return None


SendCan = tuple[int, int, bytes, int]


class CarControllerBase(ABC):
  def __init__(self, dbc_name: str, CP, VM):
    pass

  @abstractmethod
  def update(self, CC: car.CarControl.Actuators, CS: car.CarState, now_nanos: int) -> tuple[car.CarControl.Actuators, list[SendCan]]:
    pass


INTERFACE_ATTR_FILE = {
  "FINGERPRINTS": "fingerprints",
  "FW_VERSIONS": "fingerprints",
}

# interface-specific helpers

def get_interface_attr(attr: str, combine_brands: bool = False, ignore_none: bool = False) -> dict[str | StrEnum, Any]:
  # read all the folders in selfdrive/car and return a dict where:
  # - keys are all the car models or brand names
  # - values are attr values from all car folders
  result = {}
  for car_folder in sorted([x[0] for x in os.walk(BASEDIR + '/selfdrive/car')]):
    try:
      brand_name = car_folder.split('/')[-1]
      brand_values = __import__(f'openpilot.selfdrive.car.{brand_name}.{INTERFACE_ATTR_FILE.get(attr, "values")}', fromlist=[attr])
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
