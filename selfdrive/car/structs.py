from dataclasses import dataclass as _dataclass, field, is_dataclass
from enum import Enum, StrEnum as _StrEnum, auto
from typing import dataclass_transform, get_origin

AUTO_OBJ = object()


def auto_field():
  return AUTO_OBJ


@dataclass_transform()
def auto_dataclass(cls=None, /, **kwargs):
  cls_annotations = cls.__dict__.get('__annotations__', {})
  for name, typ in cls_annotations.items():
    current_value = getattr(cls, name)
    if current_value is AUTO_OBJ:
      origin_typ = get_origin(typ) or typ
      if isinstance(origin_typ, str):
        raise TypeError(f"Forward references are not supported for auto_field: '{origin_typ}'. Use a default_factory with lambda instead.")
      elif origin_typ in (int, float, str, bytes, list, tuple, bool) or is_dataclass(origin_typ):
        setattr(cls, name, field(default_factory=origin_typ))
      elif issubclass(origin_typ, Enum):  # first enum is the default
        setattr(cls, name, field(default=next(iter(origin_typ))))
      else:
        raise TypeError(f"Unsupported type for auto_field: {origin_typ}")

  # TODO: use slots, this prevents accidentally setting attributes that don't exist
  return _dataclass(cls, **kwargs)


class StrEnum(_StrEnum):
  @staticmethod
  def _generate_next_value_(name, *args):
    # auto() defaults to name.lower()
    return name


@auto_dataclass
class CarState:
  # CAN health
  canValid: bool = auto_field()  # invalid counter/checksums
  canTimeout: bool = auto_field()  # CAN bus dropped out
  canErrorCounter: int = auto_field()

  # car speed
  vEgo: float = auto_field()  # best estimate of speed
  aEgo: float = auto_field()  # best estimate of acceleration
  vEgoRaw: float = auto_field()  # unfiltered speed from CAN sensors
  vEgoCluster: float = auto_field()  # best estimate of speed shown on car's instrument cluster, used for UI

  yawRate: float = auto_field()  # best estimate of yaw rate
  standstill: bool = auto_field()
  wheelSpeeds: 'CarState.WheelSpeeds' = field(default_factory=lambda: CarState.WheelSpeeds())

  # gas pedal, 0.0-1.0
  gas: float = auto_field()  # this is user pedal only
  gasPressed: bool = auto_field()  # this is user pedal only

  engineRpm: float = auto_field()

  # brake pedal, 0.0-1.0
  brake: float = auto_field()  # this is user pedal only
  brakePressed: bool = auto_field()  # this is user pedal only
  regenBraking: bool = auto_field()  # this is user pedal only
  parkingBrake: bool = auto_field()
  brakeHoldActive: bool = auto_field()

  # steering wheel
  steeringAngleDeg: float = auto_field()
  steeringAngleOffsetDeg: float = auto_field()  # Offset betweens sensors in case there multiple
  steeringRateDeg: float = auto_field()
  steeringTorque: float = auto_field()  # TODO: standardize units
  steeringTorqueEps: float = auto_field()  # TODO: standardize units
  steeringPressed: bool = auto_field()  # if the user is using the steering wheel
  steerFaultTemporary: bool = auto_field()  # temporary EPS fault
  steerFaultPermanent: bool = auto_field()  # permanent EPS fault
  stockAeb: bool = auto_field()
  stockFcw: bool = auto_field()
  espDisabled: bool = auto_field()
  accFaulted: bool = auto_field()
  carFaultedNonCritical: bool = auto_field()  # some ECU is faulted, but car remains controllable
  espActive: bool = auto_field()
  vehicleSensorsInvalid: bool = auto_field()  # invalid steering angle readings, etc.

  # cruise state
  cruiseState: 'CarState.CruiseState' = field(default_factory=lambda: CarState.CruiseState())

  # gear
  gearShifter: 'CarState.GearShifter' = field(default_factory=lambda: CarState.GearShifter.unknown)

  # button presses
  buttonEvents: list['CarState.ButtonEvent'] = auto_field()
  leftBlinker: bool = auto_field()
  rightBlinker: bool = auto_field()
  genericToggle: bool = auto_field()

  # lock info
  doorOpen: bool = auto_field()
  seatbeltUnlatched: bool = auto_field()

  # clutch (manual transmission only)
  clutchPressed: bool = auto_field()

  # blindspot sensors
  leftBlindspot: bool = auto_field()  # Is there something blocking the left lane change
  rightBlindspot: bool = auto_field()  # Is there something blocking the right lane change

  fuelGauge: float = auto_field()  # battery or fuel tank level from 0.0 to 1.0
  charging: bool = auto_field()

  # process meta
  cumLagMs: float = auto_field()

  @auto_dataclass
  class WheelSpeeds:
    # optional wheel speeds
    fl: float = auto_field()
    fr: float = auto_field()
    rl: float = auto_field()
    rr: float = auto_field()

  @auto_dataclass
  class CruiseState:
    enabled: bool = auto_field()
    speed: float = auto_field()
    speedCluster: float = auto_field()  # Set speed as shown on instrument cluster
    available: bool = auto_field()
    speedOffset: float = auto_field()
    standstill: bool = auto_field()
    nonAdaptive: bool = auto_field()

  class GearShifter(StrEnum):
    unknown = auto()
    park = auto()
    drive = auto()
    neutral = auto()
    reverse = auto()
    sport = auto()
    low = auto()
    brake = auto()
    eco = auto()
    manumatic = auto()

  # send on change
  @auto_dataclass
  class ButtonEvent:
    pressed: bool = auto_field()
    type: 'CarState.ButtonEvent.Type' = field(default_factory=lambda: CarState.ButtonEvent.Type.unknown)

    class Type(StrEnum):
      unknown = auto()
      leftBlinker = auto()
      rightBlinker = auto()
      accelCruise = auto()
      decelCruise = auto()
      cancel = auto()
      altButton1 = auto()
      altButton2 = auto()
      altButton3 = auto()
      setCruise = auto()
      resumeCruise = auto()
      gapAdjustCruise = auto()


@auto_dataclass
class RadarData:
  errors: list['Error'] = auto_field()
  points: list['RadarPoint'] = auto_field()

  class Error(StrEnum):
    canError = auto()
    fault = auto()
    wrongConfig = auto()

  @auto_dataclass
  class RadarPoint:
    trackId: int = auto_field()  # no trackId reuse

    # these 3 are the minimum required
    dRel: float = auto_field()  # m from the front bumper of the car
    yRel: float = auto_field()  # m
    vRel: float = auto_field()  # m/s

    # these are optional and valid if they are not NaN
    aRel: float = auto_field()  # m/s^2
    yvRel: float = auto_field()  # m/s

    # some radars flag measurements VS estimates
    measured: bool = auto_field()


@auto_dataclass
class CarControl:
  # must be true for any actuator commands to work
  enabled: bool = auto_field()
  latActive: bool = auto_field()
  longActive: bool = auto_field()

  # Actuator commands as computed by controlsd
  actuators: 'CarControl.Actuators' = field(default_factory=lambda: CarControl.Actuators())

  leftBlinker: bool = auto_field()
  rightBlinker: bool = auto_field()

  orientationNED: list[float] = auto_field()
  angularVelocity: list[float] = auto_field()

  cruiseControl: 'CarControl.CruiseControl' = field(default_factory=lambda: CarControl.CruiseControl())
  hudControl: 'CarControl.HUDControl' = field(default_factory=lambda: CarControl.HUDControl())

  @auto_dataclass
  class Actuators:
    # range from 0.0 - 1.0
    gas: float = auto_field()
    brake: float = auto_field()
    # range from -1.0 - 1.0
    steer: float = auto_field()
    # value sent over can to the car
    steerOutputCan: float = auto_field()
    steeringAngleDeg: float = auto_field()

    curvature: float = auto_field()

    speed: float = auto_field()  # m/s
    accel: float = auto_field()  # m/s^2
    longControlState: 'CarControl.Actuators.LongControlState' = field(default_factory=lambda: CarControl.Actuators.LongControlState.off)

    class LongControlState(StrEnum):
      off = auto()
      pid = auto()
      stopping = auto()
      starting = auto()

  @auto_dataclass
  class CruiseControl:
    cancel: bool = auto_field()
    resume: bool = auto_field()
    override: bool = auto_field()

  @auto_dataclass
  class HUDControl:
    speedVisible: bool = auto_field()
    setSpeed: float = auto_field()
    lanesVisible: bool = auto_field()
    leadVisible: bool = auto_field()
    visualAlert: 'CarControl.HUDControl.VisualAlert' = field(default_factory=lambda: CarControl.HUDControl.VisualAlert.none)
    audibleAlert: 'CarControl.HUDControl.AudibleAlert' = field(default_factory=lambda: CarControl.HUDControl.AudibleAlert.none)
    rightLaneVisible: bool = auto_field()
    leftLaneVisible: bool = auto_field()
    rightLaneDepart: bool = auto_field()
    leftLaneDepart: bool = auto_field()
    leadDistanceBars: int = auto_field()  # 1-3: 1 is closest, 3 is farthest. some ports may utilize 2-4 bars instead

    class VisualAlert(StrEnum):
      # these are the choices from the Honda
      # map as good as you can for your car
      none = auto()
      fcw = auto()
      steerRequired = auto()
      brakePressed = auto()
      wrongGear = auto()
      seatbeltUnbuckled = auto()
      speedTooHigh = auto()
      ldw = auto()

    class AudibleAlert(StrEnum):
      none = auto()

      engage = auto()
      disengage = auto()
      refuse = auto()

      warningSoft = auto()
      warningImmediate = auto()

      prompt = auto()
      promptRepeat = auto()
      promptDistracted = auto()


@auto_dataclass
class CarParams:
  carName: str = auto_field()
  carFingerprint: str = auto_field()
  fuzzyFingerprint: bool = auto_field()

  notCar: bool = auto_field()  # flag for non-car robotics platforms

  pcmCruise: bool = auto_field()  # is openpilot's state tied to the PCM's cruise state?
  enableDsu: bool = auto_field()  # driving support unit
  enableBsm: bool = auto_field()  # blind spot monitoring
  flags: int = auto_field()  # flags for car specific quirks
  experimentalLongitudinalAvailable: bool = auto_field()

  minEnableSpeed: float = auto_field()
  minSteerSpeed: float = auto_field()
  safetyConfigs: list['CarParams.SafetyConfig'] = auto_field()
  alternativeExperience: int = auto_field()  # panda flag for features like no disengage on gas

  maxLateralAccel: float = auto_field()
  autoResumeSng: bool = auto_field()  # describes whether car can resume from a stop automatically

  mass: float = auto_field()  # [kg] curb weight: all fluids no cargo
  wheelbase: float = auto_field()  # [m] distance from rear axle to front axle
  centerToFront: float = auto_field()  # [m] distance from center of mass to front axle
  steerRatio: float = auto_field()  # [] ratio of steering wheel angle to front wheel angle
  steerRatioRear: float = auto_field()  # [] ratio of steering wheel angle to rear wheel angle (usually 0)

  rotationalInertia: float = auto_field()  # [kg*m2] body rotational inertia
  tireStiffnessFactor: float = auto_field()  # scaling factor used in calculating tireStiffness[Front,Rear]
  tireStiffnessFront: float = auto_field()  # [N/rad] front tire coeff of stiff
  tireStiffnessRear: float = auto_field()  # [N/rad] rear tire coeff of stiff

  longitudinalTuning: 'CarParams.LongitudinalPIDTuning' = field(default_factory=lambda: CarParams.LongitudinalPIDTuning())
  lateralParams: 'CarParams.LateralParams' = field(default_factory=lambda: CarParams.LateralParams())
  lateralTuning: 'CarParams.LateralTuning' = field(default_factory=lambda: CarParams.LateralTuning())

  @auto_dataclass
  class LateralTuning:
    def init(self, which: str):
      self.which = CarParams.LateralTuning.Which(which)

    class Which(StrEnum):
      pid = auto()
      torque = auto()

      def __call__(self):
        return self.value

    which: 'CarParams.LateralTuning.Which' = field(default_factory=lambda: CarParams.LateralTuning.Which.pid)

    pid: 'CarParams.LateralPIDTuning' = field(default_factory=lambda: CarParams.LateralPIDTuning())
    torque: 'CarParams.LateralTorqueTuning' = field(default_factory=lambda: CarParams.LateralTorqueTuning())

  @auto_dataclass
  class SafetyConfig:
    safetyModel: 'CarParams.SafetyModel' = field(default_factory=lambda: CarParams.SafetyModel.silent)
    safetyParam: int = auto_field()

  @auto_dataclass
  class LateralParams:
    torqueBP: list[int] = auto_field()
    torqueV: list[int] = auto_field()

  @auto_dataclass
  class LateralPIDTuning:
    kpBP: list[float] = auto_field()
    kpV: list[float] = auto_field()
    kiBP: list[float] = auto_field()
    kiV: list[float] = auto_field()
    kf: float = auto_field()

  @auto_dataclass
  class LateralTorqueTuning:
    useSteeringAngle: bool = auto_field()
    kp: float = auto_field()
    ki: float = auto_field()
    friction: float = auto_field()
    kf: float = auto_field()
    steeringAngleDeadzoneDeg: float = auto_field()
    latAccelFactor: float = auto_field()
    latAccelOffset: float = auto_field()

  steerLimitAlert: bool = auto_field()
  steerLimitTimer: float = auto_field()  # time before steerLimitAlert is issued

  vEgoStopping: float = auto_field()  # Speed at which the car goes into stopping state
  vEgoStarting: float = auto_field()  # Speed at which the car goes into starting state
  stoppingControl: bool = auto_field()  # Does the car allow full control even at lows speeds when stopping
  steerControlType: 'CarParams.SteerControlType' = field(default_factory=lambda: CarParams.SteerControlType.torque)
  radarUnavailable: bool = auto_field()  # True when radar objects aren't visible on CAN or aren't parsed out
  stopAccel: float = auto_field()  # Required acceleration to keep vehicle stationary
  stoppingDecelRate: float = auto_field()  # m/s^2/s while trying to stop
  startAccel: float = auto_field()  # Required acceleration to get car moving
  startingState: bool = auto_field()  # Does this car make use of special starting state

  steerActuatorDelay: float = auto_field()  # Steering wheel actuator delay in seconds
  longitudinalActuatorDelay: float = auto_field()  # Gas/Brake actuator delay in seconds
  openpilotLongitudinalControl: bool = auto_field()  # is openpilot doing the longitudinal control?
  carVin: str = auto_field()  # VIN number queried during fingerprinting
  dashcamOnly: bool = auto_field()
  passive: bool = auto_field()  # is openpilot in control?
  transmissionType: 'CarParams.TransmissionType' = field(default_factory=lambda: CarParams.TransmissionType.unknown)
  carFw: list['CarParams.CarFw'] = auto_field()

  radarTimeStep: float = 0.05  # time delta between radar updates, 20Hz is very standard
  fingerprintSource: 'CarParams.FingerprintSource' = field(default_factory=lambda: CarParams.FingerprintSource.can)
  # Where Panda/C2 is integrated into the car's CAN network
  networkLocation: 'CarParams.NetworkLocation' = field(default_factory=lambda: CarParams.NetworkLocation.fwdCamera)

  wheelSpeedFactor: float = auto_field()  # Multiplier on wheels speeds to computer actual speeds

  @auto_dataclass
  class LongitudinalPIDTuning:
    kpBP: list[float] = auto_field()
    kpV: list[float] = auto_field()
    kiBP: list[float] = auto_field()
    kiV: list[float] = auto_field()
    kf: float = auto_field()

  class SafetyModel(StrEnum):
    silent = auto()
    hondaNidec = auto()
    toyota = auto()
    elm327 = auto()
    gm = auto()
    hondaBoschGiraffe = auto()
    ford = auto()
    cadillac = auto()
    hyundai = auto()
    chrysler = auto()
    tesla = auto()
    subaru = auto()
    gmPassive = auto()
    mazda = auto()
    nissan = auto()
    volkswagen = auto()
    toyotaIpas = auto()
    allOutput = auto()
    gmAscm = auto()
    noOutput = auto()  # like silent but without silent CAN TXs
    hondaBosch = auto()
    volkswagenPq = auto()
    subaruPreglobal = auto()  # pre-Global platform
    hyundaiLegacy = auto()
    hyundaiCommunity = auto()
    volkswagenMlb = auto()
    hongqi = auto()
    body = auto()
    hyundaiCanfd = auto()
    volkswagenMqbEvo = auto()
    chryslerCusw = auto()
    psa = auto()

  class SteerControlType(StrEnum):
    torque = auto()
    angle = auto()

  class TransmissionType(StrEnum):
    unknown = auto()
    automatic = auto()  # Traditional auto, including DSG
    manual = auto()  # True "stick shift" only
    direct = auto()  # Electric vehicle or other direct drive
    cvt = auto()

  @auto_dataclass
  class CarFw:
    ecu: 'CarParams.Ecu' = field(default_factory=lambda: CarParams.Ecu.unknown)
    fwVersion: bytes = auto_field()
    address: int = auto_field()
    subAddress: int = auto_field()
    responseAddress: int = auto_field()
    request: list[bytes] = auto_field()
    brand: str = auto_field()
    bus: int = auto_field()
    logging: bool = auto_field()
    obdMultiplexing: bool = auto_field()

  class Ecu(StrEnum):
    eps = auto()
    abs = auto()
    fwdRadar = auto()
    fwdCamera = auto()
    engine = auto()
    unknown = auto()
    transmission = auto()  # Transmission Control Module
    hybrid = auto()  # hybrid control unit, e.g. Chrysler's HCP, Honda's IMA Control Unit, Toyota's hybrid control computer
    srs = auto()  # airbag
    gateway = auto()  # can gateway
    hud = auto()  # heads up display
    combinationMeter = auto()  # instrument cluster
    electricBrakeBooster = auto()
    shiftByWire = auto()
    adas = auto()
    cornerRadar = auto()
    hvac = auto()
    parkingAdas = auto()  # parking assist system ECU, e.g. Toyota's IPAS, Hyundai's RSPA, etc.
    epb = auto()  # electronic parking brake
    telematics = auto()
    body = auto()  # body control module

    # Toyota only
    dsu = auto()

    # Honda only
    vsa = auto()  # Vehicle Stability Assist
    programmedFuelInjection = auto()

    debug = auto()

  class FingerprintSource(StrEnum):
    can = auto()
    fw = auto()
    fixed = auto()

  class NetworkLocation(StrEnum):
    fwdCamera = auto()  # Standard/default integration at LKAS camera
    gateway = auto()    # Integration at vehicle's CAN gateway
