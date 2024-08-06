# TODO: maybe call comms.py?
# TODO: maybe put in car_helpers.py? need to remove fingerprinting from car_helpers.py
from collections import namedtuple
from dataclasses import dataclass
from typing import NamedTuple


# TODO namedtuple doesn't support dict -> tuple conversion, it's purely order based.
# dataclass seems to be the best for easy conversion in openpilot
@dataclass
class CarParams:
  carName: str
  carFingerprint: str
  fuzzyFingerprint: bool

  notCar: bool

  pcmCruise: bool
  enableDsu: bool
  enableBsm: bool
  flags: int
  experimentalLongitudinalAvailable: bool

  minEnableSpeed: float
  minSteerSpeed: float
  # safetyConfigs: list[SafetyConfig]
  alternativeExperience: int

  # Car docs fields
  maxLateralAccel: float
  autoResumeSng: bool

  # things about the car in the manual
  mass: float
  wheelbase: float
  centerToFront: float
  steerRatio: float
  steerRatioRear: float

  # things we can derive
  rotationalInertia: float
  tireStiffnessFactor: float
  tireStiffnessFront: float
  tireStiffnessRear: float

  # TODO: the rest
  # longitudinalTuning: LongitudinalPIDTuning
  # lateralParams: LateralParams
  #
  # pid: LateralPIDTuning
  # indiDEPRECATED: LateralINDITuning
  # lqrDEPRECATED: LateralLQRTuning
  # torque: LateralTorqueTuning
  # steerLimitAlert: bool
  # steerLimitTimer: float
  #
  # vEgoStopping: float
  # vEgoStarting: float
  # stoppingControl: bool
  # steerControlType: SteerControlType
  # radarUnavailable: bool
  # stopAccel: float
  # stoppingDecelRate: float
  # startAccel: float
  # startingState: bool
  #
  # steerActuatorDelay: float
  # longitudinalActuatorDelay: float
  # openpilotLongitudinalControl: bool
  # carVin: str
  # transmissionType: TransmissionType
  # networkLocation: NetworkLocation
  #
  # wheelSpeedFactor: float
  #
  #
  #
  # safetyModel: SafetyModel
  # safetyParam: UInt16
  # safetyParamDEPRECATED: Int16
  # safetyParam2DEPRECATED: int
  # kf: float
  # useSteeringAngle: bool
  # kp: float
  # ki: float
  # friction: float
  # kf: float
  # steeringAngleDeadzoneDeg: float
  # latAccelFactor: float
  # latAccelOffset: float
  # kf: float
  # outerLoopGainDEPRECATED: float
  # innerLoopGainDEPRECATED: float
  # timeConstantDEPRECATED: float
  # actuatorEffectivenessDEPRECATED: float
  # scale: float
  # ki: float
  # dcGain: float
  # ecu: Ecu
  # fwVersion: Data
  # address: int
  # subAddress: UInt8
  # responseAddress: int
  # brand: str
  # bus: UInt8
  # logging: bool
  # obdMultiplexing: bool
  # enableGasInterceptorDEPRECATED: bool
  # enableCameraDEPRECATED: bool
  # enableApgsDEPRECATED: bool
  # steerRateCostDEPRECATED: float
  # isPandaBlackDEPRECATED: bool
  # hasStockCameraDEPRECATED: bool
  # safetyParamDEPRECATED: Int16
  # safetyModelDEPRECATED: SafetyModel
  # minSpeedCanDEPRECATED: float
  # startingAccelRateDEPRECATED: float
  # directAccelControlDEPRECATED: bool
  # maxSteeringAngleDegDEPRECATED: float
  # longitudinalActuatorDelayLowerBoundDEPRECATEDDEPRECATED: float
