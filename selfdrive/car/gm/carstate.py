import copy
from cereal import car
from opendbc.can.can_define import CANDefine
from opendbc.can.parser import CANParser
from openpilot.selfdrive.car.conversions import Conversions as CV
from openpilot.selfdrive.car.helpers import mean
from openpilot.selfdrive.car.interfaces import CarStateBase
from openpilot.selfdrive.car.gm.values import DBC, AccState, CanBus, STEER_THRESHOLD

TransmissionType = car.CarParams.TransmissionType
NetworkLocation = car.CarParams.NetworkLocation
STANDSTILL_THRESHOLD = 10 * 0.0311 * CV.KPH_TO_MS


class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)
    can_define = CANDefine(DBC[CP.carFingerprint]["pt"])
    self.shifter_values = can_define.dv["ECMPRDNL2"]["PRNDL2"]
    self.cluster_speed_hyst_gap = CV.KPH_TO_MS / 2.
    self.cluster_min_speed = CV.KPH_TO_MS / 2.

    self.loopback_lka_steering_cmd_updated = False
    self.loopback_lka_steering_cmd_ts_nanos = 0
    self.pt_lka_steering_cmd_counter = 0
    self.cam_lka_steering_cmd_counter = 0
    self.buttons_counter = 0

    self.prev_distance_button = 0
    self.distance_button = 0

  def update(self, cp, cp_cam, _, __, loopback_cp):
    ret = car.CarState.new_message()

    self.prev_cruise_buttons = self.cruise_buttons
    self.prev_distance_button = self.distance_button
    self.cruise_buttons = cp.vl["ASCMSteeringButton"]["ACCButtons"]
    self.distance_button = cp.vl["ASCMSteeringButton"]["DistanceButton"]
    self.buttons_counter = cp.vl["ASCMSteeringButton"]["RollingCounter"]
    self.pscm_status = copy.copy(cp.vl["PSCMStatus"])
    # This is to avoid a fault where you engage while still moving backwards after shifting to D.
    # An Equinox has been seen with an unsupported status (3), so only check if either wheel is in reverse (2)
    self.moving_backward = (cp.vl["EBCMWheelSpdRear"]["RLWheelDir"] == 2) or (cp.vl["EBCMWheelSpdRear"]["RRWheelDir"] == 2)

    # Variables used for avoiding LKAS faults
    self.loopback_lka_steering_cmd_updated = len(loopback_cp.vl_all["ASCMLKASteeringCmd"]["RollingCounter"]) > 0
    if self.loopback_lka_steering_cmd_updated:
      self.loopback_lka_steering_cmd_ts_nanos = loopback_cp.ts_nanos["ASCMLKASteeringCmd"]["RollingCounter"]
    if self.CP.networkLocation == NetworkLocation.fwdCamera:
      self.pt_lka_steering_cmd_counter = cp.vl["ASCMLKASteeringCmd"]["RollingCounter"]
      self.cam_lka_steering_cmd_counter = cp_cam.vl["ASCMLKASteeringCmd"]["RollingCounter"]

    ret.wheelSpeeds = self.get_wheel_speeds(
      cp.vl["EBCMWheelSpdFront"]["FLWheelSpd"],
      cp.vl["EBCMWheelSpdFront"]["FRWheelSpd"],
      cp.vl["EBCMWheelSpdRear"]["RLWheelSpd"],
      cp.vl["EBCMWheelSpdRear"]["RRWheelSpd"],
    )
    ret.vEgoRaw = mean([ret.wheelSpeeds.fl, ret.wheelSpeeds.fr, ret.wheelSpeeds.rl, ret.wheelSpeeds.rr])
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
    # sample rear wheel speeds, standstill=True if ECM allows engagement with brake
    ret.standstill = ret.wheelSpeeds.rl <= STANDSTILL_THRESHOLD and ret.wheelSpeeds.rr <= STANDSTILL_THRESHOLD

    if cp.vl["ECMPRDNL2"]["ManualMode"] == 1:
      ret.gearShifter = self.parse_gear_shifter("T")
    else:
      ret.gearShifter = self.parse_gear_shifter(self.shifter_values.get(cp.vl["ECMPRDNL2"]["PRNDL2"], None))

    ret.brake = cp.vl["ECMAcceleratorPos"]["BrakePedalPos"]
    if self.CP.networkLocation == NetworkLocation.fwdCamera:
      ret.brakePressed = cp.vl["ECMEngineStatus"]["BrakePressed"] != 0
    else:
      # Some Volt 2016-17 have loose brake pedal push rod retainers which causes the ECM to believe
      # that the brake is being intermittently pressed without user interaction.
      # To avoid a cruise fault we need to use a conservative brake position threshold
      # https://static.nhtsa.gov/odi/tsbs/2017/MC-10137629-9999.pdf
      ret.brakePressed = ret.brake >= 8

    # Regen braking is braking
    if self.CP.transmissionType == TransmissionType.direct:
      ret.regenBraking = cp.vl["EBCMRegenPaddle"]["RegenPaddle"] != 0

    ret.gas = cp.vl["AcceleratorPedal2"]["AcceleratorPedal2"] / 254.
    ret.gasPressed = ret.gas > 1e-5

    ret.steeringAngleDeg = cp.vl["PSCMSteeringAngle"]["SteeringWheelAngle"]
    ret.steeringRateDeg = cp.vl["PSCMSteeringAngle"]["SteeringWheelRate"]
    ret.steeringTorque = cp.vl["PSCMStatus"]["LKADriverAppldTrq"]
    ret.steeringTorqueEps = cp.vl["PSCMStatus"]["LKATorqueDelivered"]
    ret.steeringPressed = abs(ret.steeringTorque) > STEER_THRESHOLD

    # 0 inactive, 1 active, 2 temporarily limited, 3 failed
    self.lkas_status = cp.vl["PSCMStatus"]["LKATorqueDeliveredStatus"]
    ret.steerFaultTemporary = self.lkas_status == 2
    ret.steerFaultPermanent = self.lkas_status == 3

    # 1 - open, 0 - closed
    ret.doorOpen = (cp.vl["BCMDoorBeltStatus"]["FrontLeftDoor"] == 1 or
                    cp.vl["BCMDoorBeltStatus"]["FrontRightDoor"] == 1 or
                    cp.vl["BCMDoorBeltStatus"]["RearLeftDoor"] == 1 or
                    cp.vl["BCMDoorBeltStatus"]["RearRightDoor"] == 1)

    # 1 - latched
    ret.seatbeltUnlatched = cp.vl["BCMDoorBeltStatus"]["LeftSeatBelt"] == 0
    ret.leftBlinker = cp.vl["BCMTurnSignals"]["TurnSignals"] == 1
    ret.rightBlinker = cp.vl["BCMTurnSignals"]["TurnSignals"] == 2

    ret.parkingBrake = cp.vl["BCMGeneralPlatformStatus"]["ParkBrakeSwActive"] == 1
    ret.cruiseState.available = cp.vl["ECMEngineStatus"]["CruiseMainOn"] != 0
    ret.espDisabled = cp.vl["ESPStatus"]["TractionControlOn"] != 1
    ret.accFaulted = (cp.vl["AcceleratorPedal2"]["CruiseState"] == AccState.FAULTED or
                      cp.vl["EBCMFrictionBrakeStatus"]["FrictionBrakeUnavailable"] == 1)

    ret.cruiseState.enabled = cp.vl["AcceleratorPedal2"]["CruiseState"] != AccState.OFF
    ret.cruiseState.standstill = cp.vl["AcceleratorPedal2"]["CruiseState"] == AccState.STANDSTILL
    if self.CP.networkLocation == NetworkLocation.fwdCamera:
      ret.cruiseState.speed = cp_cam.vl["ASCMActiveCruiseControlStatus"]["ACCSpeedSetpoint"] * CV.KPH_TO_MS
      ret.stockAeb = cp_cam.vl["AEBCmd"]["AEBCmdActive"] != 0
      # openpilot controls nonAdaptive when not pcmCruise
      if self.CP.pcmCruise:
        ret.cruiseState.nonAdaptive = cp_cam.vl["ASCMActiveCruiseControlStatus"]["ACCCruiseState"] not in (2, 3)

    if self.CP.enableBsm:
      ret.leftBlindspot = cp.vl["BCMBlindSpotMonitor"]["LeftBSM"] == 1
      ret.rightBlindspot = cp.vl["BCMBlindSpotMonitor"]["RightBSM"] == 1

    return ret

  @staticmethod
  def get_cam_can_parser(CP):
    messages = []
    if CP.networkLocation == NetworkLocation.fwdCamera:
      messages += [
        ("AEBCmd", 10),
        ("ASCMLKASteeringCmd", 10),
        ("ASCMActiveCruiseControlStatus", 25),
      ]

    return CANParser(DBC[CP.carFingerprint]["pt"], messages, CanBus.CAMERA)

  @staticmethod
  def get_can_parser(CP):
    messages = [
      ("BCMTurnSignals", 1),
      ("ECMPRDNL2", 10),
      ("PSCMStatus", 10),
      ("ESPStatus", 10),
      ("BCMDoorBeltStatus", 10),
      ("BCMGeneralPlatformStatus", 10),
      ("EBCMWheelSpdFront", 20),
      ("EBCMWheelSpdRear", 20),
      ("EBCMFrictionBrakeStatus", 20),
      ("AcceleratorPedal2", 33),
      ("ASCMSteeringButton", 33),
      ("ECMEngineStatus", 100),
      ("PSCMSteeringAngle", 100),
      ("ECMAcceleratorPos", 80),
    ]

    if CP.enableBsm:
      messages.append(("BCMBlindSpotMonitor", 10))

    # Used to read back last counter sent to PT by camera
    if CP.networkLocation == NetworkLocation.fwdCamera:
      messages += [
        ("ASCMLKASteeringCmd", 0),
      ]

    if CP.transmissionType == TransmissionType.direct:
      messages.append(("EBCMRegenPaddle", 50))

    return CANParser(DBC[CP.carFingerprint]["pt"], messages, CanBus.POWERTRAIN)

  @staticmethod
  def get_loopback_can_parser(CP):
    messages = [
      ("ASCMLKASteeringCmd", 0),
    ]

    return CANParser(DBC[CP.carFingerprint]["pt"], messages, CanBus.LOOPBACK)
