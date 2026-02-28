# Opel Corsa F (PSA CMP Platform) - Car State
# Reads and parses CAN bus signals for vehicle state
import numpy as np
from cereal import car
from common.conversions import Conversions as CV
from selfdrive.car.interfaces import CarStateBase
from opendbc.can.parser import CANParser
from opendbc.can.can_define import CANDefine
from selfdrive.car.opel.values import DBC_FILES, CANBUS, NetworkLocation, TransmissionType, GearShifter, BUTTON_STATES, CarControllerParams


class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)
    can_define = CANDefine(DBC_FILES.psa_cmp)
    if CP.transmissionType == TransmissionType.automatic:
      self.shifter_values = can_define.dv["PSA_Transmission"]["GearPosition"]
    self.buttonStates = BUTTON_STATES.copy()

  def update(self, pt_cp, cam_cp, ext_cp, trans_type):
    ret = car.CarState.new_message()

    # Update vehicle speed from ABS wheel speeds
    # PSA CMP platform uses different signal names than VW MQB
    ret.wheelSpeeds = self.get_wheel_speeds(
      pt_cp.vl["PSA_ABS"]["WheelSpeed_FL"],
      pt_cp.vl["PSA_ABS"]["WheelSpeed_FR"],
      pt_cp.vl["PSA_ABS"]["WheelSpeed_RL"],
      pt_cp.vl["PSA_ABS"]["WheelSpeed_RR"],
    )

    ret.vEgoRaw = float(np.mean([ret.wheelSpeeds.fl, ret.wheelSpeeds.fr, ret.wheelSpeeds.rl, ret.wheelSpeeds.rr]))
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
    ret.standstill = ret.vEgo < 0.1

    # Update steering angle, rate, and driver input torque
    ret.steeringAngleDeg = pt_cp.vl["PSA_Steering"]["SteeringAngle"]
    ret.steeringRateDeg = pt_cp.vl["PSA_Steering"]["SteeringRate"]
    ret.steeringTorque = pt_cp.vl["PSA_EPS"]["DriverTorque"]
    ret.steeringPressed = abs(ret.steeringTorque) > CarControllerParams.STEER_DRIVER_ALLOWANCE
    ret.yawRate = pt_cp.vl["PSA_ESP"]["YawRate"] * CV.DEG_TO_RAD

    # Verify EPS readiness to accept steering commands
    eps_status = pt_cp.vl["PSA_EPS"]["EPS_Status"]
    ret.steerFaultPermanent = eps_status in (0, 4)   # Disabled or Fault
    ret.steerFaultTemporary = eps_status in (1, 3)    # Initializing or Rejected

    # Update gas, brakes, and gearshift
    ret.gas = pt_cp.vl["PSA_Accelerator"]["AcceleratorPedal"] / 100.0
    ret.gasPressed = ret.gas > 0
    ret.brake = pt_cp.vl["PSA_Brake"]["BrakePressure"] / 250.0
    ret.brakePressed = bool(pt_cp.vl["PSA_Brake"]["BrakePressed"])
    ret.parkingBrake = bool(pt_cp.vl["PSA_ESP_Status"]["ParkingBrake"])

    # Update gear and/or clutch position data
    if trans_type == TransmissionType.automatic:
      ret.gearShifter = self.parse_gear_shifter(
        self.shifter_values.get(pt_cp.vl["PSA_Transmission"]["GearPosition"], None)
      )
    elif trans_type == TransmissionType.manual:
      if bool(pt_cp.vl["PSA_Lights"]["ReverseLight"]):
        ret.gearShifter = GearShifter.reverse
      else:
        ret.gearShifter = GearShifter.drive

    # Update door and trunk/hatch lid open status
    ret.doorOpen = any([
      pt_cp.vl["PSA_Doors"]["DoorOpen_FL"],
      pt_cp.vl["PSA_Doors"]["DoorOpen_FR"],
      pt_cp.vl["PSA_Doors"]["DoorOpen_RL"],
      pt_cp.vl["PSA_Doors"]["DoorOpen_RR"],
      pt_cp.vl["PSA_Doors"]["TrunkOpen"],
    ])

    # Update seatbelt fastened status
    ret.seatbeltUnlatched = not bool(pt_cp.vl["PSA_Airbag"]["SeatbeltFastened_Driver"])

    # Update driver preference for metric
    self.displayMetricUnits = bool(pt_cp.vl["PSA_Cluster"]["MetricUnits"])

    # Consume blind-spot monitoring info if available
    if self.CP.enableBsm:
      ret.leftBlindspot = bool(ext_cp.vl["PSA_BSM"]["BSM_Left"])
      ret.rightBlindspot = bool(ext_cp.vl["PSA_BSM"]["BSM_Right"])

    # Stock FCW and AEB status
    ret.stockFcw = bool(ext_cp.vl["PSA_ACC"]["FCW_Active"])
    ret.stockAeb = bool(ext_cp.vl["PSA_ACC"]["AEB_Active"])

    # Update ACC status
    acc_status = pt_cp.vl["PSA_ACC_Status"]["ACC_Status"]
    if acc_status == 2:
      # ACC enabled but not engaged
      ret.cruiseState.available = True
      ret.cruiseState.enabled = False
    elif acc_status in (3, 4, 5):
      # ACC engaged and regulating
      ret.cruiseState.available = True
      ret.cruiseState.enabled = True
    else:
      # ACC disabled or fault
      ret.cruiseState.available = False
      ret.cruiseState.enabled = False

    # Update ACC setpoint
    if self.CP.pcmCruise:
      ret.cruiseState.speed = ext_cp.vl["PSA_ACC"]["ACC_SetSpeed"] * CV.KPH_TO_MS
      if ret.cruiseState.speed > 90:
        ret.cruiseState.speed = 0

    # Update control button states
    self.buttonStates["accelCruise"] = bool(pt_cp.vl["PSA_ACC_Buttons"]["AccelButton"])
    self.buttonStates["decelCruise"] = bool(pt_cp.vl["PSA_ACC_Buttons"]["DecelButton"])
    self.buttonStates["cancel"] = bool(pt_cp.vl["PSA_ACC_Buttons"]["CancelButton"])
    self.buttonStates["setCruise"] = bool(pt_cp.vl["PSA_ACC_Buttons"]["SetButton"])
    self.buttonStates["resumeCruise"] = bool(pt_cp.vl["PSA_ACC_Buttons"]["ResumeButton"])
    self.buttonStates["gapAdjustCruise"] = bool(pt_cp.vl["PSA_ACC_Buttons"]["GapAdjustButton"])
    ret.leftBlinker = bool(pt_cp.vl["PSA_Lights"]["LeftBlinker"])
    ret.rightBlinker = bool(pt_cp.vl["PSA_Lights"]["RightBlinker"])

    # ACC button info for passthrough
    self.accMainSwitch = pt_cp.vl["PSA_ACC_Buttons"]["MainSwitch"]
    self.accMsgBusCounter = pt_cp.vl["PSA_ACC_Buttons"]["COUNTER"]

    # ESP disabled check
    ret.espDisabled = pt_cp.vl["PSA_ESP_Status"]["ESP_Disabled"] != 0

    return ret

  @staticmethod
  def get_can_parser(CP):
    signals = [
      # sig_name, sig_address
      ("SteeringAngle", "PSA_Steering"),             # Absolute steering angle
      ("SteeringRate", "PSA_Steering"),               # Steering rate
      ("WheelSpeed_FL", "PSA_ABS"),                  # ABS wheel speed, front left
      ("WheelSpeed_FR", "PSA_ABS"),                  # ABS wheel speed, front right
      ("WheelSpeed_RL", "PSA_ABS"),                  # ABS wheel speed, rear left
      ("WheelSpeed_RR", "PSA_ABS"),                  # ABS wheel speed, rear right
      ("YawRate", "PSA_ESP"),                         # Yaw rate
      ("DoorOpen_FL", "PSA_Doors"),                   # Door open, driver
      ("DoorOpen_FR", "PSA_Doors"),                   # Door open, passenger
      ("DoorOpen_RL", "PSA_Doors"),                   # Door open, rear left
      ("DoorOpen_RR", "PSA_Doors"),                   # Door open, rear right
      ("TrunkOpen", "PSA_Doors"),                     # Trunk or hatch open
      ("LeftBlinker", "PSA_Lights"),                  # Left turn signal
      ("RightBlinker", "PSA_Lights"),                 # Right turn signal
      ("SeatbeltFastened_Driver", "PSA_Airbag"),      # Seatbelt status, driver
      ("BrakePressed", "PSA_Brake"),                  # Brake pedal pressed
      ("BrakePressure", "PSA_Brake"),                 # Brake pressure applied
      ("AcceleratorPedal", "PSA_Accelerator"),        # Accelerator pedal value
      ("DriverTorque", "PSA_EPS"),                    # Driver torque input
      ("EPS_Status", "PSA_EPS"),                      # EPS control status
      ("ESP_Disabled", "PSA_ESP_Status"),             # ESP disabled
      ("ParkingBrake", "PSA_ESP_Status"),             # Parking brake applied
      ("MetricUnits", "PSA_Cluster"),                 # KMH vs MPH
      ("ACC_Status", "PSA_ACC_Status"),               # ACC engagement status
      ("MainSwitch", "PSA_ACC_Buttons"),              # ACC main switch
      ("CancelButton", "PSA_ACC_Buttons"),            # ACC cancel
      ("SetButton", "PSA_ACC_Buttons"),               # ACC set
      ("AccelButton", "PSA_ACC_Buttons"),             # ACC accel
      ("DecelButton", "PSA_ACC_Buttons"),             # ACC decel
      ("ResumeButton", "PSA_ACC_Buttons"),            # ACC resume
      ("GapAdjustButton", "PSA_ACC_Buttons"),         # ACC gap adjust
      ("COUNTER", "PSA_ACC_Buttons"),                 # Message counter
    ]

    checks = [
      # sig_address, frequency
      ("PSA_Steering", 100),       # Steering angle sensor
      ("PSA_EPS", 100),            # EPS controller
      ("PSA_ABS", 100),            # ABS/ESP controller
      ("PSA_Brake", 50),           # Brake controller
      ("PSA_ESP_Status", 50),      # ESP status
      ("PSA_Accelerator", 50),     # Engine control module
      ("PSA_ACC_Status", 50),      # ACC status
      ("PSA_ESP", 50),             # ESP yaw rate
      ("PSA_ACC_Buttons", 33),     # ACC buttons
      ("PSA_Doors", 10),           # Door status
      ("PSA_Airbag", 5),           # Airbag control module
      ("PSA_Cluster", 2),          # Instrument cluster
      ("PSA_Lights", 1),           # Turn signals / lights
    ]

    if CP.transmissionType == TransmissionType.automatic:
      signals.append(("GearPosition", "PSA_Transmission"))   # Auto transmission gear
      checks.append(("PSA_Transmission", 20))                # Transmission control module
    elif CP.transmissionType == TransmissionType.manual:
      signals.append(("ReverseLight", "PSA_Lights"))         # Reverse light for manual trans
      # PSA_Lights already in checks

    if CP.networkLocation == NetworkLocation.fwdCamera:
      signals += PsaExtraSignals.fwd_radar_signals
      checks += PsaExtraSignals.fwd_radar_checks
      if CP.enableBsm:
        signals += PsaExtraSignals.bsm_signals
        checks += PsaExtraSignals.bsm_checks

    return CANParser(DBC_FILES.psa_cmp, signals, checks, CANBUS.pt)

  @staticmethod
  def get_cam_can_parser(CP):
    signals = []
    checks = []

    if CP.networkLocation == NetworkLocation.fwdCamera:
      signals += [
        ("LDW_Warning_Left", "PSA_LDW"),       # Lane departure warning left
        ("LDW_Warning_Right", "PSA_LDW"),       # Lane departure warning right
        ("LDW_LaneDeparture", "PSA_LDW"),       # Lane departure direction
      ]
      checks += [
        ("PSA_LDW", 10),     # Lane departure warning camera
      ]
    else:
      signals += PsaExtraSignals.fwd_radar_signals
      checks += PsaExtraSignals.fwd_radar_checks
      if CP.enableBsm:
        signals += PsaExtraSignals.bsm_signals
        checks += PsaExtraSignals.bsm_checks

    return CANParser(DBC_FILES.psa_cmp, signals, checks, CANBUS.cam)


class PsaExtraSignals:
  # Additional signals for optional controllers
  fwd_radar_signals = [
    ("ACC_SetSpeed", "PSA_ACC"),                  # ACC set speed
    ("FCW_Active", "PSA_ACC"),                    # Forward collision warning
    ("AEB_Active", "PSA_ACC"),                    # Autonomous emergency braking
  ]
  fwd_radar_checks = [
    ("PSA_ACC", 50),                              # ACC radar control module
  ]
  bsm_signals = [
    ("BSM_Left", "PSA_BSM"),                      # Blind spot left
    ("BSM_Right", "PSA_BSM"),                     # Blind spot right
  ]
  bsm_checks = [
    ("PSA_BSM", 20),                              # Blind spot monitoring
  ]
