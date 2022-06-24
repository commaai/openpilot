from cereal import car
from common.conversions import Conversions as CV
from opendbc.can.parser import CANParser
from opendbc.can.can_define import CANDefine
from selfdrive.car.interfaces import CarStateBase
from selfdrive.car.chrysler.values import DBC, STEER_THRESHOLD, RAM_CARS


class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)
    can_define = CANDefine(DBC[CP.carFingerprint]["pt"])
    self.shifter_values = can_define.dv["Transmission_Status"]["Gear_State"]

  def update(self, cp, cp_cam):

    ret = car.CarState.new_message()

    # lock info
    ret.doorOpen = any([cp.vl["BCM_1"]["DOOR_OPEN_FL"],
                        cp.vl["BCM_1"]["DOOR_OPEN_FR"],
                        cp.vl["BCM_1"]["DOOR_OPEN_RL"],
                        cp.vl["BCM_1"]["DOOR_OPEN_RR"]])
    ret.seatbeltUnlatched = cp.vl["ORC_1"]['SEATBELT_DRIVER_UNLATCHED'] == 1

    # brake pedal
    ret.brake = 0
    ret.brakePressed = cp.vl["ESP_1"]['Brake_Pedal_State'] == 1  # Physical brake pedal switch

    # gas pedal
    ret.gas = cp.vl["ECM_5"]["Accelerator_Position"]
    ret.gasPressed = ret.gas > 1e-5

    # car speed
    ret.vEgoRaw = cp.vl["ESP_8"]["VEHICLE_SPEED"] * CV.KPH_TO_MS
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
    ret.standstill = not ret.vEgoRaw > 0.001
    ret.wheelSpeeds = self.get_wheel_speeds(
      cp.vl["ESP_6"]["WHEEL_SPEED_FL"],
      cp.vl["ESP_6"]["WHEEL_SPEED_FR"],
      cp.vl["ESP_6"]["WHEEL_SPEED_RL"],
      cp.vl["ESP_6"]["WHEEL_SPEED_RR"],
      unit=1,
    )
    #ret.aEgo = cp.vl["ESP_4"]["Acceleration"] #m/s2
    #ret.yawRate = cp.vl["ESP_4"]["Yaw_Rate"] #deg/s

    # button presses
    ret.leftBlinker = cp.vl["STEERING_LEVERS"]["TURN_SIGNALS"] == 1
    ret.rightBlinker = cp.vl["STEERING_LEVERS"]["TURN_SIGNALS"] == 2
    ret.genericToggle = cp.vl["STEERING_LEVERS"]["HIGH_BEAM_FLASH"] == 1

    # steering wheel
    ret.steeringAngleDeg = cp.vl["STEERING"]["STEER_ANGLE"]
    ret.steeringRateDeg = cp.vl["STEERING"]["STEERING_RATE"]
    ret.steeringTorque = cp.vl["EPS_2"]["COLUMN_TORQUE"]
    ret.steeringTorqueEps = cp.vl["EPS_2"]["EPS_MOTOR_TORQUE"]

    ret.steeringPressed = abs(ret.steeringTorque) > STEER_THRESHOLD
    self.frame = int(cp.vl["EPS_2"]["COUNTER"])
    #steer_state = cp.vl["EPS_2"]["LKAS_STATE"]
    #ret.steerFaultPermanent = steer_state == 4 or (steer_state == 0 and ret.vEgo > self.CP.minSteerSpeed)

    # gear
    ret.gearShifter = self.parse_gear_shifter(self.shifter_values.get(cp.vl["Transmission_Status"]["Gear_State"], None))

    # cruise state
    cp_cruise = cp_cam if self.CP.carFingerprint in RAM_CARS else cp

    ret.cruiseState.available = cp_cruise.vl["DAS_3"]["ACC_AVAILABLE"] == 1  # ACC is white
    ret.cruiseState.enabled = cp_cruise.vl["DAS_3"]["ACC_ACTIVE"] == 1  # ACC is green
    ret.cruiseState.speed = cp_cruise.vl["DAS_4"]["ACC_Set_Speed"] * CV.KPH_TO_MS
    ret.cruiseState.nonAdaptive = cp_cruise.vl["DAS_4"]["ACC_Activation_Status"] in (1, 2)  # 1 NormalCCOn and 2 NormalCCSet
    ret.cruiseState.standstill = cp_cruise.vl["DAS_3"]["ACC_STANDSTILL"] == 1
    ret.accFaulted = cp_cruise.vl["DAS_3"]["ACC_FAULTED"] != 0

    if self.CP.carFingerprint in RAM_CARS:
      self.autoHighBeamBit = cp_cam.vl["DAS_6"]['Auto_High_Beam']  # Auto High Beam isn't Located in this message on chrysler or jeep currently located in 729 message
    #else:
    #  steer_state = cp.vl["EPS_2"]["Torque_Overlay_Status"]
    #  ret.steerFaultPermanent = steer_state == 4 or (steer_state == 0 and ret.vEgo > self.CP.minSteerSpeed)

    # blindspot sensors
    if self.CP.enableBsm:
      ret.leftBlindspot = cp.vl["BSM_1"]["BSM_LEFT_STATUS"] == 1
      ret.rightBlindspot = cp.vl["BSM_1"]["BSM_RIGHT_STATUS"] == 1

    self.lkas_counter = cp_cam.vl["DAS_3"]["COUNTER"]
    self.lkas_car_model = cp_cam.vl["DAS_6"]["CAR_MODEL"]
    self.button_counter = cp.vl["CRUISE_BUTTONS"]["COUNTER"]

    return ret

  @staticmethod
  def get_can_parser(CP):
    signals = [
      # sig_name, sig_address
      ("Gear_State", "Transmission_Status"),
      ("DOOR_OPEN_FL", "BCM_1"),
      ("DOOR_OPEN_FR", "BCM_1"),
      ("DOOR_OPEN_RL", "BCM_1"),
      ("DOOR_OPEN_RR", "BCM_1"),
      ("Brake_Pedal_State", "ESP_1"),
      ("Accelerator_Position", "ECM_5"),
      ("WHEEL_SPEED_FL", "ESP_6"),
      ("WHEEL_SPEED_RR", "ESP_6"),
      ("WHEEL_SPEED_RL", "ESP_6"),
      ("WHEEL_SPEED_FR", "ESP_6"),
      ("STEER_ANGLE", "STEERING"),
      ("STEERING_RATE", "STEERING"),
      ("TURN_SIGNALS", "STEERING_LEVERS"),
      ("HIGH_BEAM_FLASH", "STEERING_LEVERS"),
      ("COLUMN_TORQUE", "EPS_2"),
      ("EPS_MOTOR_TORQUE", "EPS_2"),
      ("LKAS_STATE", "EPS_2"),
      ("COUNTER", "EPS_2",),
      #("TRACTION_OFF", "TRACTION_BUTTON"),
      ("SEATBELT_DRIVER_UNLATCHED", "ORC_1"),
      ("COUNTER", "CRUISE_BUTTONS"),
      ("VEHICLE_SPEED", "ESP_8"),
    ]

    checks = [
      # sig_address, frequency
      ("ESP_1", 50),
      ("EPS_2", 100),
      ("ESP_6", 50),
      ("ESP_8", 50),
      ("STEERING", 100),
      ("Transmission_Status", 50),
      ("ECM_5", 50),
      ("CRUISE_BUTTONS", 50),
      ("STEERING_LEVERS", 10),
      ("ORC_1", 2),
      ("BCM_1", 1),
      #("TRACTION_BUTTON", 1),
    ]

    if CP.enableBsm:
      signals += [
        ("BSM_RIGHT_STATUS", "BSM_1"),
        ("BSM_LEFT_STATUS", "BSM_1"),
      ]
      checks += [("BSM_1", 2)]

    return CANParser(DBC[CP.carFingerprint]["pt"], signals, checks, 0)

  @staticmethod
  def get_cam_can_parser(CP):
    signals = [
      # sig_name, sig_address, default
      ("CAR_MODEL", "DAS_6"),
    ]
    checks = [
      ("DAS_6", 15),
    ]

    if CP.carFingerprint in RAM_CARS:
      signals += [
        ("ACC_AVAILABLE", "DAS_3"),
        ("ACC_ACTIVE", "DAS_3"),
        ("ACC_FAULTED", "DAS_3"),
        ("ACC_STANDSTILL", "DAS_3"),
        ("COUNTER", "DAS_3"),
        ("ACC_Set_Speed", "DAS_4"),
        ("ACC_Activation_Status", "DAS_4"),
        ("Auto_High_Beam", "DAS_6"),
      ]
      checks += [
        ("DAS_3", 50),
        ("DAS_4", 50),
      ]

    return CANParser(DBC[CP.carFingerprint]["pt"], signals, checks, 2)
