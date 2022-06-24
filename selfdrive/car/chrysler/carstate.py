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
    ret.vEgoRaw = cp.vl["ESP_8"]["Vehicle_Speed"] * CV.KPH_TO_MS
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
    ret.standstill = not ret.vEgoRaw > 0.001
    ret.wheelSpeeds = self.get_wheel_speeds(
      cp.vl["ESP_6"]["WHEEL_SPEED_FL"],
      cp.vl["ESP_6"]["WHEEL_SPEED_RR"],
      cp.vl["ESP_6"]["WHEEL_SPEED_RL"],
      cp.vl["ESP_6"]["WHEEL_SPEED_FR"],
      unit=1,
    )
    #ret.aEgo = cp.vl["ESP_4"]["Acceleration"] #m/s2
    #ret.yawRate = cp.vl["ESP_4"]["Yaw_Rate"] #deg/s

    # button presses
    ret.leftBlinker = cp.vl["STEERING_LEVERS"]["Turn_Signal_Status"] == 1
    ret.rightBlinker = cp.vl["STEERING_LEVERS"]["Turn_Signal_Status"] == 2
    ret.genericToggle = bool(cp.vl["STEERING_LEVERS"]["High_Beam_Lever_Status"])

    # steering wheel
    ret.steeringAngleDeg = cp.vl["Steering_Column_Angle_Status"]["Steering_Wheel_Angle"]
    ret.steeringRateDeg = cp.vl["Steering_Column_Angle_Status"]["Steering_Rate"]
    ret.steeringTorque = cp.vl["EPS_2"]["COLUMN_TORQUE"]
    ret.steeringTorqueEps = cp.vl["EPS_2"]["EPS_MOTOR_TORQUE"]

    ret.steeringPressed = abs(ret.steeringTorque) > STEER_THRESHOLD
    #ret.espDisabled = (cp.vl["Center_Stack_1"]["Traction_Button"] == 1)  # button is pressed. This doesn't mean ESP is disabled.
    self.frame = int(cp.vl["EPS_2"]["COUNTER"])
    #steer_state = cp.vl["EPS_2"]["LKAS_STATE"]
    #ret.steerFaultPermanent = steer_state == 4 or (steer_state == 0 and ret.vEgo > self.CP.minSteerSpeed)

    # gear
    ret.gearShifter = self.parse_gear_shifter(self.shifter_values.get(cp.vl["Transmission_Status"]["Gear_State"], None))

    # cruise state
    cp_cruise = cp_cam if self.CP.carFingerprint in RAM_CARS else cp

    ret.cruiseState.available = cp_cruise.vl["DAS_3"]["ACC_AVAILABLE"] == 1  # ACC is white
    ret.cruiseState.enabled = cp_cruise.vl["DAS_3"]["ACC_ACTIVE"] == 1  # ACC is green
    ret.cruiseState.speed = cp_cruise.vl["DAS_4"]["ACC_SPEED_CONFIG_KPH"] * CV.KPH_TO_MS
    ret.cruiseState.nonAdaptive = cp_cruise.vl["DAS_4"]["CRUISE_STATE"] in (1, 2)

    ret.cruiseState.speed = cp_cruise.vl["DAS_4"]["ACC_Set_Speed"] * CV.KPH_TO_MS
    ret.cruiseState.nonAdaptive = cp_cruise.vl["DAS_4"]["ACC_Activation_Status"] in (1, 2)  # 1 NormalCCOn and 2 NormalCCSet
    ret.cruiseState.standstill = cp_cruise.vl["DAS_3"]["ACC_StandStill"] == 1
    ret.accFaulted = cp.vl["DAS_3"]["ACC_FAULTED"] != 0

    if self.CP.carFingerprint in RAM_CARS:
      self.autoHighBeamBit = cp_cam.vl["DAS_6"]['Auto_High_Beam']  # Auto High Beam isn't Located in this message on chrysler or jeep currently located in 729 message
      ret.steerFaultTemporary = cp.vl["EPS_3"]["DASM_FAULT"] == 1
    else:
      steer_state = cp.vl["EPS_2"]["Torque_Overlay_Status"]
      ret.steerFaultPermanent = steer_state == 4 or (steer_state == 0 and ret.vEgo > self.CP.minSteerSpeed)

    # blindspot sensors
    if self.CP.enableBsm:
      ret.leftBlindspot = cp.vl["BSM_1"]["Blind_Spot_Monitor_Left"] == 1
      ret.rightBlindspot = cp.vl["BSM_1"]["Blind_Spot_Monitor_Right"] == 1

    self.lkas_counter = cp_cam.vl["DAS_3"]["COUNTER"]
    self.lkas_car_model = cp_cam.vl["DAS_6"]["CAR_MODEL"]
    self.button_counter = cp.vl["Cruise_Control_Buttons"]["COUNTER"]

    return ret

  @staticmethod
  def get_can_parser(CP):
    signals = [
      # sig_name, sig_address
      ("PRNDL", "GEAR"),
      ("DOOR_OPEN_FL", "BCM_1"),
      ("DOOR_OPEN_FR", "BCM_1"),
      ("DOOR_OPEN_RL", "BCM_1"),
      ("DOOR_OPEN_RR", "BCM_1"),
      ("Brake_Pedal_State", "ESP_1"),
      ("Accelerator_Position", "ECM_5"),
      ("SPEED_LEFT", "SPEED_1"),
      ("SPEED_RIGHT", "SPEED_1"),
      ("WHEEL_SPEED_FL", "WHEEL_SPEEDS"),
      ("WHEEL_SPEED_RR", "WHEEL_SPEEDS"),
      ("WHEEL_SPEED_RL", "WHEEL_SPEEDS"),
      ("WHEEL_SPEED_FR", "WHEEL_SPEEDS"),
      ("STEER_ANGLE", "STEERING"),
      ("STEERING_RATE", "STEERING"),
      ("TURN_SIGNALS", "STEERING_LEVERS"),
      ("ACC_AVAILABLE", "DAS_3"),
      ("ACC_ACTIVE", "DAS_3"),
      ("ACC_FAULTED", "DAS_3"),
      ("HIGH_BEAM_FLASH", "STEERING_LEVERS"),
      ("ACC_SPEED_CONFIG_KPH", "DAS_4"),
      ("CRUISE_STATE", "DAS_4"),
      ("TORQUE_DRIVER", "EPS_STATUS"),
      ("TORQUE_MOTOR", "EPS_STATUS"),
      ("LKAS_STATE", "EPS_STATUS"),
      ("COUNTER", "EPS_STATUS",),
      ("TRACTION_OFF", "TRACTION_BUTTON"),
      ("SEATBELT_DRIVER_UNLATCHED", "SEATBELT_STATUS"),
      ("COUNTER", "WHEEL_BUTTONS"),
    ]

    checks = [
      # sig_address, frequency
      ("ESP_1", 50),
      ("EPS_STATUS", 100),
      ("SPEED_1", 100),
      ("WHEEL_SPEEDS", 50),
      ("STEERING", 100),
      ("DAS_3", 50),
      ("GEAR", 50),
      ("ECM_5", 50),
      ("WHEEL_BUTTONS", 50),
      ("DAS_4", 15),
      ("STEERING_LEVERS", 10),
      ("SEATBELT_STATUS", 2),
      ("BCM_1", 1),
      ("TRACTION_BUTTON", 1),
    ]

    if CP.enableBsm:
      signals += [
        ("Blind_Spot_Monitor_Left", "BSM_1"),
        ("Blind_Spot_Monitor_Right", "BSM_1"),
      ]
      checks += [("BSM_1", 2)]

    if CP.carFingerprint in RAM_CARS:
      signals += [
        #("LKAS_Button", "Center_Stack_2"),  # LKAS Button
        ("DASM_FAULT", "EPS_3"),  # EPS Fault
      ]
      checks += [
        #("Center_Stack_2", 1),
        ("EPS_3", 50),
      ]
    else:
      signals += [
        ("ACC_Engaged", "DAS_3"),  # ACC Engaged
        ("ACC_StandStill", "DAS_3"),  # ACC Engaged
        ("ACC_Set_Speed", "DAS_4"),
        ("ACC_Activation_Status", "DAS_4"),
      ]
      checks += [
        ("DAS_3", 50),
        ("DAS_4", 50),
      ]

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
        ("ACC_Engaged", "DAS_3"),  # ACC Engaged
        ("ACC_StandStill", "DAS_3"),  # ACC Engaged
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
