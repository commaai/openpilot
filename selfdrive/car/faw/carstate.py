import numpy as np
from cereal import car
from selfdrive.car.interfaces import CarStateBase
from opendbc.can.parser import CANParser
from opendbc.can.can_define import CANDefine
from selfdrive.car.faw.values import DBC_FILES, CANBUS, GearShifter, CarControllerParams

class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)
    can_define = CANDefine(DBC_FILES.faw)
    # TODO: populate this
    # self.shifter_values = can_define.dv["Getriebe_11"]["GE_Fahrstufe"]
    self.lkas_status_values = can_define.dv["EPS_2"]["LKAS_STATUS"]

  def update(self, pt_cp, cam_cp):
    ret = car.CarState.new_message()
    # Update vehicle speed and acceleration from ABS wheel speeds.
    ret.wheelSpeeds = self.get_wheel_speeds(
      pt_cp.vl["ABS_1"]["FRONT_LEFT"],
      pt_cp.vl["ABS_1"]["FRONT_RIGHT"],
      pt_cp.vl["ABS_2"]["REAR_LEFT"],
      pt_cp.vl["ABS_2"]["REAR_RIGHT"],
    )

    ret.vEgoRaw = float(np.mean([ret.wheelSpeeds.fl, ret.wheelSpeeds.fr, ret.wheelSpeeds.rl, ret.wheelSpeeds.rr]))
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
    ret.standstill = ret.vEgo < 0.1

    # Update steering angle, rate, yaw rate, and driver input torque. FAW send
    # the sign/direction in a separate signal so they must be recombined.
    ret.steeringAngleDeg = pt_cp.vl["EPS_1"]["STEER_ANGLE"] * (1, -1)[int(pt_cp.vl["EPS_1"]["STEER_ANGLE_DIRECTION"])]
    ret.steeringRateDeg = pt_cp.vl["EPS_1"]["STEER_RATE"] * (1, -1)[int(pt_cp.vl["EPS_1"]["STEER_RATE_DIRECTION"])]
    # FIXME: don't have a solid direction bit for this yet, borrow from overall EPS output
    ret.steeringTorque = pt_cp.vl["EPS_2"]["DRIVER_INPUT_TORQUE"] * (1, -1)[int(pt_cp.vl["EPS_2"]["EPS_TORQUE_DIRECTION"])]
    ret.steeringPressed = abs(ret.steeringTorque) > CarControllerParams.STEER_DRIVER_ALLOWANCE
    # TODO: populate this
    # ret.yawRate = pt_cp.vl["ESP_02"]["ESP_Gierrate"] * (1, -1)[int(pt_cp.vl["ESP_02"]["ESP_VZ_Gierrate"])] * CV.DEG_TO_RAD

    # Verify EPS readiness to accept steering commands
    lkas_status = self.lkas_status_values.get(pt_cp.vl["EPS_2"]["LKAS_STATUS"])
    ret.steerFaultPermanent = lkas_status == "FAULT"
    ret.steerFaultTemporary = lkas_status == "INITIALIZING"

    # Update gas, brakes, and gearshift.
    ret.gas = pt_cp.vl["ECM_1"]["DRIVER_THROTTLE"]
    ret.gasPressed = ret.gas > 0
    ret.brake = pt_cp.vl["ABS_2"]["BRAKE_PRESSURE"]
    ret.brakePressed = ret.brake > 0  # TODO: should be fine, but check for a nice boolean anyway
    # TODO: populate this
    # ret.parkingBrake = bool(pt_cp.vl["Kombi_01"]["KBI_Handbremse"])  # FIXME: need to include an EPB check as well

    # Update gear and/or clutch position data.
    # TODO: populate this
    # ret.gearShifter = self.parse_gear_shifter(self.shifter_values.get(pt_cp.vl["Getriebe_11"]["GE_Fahrstufe"], None))
    ret.gearShifter = GearShifter.drive

    # Update door and trunk/hatch lid open status.
    # TODO: populate this
    # ret.doorOpen = any([pt_cp.vl["Gateway_72"]["ZV_FT_offen"],
    #                     pt_cp.vl["Gateway_72"]["ZV_BT_offen"],
    #                     pt_cp.vl["Gateway_72"]["ZV_HFS_offen"],
    #                     pt_cp.vl["Gateway_72"]["ZV_HBFS_offen"],
    #                     pt_cp.vl["Gateway_72"]["ZV_HD_offen"]])

    # Update seatbelt fastened status.
    # TODO: populate this
    # ret.seatbeltUnlatched = pt_cp.vl["Airbag_02"]["AB_Gurtschloss_FA"] != 3

    # Consume blind-spot monitoring info/warning LED states, if available.
    # Infostufe: BSM LED on, Warnung: BSM LED flashing
    # TODO: populate this
    # if self.CP.enableBsm:
    #   ret.leftBlindspot = bool(ext_cp.vl["SWA_01"]["SWA_Infostufe_SWA_li"]) or bool(ext_cp.vl["SWA_01"]["SWA_Warnung_SWA_li"])
    #   ret.rightBlindspot = bool(ext_cp.vl["SWA_01"]["SWA_Infostufe_SWA_re"]) or bool(ext_cp.vl["SWA_01"]["SWA_Warnung_SWA_re"])

    # TODO: populate if possible
    # ret.stockFcw = bool(ext_cp.vl["ACC_10"]["AWV2_Freigabe"])
    # ret.stockAeb = bool(ext_cp.vl["ACC_10"]["ANB_Teilbremsung_Freigabe"]) or bool(ext_cp.vl["ACC_10"]["ANB_Zielbremsung_Freigabe"])

    # Update ACC radar status.
    # TODO: populate this properly, need an available signal and overrides (11 avail?, 27 gas override?, 19 coastdown?)
    ret.cruiseState.available = True
    ret.cruiseState.enabled = cam_cp.vl["ACC"]["STATUS"] == 20

    # Update ACC setpoint.
    # TODO: populate this
    #    ret.cruiseState.speed = 0

    # Update control button states for turn signals and ACC controls.
    # TODO: populate this
    # CC button states
    # ret.leftBlinker = bool(pt_cp.vl["Blinkmodi_02"]["Comfort_Signal_Left"])
    # ret.rightBlinker = bool(pt_cp.vl["Blinkmodi_02"]["Comfort_Signal_Right"])

    # TODO: populate this
    # ret.espDisabled = pt_cp.vl["ESP_21"]["ESP_Tastung_passiv"] != 0

    return ret

  @staticmethod
  def get_can_parser(CP):
    signals = [
      # sig_name, sig_address
      ("DRIVER_THROTTLE", "ECM_1"),
      ("DRIVER_THROTTLE", "ECM_1"),
      ("FRONT_LEFT", "ABS_1"),
      ("FRONT_RIGHT", "ABS_1"),
      ("REAR_LEFT", "ABS_2"),
      ("REAR_RIGHT", "ABS_2"),
      ("BRAKE_PRESSURE", "ABS_2"),
      ("STEER_ANGLE", "EPS_1"),
      ("STEER_ANGLE_DIRECTION", "EPS_1"),
      ("STEER_RATE", "EPS_1"),
      ("STEER_RATE_DIRECTION", "EPS_1"),
      ("EPS_TORQUE", "EPS_2"),
      ("EPS_TORQUE_DIRECTION", "EPS_2"),
      ("DRIVER_INPUT_TORQUE", "EPS_2"),
      ("LKAS_STATUS", "EPS_2"),
    ]

    checks = [
      # sig_address, frequency
      ("ECM_1", 100),
      ("ABS_1", 100),
      ("ABS_2", 100),
      ("EPS_1", 50),
      ("EPS_2", 50),
    ]

    return CANParser(DBC_FILES.faw, signals, checks, CANBUS.pt)

  @staticmethod
  def get_cam_can_parser(CP):
    signals = [
      # sig_name, sig_address
      ("STATUS", "ACC")
    ]
    checks = [
      # sig_address, frequency
      ("ACC", 50),
    ]

    return CANParser(DBC_FILES.faw, signals, checks, CANBUS.cam)
