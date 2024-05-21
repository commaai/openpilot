import crcmod

from openpilot.common.conversions import Conversions as CV
from openpilot.selfdrive.car.tesla.values import CANBUS, CarControllerParams


class TeslaCAN:
  def __init__(self, packer, pt_packer):
    self.packer = packer
    self.pt_packer = pt_packer
    self.crc = crcmod.mkCrcFun(0x11d, initCrc=0x00, rev=False, xorOut=0xff)

  @staticmethod
  def checksum(msg_id, dat):
    # TODO: get message ID from name instead
    ret = (msg_id & 0xFF) + ((msg_id >> 8) & 0xFF)
    ret += sum(dat)
    return ret & 0xFF

  @staticmethod
  def right_stalk_crc(dat):
    right_stalk_val = [0x7C, 0xB6, 0xF0, 0x2F, 0x69, 0xA3, 0xDD, 0x1C, 0x56, 0x90, 0xCA, 0x09, 0x43, 0x7D, 0xB7, 0xF1]
    cntr = dat[0] & 0xF
    crc1_func = crcmod.mkCrcFun(0x12F, initCrc=0x00, xorOut=0xFF, rev=False)
    crc1 = crc1_func(dat) & 0xFF
    crc2_func = crcmod.mkCrcFun(0x12F, initCrc=crc1, xorOut=0xFF, rev=False)
    return crc2_func(bytes([right_stalk_val[cntr]])) & 0xFF

  def create_steering_control(self, angle, enabled, counter):
    values = {
      "DAS_steeringAngleRequest": -angle,
      "DAS_steeringHapticRequest": 0,
      "DAS_steeringControlType": 1 if enabled else 0,
      "DAS_steeringControlCounter": counter,
    }

    data = self.packer.make_can_msg("DAS_steeringControl", CANBUS.party, values)[2]
    values["DAS_steeringControlChecksum"] = self.checksum(0x488, data[:3])
    return self.packer.make_can_msg("DAS_steeringControl", CANBUS.party, values)

  def create_longitudinal_commands(self, acc_state, speed, min_accel, max_accel, cnt):
    values = {
      "DAS_setSpeed": speed * CV.MS_TO_KPH,
      "DAS_accState": acc_state,
      "DAS_aebEvent": 0,
      "DAS_jerkMin": CarControllerParams.JERK_LIMIT_MIN,
      "DAS_jerkMax": CarControllerParams.JERK_LIMIT_MAX,
      "DAS_accelMin": min_accel,
      "DAS_accelMax": max_accel,
      "DAS_controlCounter": cnt,
      "DAS_controlChecksum": 0,
    }
    data = self.packer.make_can_msg("DAS_control", CANBUS.party, values)[2]
    values["DAS_controlChecksum"] = self.checksum(0x2b9, data[:7])
    return self.packer.make_can_msg("DAS_control", CANBUS.party, values)

  def right_stalk_press(self, counter, position):
    values = {
              "SCCM_rightStalkCrc": 0,
              "SCCM_rightStalkCounter": counter,
              "SCCM_rightStalkStatus": position,
              "SCCM_rightStalkReserved1": 0,
              "SCCM_parkButtonStatus": 0,
              "SCCM_rightStalkReserved2": 0,
              }

    data = self.pt_packer.make_can_msg("SCCM_rightStalk", CANBUS.vehicle, values)[2]
    values["SCCM_rightStalkCrc"] = self.right_stalk_crc(data[1:])
    return self.pt_packer.make_can_msg("SCCM_rightStalk", CANBUS.vehicle, values)
