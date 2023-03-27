from common.numpy_fast import clip
from selfdrive.car.hyundai.values import HyundaiFlags

def get_e_can_bus(CP):
  # On the CAN-FD platforms, the LKAS camera is on both A-CAN and E-CAN. HDA2 cars
  # have a different harness than the HDA1 and non-HDA variants in order to split
  # a different bus, since the steering is done by different ECUs.
  return 5 if CP.flags & HyundaiFlags.CANFD_HDA2 else 4


def create_steering_messages(packer, CP, enabled, lat_active, apply_steer):

  ret = []

  values = {
    "LKA_MODE": 2,
    "LKA_ICON": 2 if enabled else 1,
    "TORQUE_REQUEST": apply_steer,
    "LKA_ASSIST": 0,
    "STEER_REQ": 1 if lat_active else 0,
    "STEER_MODE": 0,
    "SET_ME_1": 0,
    "NEW_SIGNAL_1": 0,
    "NEW_SIGNAL_2": 0,
  }

  if CP.flags & HyundaiFlags.CANFD_HDA2:
    if CP.openpilotLongitudinalControl:
      ret.append(packer.make_can_msg("LFA", 5, values))
    ret.append(packer.make_can_msg("LKAS", 4, values))
  else:
    ret.append(packer.make_can_msg("LFA", 4, values))

  return ret

def create_cam_0x2a4(packer, camera_values):
  camera_values.update({
    "BYTE7": 0,
  })
  return packer.make_can_msg("CAM_0x2a4", 4, camera_values)

def create_buttons(packer, CP, cnt, btn):
  values = {
    "COUNTER": cnt,
    "SET_ME_1": 1,
    "CRUISE_BUTTONS": btn,
  }

  bus = 5 if CP.flags & HyundaiFlags.CANFD_HDA2 else 6
  return packer.make_can_msg("CRUISE_BUTTONS", bus, values)

def create_acc_cancel(packer, CP, cruise_info_copy):
  values = cruise_info_copy
  values.update({
    "ACCMode": 4,
  })
  return packer.make_can_msg("SCC_CONTROL", get_e_can_bus(CP), values)

def create_lfahda_cluster(packer, CP, enabled):
  values = {
    "HDA_ICON": 1 if enabled else 0,
    "LFA_ICON": 2 if enabled else 0,
  }
  return packer.make_can_msg("LFAHDA_CLUSTER", get_e_can_bus(CP), values)


def create_acc_control(packer, CP, enabled, accel_last, accel, stopping, gas_override, set_speed):
  jerk = 5
  jn = jerk / 50
  if not enabled or gas_override:
    a_val, a_raw = 0, 0
  else:
    a_raw = accel
    a_val = clip(accel, accel_last - jn, accel_last + jn)

  values = {
    "ACCMode": 0 if not enabled else (2 if gas_override else 1),
    "MainMode_ACC": 1,
    "StopReq": 1 if stopping else 0,
    "aReqValue": a_val,
    "aReqRaw": a_raw,
    "VSetDis": set_speed,
    "JerkLowerLimit": jerk if enabled else 1,
    "JerkUpperLimit": 3.0,

    "ACC_ObjDist": 1,
    "ObjValid": 0,
    "OBJ_STATUS": 2,
    "SET_ME_2": 0x4,
    "SET_ME_3": 0x3,
    "SET_ME_TMP_64": 0x64,
    "DISTANCE_SETTING": 4,
  }

  return packer.make_can_msg("SCC_CONTROL", get_e_can_bus(CP), values)


def create_spas_messages(packer, frame, left_blink, right_blink):
  ret = []

  values = {
  }
  ret.append(packer.make_can_msg("SPAS1", 5, values))

  blink = 0
  if left_blink:
    blink = 3
  elif right_blink:
    blink = 4
  values = {
    "BLINKER_CONTROL": blink,
  }
  ret.append(packer.make_can_msg("SPAS2", 5, values))

  return ret


def create_adrv_messages(packer, frame):
  # messages needed to car happy after disabling
  # the ADAS Driving ECU to do longitudinal control

  ret = []

  values = {
  }
  ret.append(packer.make_can_msg("ADRV_0x51", 4, values))

  if frame % 2 == 0:
    values = {
      'AEB_SETTING': 0x1,  # show AEB disabled icon
      'SET_ME_2': 0x2,
      'SET_ME_FF': 0xff,
      'SET_ME_FC': 0xfc,
      'SET_ME_9': 0x9,
    }
    ret.append(packer.make_can_msg("ADRV_0x160", 5, values))

  if frame % 5 == 0:
    values = {
      'SET_ME_1C': 0x1c,
      'SET_ME_FF': 0xff,
      'SET_ME_TMP_F': 0xf,
      'SET_ME_TMP_F_2': 0xf,
    }
    ret.append(packer.make_can_msg("ADRV_0x1ea", 5, values))

    values = {
      'SET_ME_E1': 0xe1,
      'SET_ME_3A': 0x3a,
    }
    ret.append(packer.make_can_msg("ADRV_0x200", 5, values))

  if frame % 20 == 0:
    values = {
      'SET_ME_15': 0x15,
    }
    ret.append(packer.make_can_msg("ADRV_0x345", 5, values))

  if frame % 100 == 0:
    values = {
      'SET_ME_22': 0x22,
      'SET_ME_41': 0x41,
    }
    ret.append(packer.make_can_msg("ADRV_0x1da", 5, values))

  return ret
