import copy

from selfdrive.car.mazda.values import GEN1, GEN2, Buttons


def create_steering_control(packer, car_fingerprint, frame, apply_steer, lkas):

  if car_fingerprint in GEN1:
    tmp = apply_steer + 2048

    lo = tmp & 0xFF
    hi = tmp >> 8

    # copy values from camera
    b1 = int(lkas["BIT_1"])
    er1 = int(lkas["ERR_BIT_1"])
    lnv = 0
    ldw = 0
    er2 = int(lkas["ERR_BIT_2"])

    # Some older models do have these, newer models don't.
    # Either way, they all work just fine if set to zero.
    steering_angle = 0
    b2 = 0

    tmp = steering_angle + 2048
    ahi = tmp >> 10
    amd = (tmp & 0x3FF) >> 2
    amd = (amd >> 4) | (( amd & 0xF) << 4)
    alo = (tmp & 0x3) << 2

    ctr = frame % 16
    # bytes:     [    1  ] [ 2 ] [             3               ]  [           4         ]
    csum = 249 - ctr - hi - lo - (lnv << 3) - er1 - (ldw << 7) - ( er2 << 4) - (b1 << 5)

    # bytes      [ 5 ] [ 6 ] [    7   ]
    csum = csum - ahi - amd - alo - b2

    if ahi == 1:
      csum = csum + 15

    if csum < 0:
      if csum < -256:
        csum = csum + 512
      else:
        csum = csum + 256

    csum = csum % 256
    
    bus = 0
    sig_name = "CAM_LKAS"
    
    if car_fingerprint in GEN1:
      values = {
        "LKAS_REQUEST": apply_steer,
        "CTR": ctr,
        "ERR_BIT_1": er1,
        "LINE_NOT_VISIBLE" : lnv,
        "LDW": ldw,
        "BIT_1": b1,
        "ERR_BIT_2": er2,
        "STEERING_ANGLE": steering_angle,
        "ANGLE_ENABLED": b2,
        "CHKSUM": csum
      }
      
  elif car_fingerprint in GEN2:
    bus = 1
    sig_name = "EPS_LKAS"
    values = {
      "LKAS_REQUEST": apply_steer,
    }

  return packer.make_can_msg(sig_name, bus, values)


def create_alert_command(packer, cam_msg: dict, ldw: bool, steer_required: bool):
  values = copy.copy(cam_msg)
  values.update({
    # TODO: what's the difference between all these? do we need to send all?
    "HANDS_WARN_3_BITS": 0b111 if steer_required else 0,
    "HANDS_ON_STEER_WARN": steer_required,
    "HANDS_ON_STEER_WARN_2": steer_required,

    # TODO: right lane works, left doesn't
    # TODO: need to do something about L/R
    "LDW_WARN_LL": 0,
    "LDW_WARN_RL": 0,
  })
  return packer.make_can_msg("CAM_LANEINFO", 0, values)


def create_button_cmd(packer, car_fingerprint, counter, button):

  can = int(button == Buttons.CANCEL)
  res = int(button == Buttons.RESUME)

  if car_fingerprint in GEN1:
    values = {
      "CAN_OFF": can,
      "CAN_OFF_INV": (can + 1) % 2,

      "SET_P": 0,
      "SET_P_INV": 1,

      "RES": res,
      "RES_INV": (res + 1) % 2,

      "SET_M": 0,
      "SET_M_INV": 1,

      "DISTANCE_LESS": 0,
      "DISTANCE_LESS_INV": 1,

      "DISTANCE_MORE": 0,
      "DISTANCE_MORE_INV": 1,

      "MODE_X": 0,
      "MODE_X_INV": 1,

      "MODE_Y": 0,
      "MODE_Y_INV": 1,

      "BIT1": 1,
      "BIT2": 1,
      "BIT3": 1,
      "CTR": (counter + 1) % 16,
    }

    return packer.make_can_msg("CRZ_BTNS", 0, values)

def create_acc_cmd(self, packer, CS, CC, hold, resume):
  if self.CP.carFingerprint in GEN2:
    values = CS.acc
    msg_name = "ACC"
    bus = 2

    if (values["ACC_ENABLED"]):
      values["ACCEL_CMD"] = (CC.actuators.accel * 300) + 2000
      values["HOLD"] = hold
      values["RESUME"] = resume
    else:
      pass

  return packer.make_can_msg(msg_name, bus, values)