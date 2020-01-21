from selfdrive.car.mazda.values import CAR

def create_steering_control(packer, car_fingerprint, frame, apply_steer, lkas):

  tmp = apply_steer + 2048

  lo = tmp & 0xFF
  hi = tmp >> 8

  b1 = int(lkas["BIT_1"])
  b2 = int(lkas[ "BIT_2"])
  ldw = int(lkas["LDW"])
  lnv = 0 #int(lkas["LINE_NOT_VISIBLE"])

  ctr = frame % 16

  csum = 241 - ctr - (hi - 8) - lo - (lnv << 3) - (b1 << 5)  - (b2 << 1) - (ldw << 7)

  if csum < 0:
      csum = csum + 256

  csum = csum % 256

  if car_fingerprint == CAR.CX5:
    values = {
      "CTR"              : ctr,
      "LKAS_REQUEST"     : apply_steer,
      "BIT_1"            : b1,
      "BIT_2"            : b2,
      "LDW"              : ldw,
      "LINE_NOT_VISIBLE" : lnv,
      "ERR_BIT_1"        : 0,
      "ERR_BIT_2"        : 0,
      "CHKSUM"           : csum
    }

  return packer.make_can_msg("CAM_LKAS", 0, values)
