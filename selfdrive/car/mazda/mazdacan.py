from selfdrive.car.mazda.values import CAR

def create_steering_control(packer, bus, car_fingerprint, ctr, apply_steer, lkas):
  # The checksum is calculated by subtracting all byte values across the msg from 241
  # however, the first byte is devided in half and the two halves are
  # subtracted separtaley. The second half must be subtracted from 8 first.
  # bytes 3 and 4 are constants at 32 and 2 repectively
  # for example:
  # the checksum for the msg b8 00 00 20 02 00 00 c4 would be
  #  hex: checksum = f1 - b - (8-8) - 00 - 20 - 02 - 00 - 00 = c4
  #  dec: chechsum = 241 - 11 - (8-8) - 0 - 32 - 2  - 0 - 0   = 196

  tmp = apply_steer + 2048

  lo = tmp & 0xFF
  hi = tmp >> 8

  b1 = int(lkas["BIT_1"])
  b2 = int(lkas[ "BIT_2"])
  ldw = int(lkas["LDW"])
  lnv = int(lkas["LINE_NOT_VISIBLE"])

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

  return packer.make_can_msg("CAM_LKAS", bus, values)
