from selfdrive.car.mazda.values import CAR

def create_steering_control(packer, car_fingerprint, frame, apply_steer, lkas):

  tmp = apply_steer + 2048

  lo = tmp & 0xFF
  hi = tmp >> 8

  b1 = int(lkas["BIT_1"])
  ldw = int(lkas["LDW"])
  er1= int(lkas["ERR_BIT_1"])
  lnv = 0
  er2= int(lkas["ERR_BIT_2"])

  steering_angle = int(lkas["STEERING_ANGLE"])
  b2 = int(lkas["ANGLE_ENABLED"])

  tmp = steering_angle + 2048
  ahi =  tmp >> 10
  amd =  (tmp  & 0x3FF) >> 2
  amd =  (amd >> 4) | (( amd & 0xF) << 4)
  alo =  (tmp & 0x3) << 2

  ctr = frame % 16
  # bytes:     [    1  ] [ 2 ] [             3               ]  [           4         ]
  csum = 249 - ctr - hi - lo - (lnv << 3) - er1  - (ldw << 7) - ( er2 << 4) - (b1 << 5)

  #bytes       [ 5 ] [ 6 ] [    7   ]
  csum = csum - ahi - amd - alo - b2

  if ahi == 1:
    csum = csum + 15

  if csum < 0:
    if csum < -256:
      csum = csum + 512
    else:
      csum = csum + 256

  csum = csum % 256

  if car_fingerprint == CAR.CX5:
    values = {
      "LKAS_REQUEST"     : apply_steer,
      "CTR"              : ctr,
      "ERR_BIT_1"        : er1,
      "LINE_NOT_VISIBLE" : lnv,
      "LDW"              : ldw,
      "BIT_1"            : b1,
      "ERR_BIT_2"        : er2,
      "STEERING_ANGLE"   : steering_angle,
      "ANGLE_ENABLED"    : b2,
      "CHKSUM"           : csum
    }

  return packer.make_can_msg("CAM_LKAS", 0, values)
