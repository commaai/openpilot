# Car button codes
class CruiseButtons:
  RES_ACCEL   = 4
  DECEL_SET   = 3
  CANCEL      = 2
  MAIN        = 1


#car chimes: enumeration from dbc file. Chimes are for alerts and warnings
class CM:
  MUTE = 0
  SINGLE = 3
  DOUBLE = 4
  REPEATED = 1
  CONTINUOUS = 2


#car beepss: enumeration from dbc file. Beeps are for activ and deactiv
class BP:
  MUTE = 0
  SINGLE = 3
  TRIPLE = 2
  REPEATED = 1

class AH:
  #[alert_idx, value]
  # See dbc files for info on values"
  NONE           = [0, 0]
  FCW            = [1, 0x8]
  STEER          = [2, 1]
  BRAKE_PRESSED  = [3, 10]
  GEAR_NOT_D     = [4, 6]
  SEATBELT       = [5, 5]
  SPEED_TOO_HIGH = [6, 8]
