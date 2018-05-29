# Car button codes
class CruiseButtons:
  # VAL_ 69 SpdCtrlLvr_Stat 32 "DN_1ST" 16 "UP_1ST" 8 "DN_2ND" 4 "UP_2ND" 2 "RWD" 1 "FWD" 0 "IDLE" ;
  RES_ACCEL   = 16
  DECEL_SET   = 32
  CANCEL      = 1
  MAIN        = 2


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
  FCW            = [1, 1]
  STEER          = [2, 1]
  BRAKE_PRESSED  = [3, 10]
  GEAR_NOT_D     = [4, 6]
  SEATBELT       = [5, 5]
  SPEED_TOO_HIGH = [6, 8]
