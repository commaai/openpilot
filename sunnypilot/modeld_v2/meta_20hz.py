from openpilot.sunnypilot.modeld_v2.constants import Meta


class Meta20hz(Meta):
  ENGAGED = slice(0, 1)
  # next 2, 4, 6, 8, 10 seconds
  GAS_DISENGAGE = slice(1, 31, 6)
  BRAKE_DISENGAGE = slice(2, 31, 6)
  STEER_OVERRIDE = slice(3, 31, 6)
  HARD_BRAKE_3 = slice(4, 31, 6)
  HARD_BRAKE_4 = slice(5, 31, 6)
  HARD_BRAKE_5 = slice(6, 31, 6)
  # next 0, 2, 4, 6, 8, 10 seconds
  GAS_PRESS = slice(31, 55, 4)
  BRAKE_PRESS = slice(32, 55, 4)
  LEFT_BLINKER = slice(33, 55, 4)
  RIGHT_BLINKER = slice(34, 55, 4)
