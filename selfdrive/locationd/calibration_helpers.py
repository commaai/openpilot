import math

class Filter:
  MIN_SPEED = 7  # m/s  (~15.5mph)
  MAX_YAW_RATE = math.radians(3)  # per second

class Calibration:
  UNCALIBRATED = 0
  CALIBRATED = 1
  INVALID = 2
