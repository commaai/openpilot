from enum import Enum

class Column(Enum):
  MAKE = "Make"
  MODEL = "Model"
  PACKAGE = "Supported Package"
  LONGITUDINAL = "ACC"
  FSR_LONGITUDINAL = "No ACC accel below"
  FSR_STEERING = "No ALC below"
  STEERING_TORQUE = "Steering Torque"
  AUTO_RESUME = "Resume from stop"
  HARDWARE = "Hardware Needed"
  VIDEO = "Video"


class Star(Enum):
  FULL = "full"
  HALF = "half"
  EMPTY = "empty"
