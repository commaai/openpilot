import numpy as np

class SpeedConversion:
  """Constants for speed unit conversions."""
  MPH_TO_KPH = 1.609344  # Miles per hour to kilometers per hour
  KPH_TO_MPH = 1. / MPH_TO_KPH
  MS_TO_KPH = 3.6  # Meters per second to kilometers per hour
  KPH_TO_MS = 1. / MS_TO_KPH
  MS_TO_MPH = MS_TO_KPH * KPH_TO_MPH
  MPH_TO_MS = MPH_TO_KPH * KPH_TO_MS
  MS_TO_KNOTS = 1.9438  # Meters per second to knots
  KNOTS_TO_MS = 1. / MS_TO_KNOTS

class AngularConversion:
  """Constants for angular unit conversions."""
  DEG_TO_RAD = np.pi / 180.  # Degrees to radians
  RAD_TO_DEG = 1. / DEG_TO_RAD

class MassConversion:
  """Constants for mass unit conversions."""
  LB_TO_KG = 0.453592  # Pounds to kilograms

# Global physics constants
GRAVITY_ACCELERATION = 9.81  # m/s^2, acceleration due to gravity


class CV:
  """Legacy namespace for backward compatibility."""
  # Speed
  MPH_TO_KPH = SpeedConversion.MPH_TO_KPH
  KPH_TO_MPH = SpeedConversion.KPH_TO_MPH
  MS_TO_KPH = SpeedConversion.MS_TO_KPH
  KPH_TO_MS = SpeedConversion.KPH_TO_MS
  MS_TO_MPH = SpeedConversion.MS_TO_MPH
  MPH_TO_MS = SpeedConversion.MPH_TO_MS
  MS_TO_KNOTS = SpeedConversion.MS_TO_KNOTS
  KNOTS_TO_MS = SpeedConversion.KNOTS_TO_MS

  # Angle
  DEG_TO_RAD = AngularConversion.DEG_TO_RAD
  RAD_TO_DEG = AngularConversion.RAD_TO_DEG

  # Mass
  LB_TO_KG = MassConversion.LB_TO_KG
