import numpy as np

class Conversions:
  MPH_TO_MS = 1.609/3.6
  MS_TO_MPH = 3.6/1.609
  KPH_TO_MS = 1./3.6
  MS_TO_KPH = 3.6
  MPH_TO_KPH = 1.609
  KPH_TO_MPH = 1./1.609
  KNOTS_TO_MS = 1/1.9438
  MS_TO_KNOTS = 1.9438

  # Car tecode decimal minutes into decimal degrees, can work with numpy arrays as input
  @staticmethod
  def dm2d(dm):
    degs = np.round(dm/100.)
    mins = dm - degs*100.
    return degs + mins/60.


# Car button codes
class CruiseButtons:
  RES_ACCEL   = 4
  DECEL_SET   = 3
  CANCEL      = 2
  MAIN        = 1


# Image params for color cam on acura, calibrated on pre las vegas drive (2016-05-21)
class ImageParams:
  def __init__(self):
    self.SX_R  = 160  # top left corner pixel shift of the visual region considered by the model
    self.SY_R  = 180  # top left corner pixel shift of the visual region considered by the model
    self.VPX_R = 319  # vanishing point reference, as calibrated in Vegas drive 
    self.VPY_R = 201  # vanishing point reference, as calibrated in Vegas drive 
    self.X     = 320  # pixel length of image for model
    self.Y     = 160  # pixel length of image for model
    self.SX  = self.SX_R   # current visual region with shift
    self.SY  = self.SY_R   # current visual region with shift
    self.VPX = self.VPX_R  # current vanishing point with shift
    self.VPY = self.VPY_R  # current vanishing point with shift
  def shift(self, shift):
    def to_int(fl):
      return int(round(fl))
    # shift comes from calibration and says how much to shift the viual region 
    self.SX  = self.SX_R + to_int(shift[0])   # current visual region with shift
    self.SY  = self.SY_R + to_int(shift[1])   # current visual region with shift
    self.VPX = self.VPX_R + to_int(shift[0])  # current vanishing point with shift
    self.VPY = self.VPY_R + to_int(shift[1])  # current vanishing point with shift

class UIParams:
  lidar_x, lidar_y, lidar_zoom = 384, 960, 8
  lidar_car_x, lidar_car_y = lidar_x/2., lidar_y/1.1
  car_hwidth = 1.7272/2 * lidar_zoom
  car_front = 2.6924 * lidar_zoom
  car_back  = 1.8796 * lidar_zoom
  car_color = 110

class VehicleParams:
  def __init__(self, civic, brake_only=False, torque_mod=False):
    if civic:
      self.wheelbase = 2.67
      self.steer_ratio = 15.3
      self.slip_factor = 0.0014
      self.civic = True
    else:
      self.wheelbase = 2.67      # from http://www.edmunds.com/acura/ilx/2016/sedan/features-specs/
      self.steer_ratio = 15.3    # from http://www.edmunds.com/acura/ilx/2016/road-test-specs/
      self.slip_factor = 0.0014
      self.civic = False
    self.brake_only = brake_only
    self.torque_mod = torque_mod
    self.ui_speed_fudge = 1.01 if self.civic else 1.025

