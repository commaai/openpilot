import numpy as np

class Conversions:
  #Speed
  MPH_TO_KPH = 1.609344
  KPH_TO_MPH = 1. / MPH_TO_KPH
  MS_TO_KPH = 3.6
  KPH_TO_MS = 1. / MS_TO_KPH
  MS_TO_MPH = MS_TO_KPH * KPH_TO_MPH
  MPH_TO_MS = MPH_TO_KPH * KPH_TO_MS
  MS_TO_KNOTS = 1.9438
  KNOTS_TO_MS = 1. / MS_TO_KNOTS
  #Angle
  DEG_TO_RAD = np.pi/180.
  RAD_TO_DEG = 1. / DEG_TO_RAD
  #Mass
  LB_TO_KG = 0.453592


RADAR_TO_CENTER = 2.7   # RADAR is ~ 2.7m ahead from center of car

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
  lidar_x, lidar_y, lidar_zoom = 384, 960, 6
  lidar_car_x, lidar_car_y = lidar_x/2., lidar_y/1.1
  car_hwidth = 1.7272/2 * lidar_zoom
  car_front = 2.6924 * lidar_zoom
  car_back  = 1.8796 * lidar_zoom
  car_color = 110

