#!/usr/bin/env python3
import os
import json
import numpy as np
from common.basedir import BASEDIR
from common.realtime import sec_since_boot
from shapely.geometry import Point, Polygon

LATITUDE_DEG_TO_M = 111133 # ~111km per deg is the mean between equator and poles
GEOFENCE_THRESHOLD = 8.
LOC_FILTER_F = 0.5   # 0.5Hz
DT = 0.1             # external gps packets are at 10Hz
LOC_FILTER_K = 2 * np.pi * LOC_FILTER_F * DT / (1 + 2 * np.pi * LOC_FILTER_F * DT)

class Geofence():
  def __init__(self, active):
    self.lat_filt = None
    self.lon_filt = None
    self.ts_last_check = 0.
    self.active = active
    # hack: does not work at north/south poles, and when longitude is ~180
    self.in_geofence = not active  # initialize false if geofence is active
    # get full geofenced polygon in lat and lon coordinates
    geofence_polygon = np.load(os.path.join(BASEDIR, 'selfdrive/controls/geofence_routes/press_demo.npy'))
    # for small latitude variations, we can assume constant conversion between longitude and meters (use the first point)
    self.longitude_deg_to_m = LATITUDE_DEG_TO_M * np.cos(np.radians(geofence_polygon[0,0]))
    # convert to m
    geofence_polygon_m = geofence_polygon * LATITUDE_DEG_TO_M
    geofence_polygon_m[:, 1] = geofence_polygon[:,1] * self.longitude_deg_to_m
    self.geofence_polygon = Polygon(geofence_polygon_m)


  def update_geofence_status(self, gps_loc, params):

    if self.lat_filt is None:
      # first time we get a location packet
      self.latitude = gps_loc.latitude
      self.longitude = gps_loc.longitude
    else:
      # apply a filter
      self.latitude = LOC_FILTER_K * gps_loc.latitude + (1. - LOC_FILTER_K) * self.latitude
      self.longitude = LOC_FILTER_K * gps_loc.longitude + (1. - LOC_FILTER_K) * self.longitude

    ts = sec_since_boot()

    if ts - self.ts_last_check > 1.:  # tun check at 1Hz, since is computationally intense
      self.active = params.get("IsGeofenceEnabled") == "1"
      self.ts_last_check = ts

      p = Point([self.latitude * LATITUDE_DEG_TO_M, self.longitude * self.longitude_deg_to_m])

      # histeresys
      geofence_distance = self.geofence_polygon.distance(p)
      if self.in_geofence and geofence_distance > GEOFENCE_THRESHOLD and self.active:
        self.in_geofence = False
      elif (not self.in_geofence and geofence_distance < 1.) or not self.active:
        self.in_geofence = True


if __name__ == "__main__":
  from common.params import Params
  # for tests
  params = Params()
  gf = Geofence(True)
  class GpsPos():
    def __init__(self, lat, lon):
      self.latitude = lat
      self.longitude = lon

  #pos = GpsPos(37.687347, -122.471117)  # True
  pos = GpsPos(37.687347, -122.571117)  # False
  gf.update_geofence_status(pos, params)
  print(gf.in_geofence)
