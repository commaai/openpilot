#!/usr/bin/env python3
import json
import math
import os
import time

import requests

import cereal.messaging as messaging
from cereal import log
from common.api import Api
from common.params import Params
from selfdrive.swaglog import cloudlog


def coordinate_from_param(param):
  json_str = Params().get(param)
  if json_str is None:
    return None

  pos = json.loads(json_str)
  if 'latitude' not in pos or 'longitude' not in pos:
    return None

  return pos['latitude'], pos['longitude']


class RouteEngine:
  def __init__(self, sm, pm) -> None:
    self.sm = sm
    self.pm = pm

    self.last_bearing = None
    self.last_position = None
    self.gps_ok = False

    self.nav_destination = None
    self.segment_dx = None
    self.route = None

    self.recompute_backoff = 0
    self.recompute_countdown = 0

    if "MAPBOX_TOKEN" in os.environ:
      self.mapbox_token = os.environ["MAPBOX_TOKEN"]
      self.mapbox_host = "https://api.mapbox.com"
    else:
      self.mapbox_token = Api(Params().get("DongleId", encoding='utf8')).get_token()
      self.mapbox_host = "https://maps.comma.ai"

  def update(self):
    self.sm.update(0)
    self.update_location()
    self.recompute_route()

  def update_location(self):
    location = self.sm['liveLocationKalman']
    self.gps_ok = location.gpsOK

    localizer_valid = (location.status == log.LiveLocationKalman.Status.valid) and location.positionGeodetic.valid

    if localizer_valid:
      self.last_bearing = math.degrees(location.calibratedOrientationNED.value[2])
      self.last_position = location.positionGeodetic.value[0], location.positionGeodetic.value[1]

  def recompute_route(self):
    if self.last_position is None:
      return

    new_destination = coordinate_from_param("NavDestination")
    if new_destination is None:
      self.clear_route()
      return

    should_recompute = self.should_recompute()
    if new_destination != self.nav_destination:
      cloudlog.warning(f"Got new destination from NavDestination param {new_destination}")
      should_recompute = True

    # Don't recompute when GPS drifts in tunnels
    if not self.gps_ok and self.segment_idx is not None:
      return

    if self.recompute_countdown == 0 and should_recompute:
      self.recompute_countdown = 2**self.recompute_backoff
      self.recompute_backoff = min(6, self.recompute_backoff + 1)
      self.calculate_route(new_destination)
    else:
      self.recompute_countdown = max(0, self.recompute_countdown - 1)

  def calculate_route(self, destination):
    cloudlog.warning(f"Calculating route {self.last_position} -> {destination}")
    self.nav_destination = destination

    params = {
      'access_token': self.mapbox_token,
      # 'annotations': 'maxspeed',
      'geometries': 'geojson',
      'overview': 'full',
      'steps': 'true',
      'banner_instructions': 'true',
      'alternatives': 'false',
    }

    if self.last_bearing is not None:
      params['bearings'] = f"{(self.last_bearing + 360) % 360:.0f},90;"

    url = self.mapbox_host + f'/directions/v5/mapbox/driving-traffic/{self.last_position[1]},{self.last_position[0]};{destination[1]},{destination[0]}'
    resp = requests.get(url, params=params)

    if resp.status_code == 200:
      r = resp.json()
      if len(r['routes']):
        self.route = r['routes'][0]
      else:
        cloudlog.warning("Got empty route response")
        self.route = None

    else:
      cloudlog.warning(f"Got error in route reply {resp.status_code}")
      self.route = None

    self.send_route()

  def send_route(self):
    coords = []

    if self.route is not None:
      for step in self.route['legs'][0]['steps']:
        for c in step['geometry']['coordinates']:
          coords.append({'latitude': c[1], 'longitude': c[0]})

    msg = messaging.new_message('navRoute')
    msg.navRoute.coordinates = coords
    self.pm.send('navRoute', msg)

  def clear_route(self):
    self.route = None
    self.segment_idx = None
    self.nav_destination = None

  def should_recompute(self):
    return False


def main(sm=None, pm=None):
  if sm is None:
    sm = messaging.SubMaster(['liveLocationKalman', 'managerState'])
  if pm is None:
    pm = messaging.PubMaster(['navInstruction', 'navRoute'])

  route_engine = RouteEngine(sm, pm)
  while True:
    route_engine.update()

    # TODO: use ratekeeper
    time.sleep(1)


if __name__ == "__main__":
  main()
