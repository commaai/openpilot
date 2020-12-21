#!/usr/bin/env python3
import os
from collections import deque

import requests

import cereal.messaging as messaging
from common.basedir import PERSIST
from selfdrive.config import Conversions as CV
from selfdrive.swaglog import cloudlog

MAPBOX_ACCESS_TOKEN_PATH = PERSIST + '/mapbox/access_token'
# to make token permissions the same as comma api rsa private key
# os.chmod(PERSIST + '/mapbox/', 0o755)
# os.chmod(PERSIST + '/mapbox/access_token', 0o744)


def get_mapbox_access_token():
  if not os.path.isfile(MAPBOX_ACCESS_TOKEN_PATH):
    return None
  with open(MAPBOX_ACCESS_TOKEN_PATH, 'r') as f:
    return f.read().strip()


def try_fetch_speed_limit(gps_entries):
  json = None
  try:
    access_token = get_mapbox_access_token()
    if access_token is None:
      cloudlog.info("Mapbox access token not found:", MAPBOX_ACCESS_TOKEN_PATH)
      return None
    data = {
      'coordinates': ';'.join(f"{x.longitude},{x.latitude}" for x in gps_entries),
      'timestamps': ';'.join(str(x.timestamp) for x in gps_entries),
      'overview': 'full',
      'annotations': 'maxspeed',
      'tidy': 'true',
    }
    # print(data)
    response = requests.post('https://api.mapbox.com/matching/v5/mapbox/driving?access_token=' + access_token, data=data, timeout=10)
    json = response.json()
    # print(json)

    for maxspeed in json['matchings'][-1]['legs'][-1]['annotation']['maxspeed']:
      if 'unknown' in maxspeed:
        continue
      elif maxspeed['unit'] == 'km/h':
        return maxspeed['speed']
      elif maxspeed['unit'] == 'mph':
        return maxspeed['speed'] * CV.MPH_TO_KPH
  except Exception as e:
    cloudlog.info('Unable to retrieve speed limit from %s:' % json, e)

  return None


def main(sm=None, pm=None):
  if sm is None:
    sm = messaging.SubMaster(['gpsLocationExternal'])
  if pm is None:
    pm = messaging.PubMaster(['liveMapData'])

  gps_entries = deque(maxlen=10) # the max allowed coordinates in an api call is 100, but we shouldn't need that many

  while True:
    sm.update()

    if sm.updated['gpsLocationExternal']:
      gps = sm['gpsLocationExternal']

      if len(gps_entries) > 0:
        if gps.timestamp - gps_entries[0].timestamp < -5_000:
          gps_entries.clear() # reset history if time's out of whack (like from skipping around in unlogger)
        elif gps.timestamp - gps_entries[-1].timestamp < 5_000:
          continue # api recommends 5 second sample rate

      gps_entries.append(gps)

      msg = messaging.new_message('liveMapData')
      msg.logMonoTime = sm.logMonoTime['gpsLocationExternal']
      msg.liveMapData.lastGps = gps

      if len(gps_entries) > 2: # min allowed coordinates in an api call
        speed_limit = try_fetch_speed_limit(gps_entries)
        # print('SPEED LIMIT', speed_limit)
        if speed_limit is not None:
          msg.liveMapData.speedLimitValid = True
          msg.liveMapData.speedLimit = speed_limit
        else:
          gps_entries.clear()

      pm.send('liveMapData', msg)


if __name__ == "__main__":
  main()
