#!/usr/bin/env python
import time
import zmq
import overpy
from datetime import datetime
from cereal import ui
import re
from selfdrive.config import Conversions as CV

from selfdrive.services import service_list
import selfdrive.messaging as messaging

def make_query(lat, lon, radius):
    pos = "  (around:%f,%f,%f)" % (radius, lat, lon)
    return """(
    way
    """ + pos + """
    [highway];
    >;);out;
    """

def parse_way(way):
    max_speed = None
    tags = way.tags

    if 'maxspeed' in tags:
        max_speed = int(re.compile("(\d+)").match(tags['maxspeed']).group(1)) #int(tags['maxspeed'])

    if 'maxspeed:conditional' in tags:
        max_speed_cond, cond = tags['maxspeed:conditional'].split(' @ ')
        cond = cond[1:-1]

        start, end = cond.split('-')
        now = datetime.now()
        start = datetime.strptime(start, "%H:%M").replace(year=now.year, month=now.month, day=now.day)
        end = datetime.strptime(end, "%H:%M").replace(year=now.year, month=now.month, day=now.day)

        if start <= now <= end:
            max_speed = int(re.compile("(\d+)").match(max_speed_cond).group(1)) #int(max_speed_cond)
    #print max_speed
    return max_speed

def get_max_speed(lat, lon, radius=5.):
    api = overpy.Overpass()

    for _ in range(10):
        result = api.query(make_query(lat, lon, radius))
        num_ways = len(result.ways)

        if num_ways == 0:
            radius *= 2.
            continue
        elif num_ways > 1:
            radius *= 0.75
            continue
        else:
            return parse_way(result.ways[0])

    return None

def speedlimitd_thread():
  context = zmq.Context()
  gps_sock = messaging.sub_sock(context, service_list['gpsLocationExternal'].port, conflate=True)
  speedlimit_sock = messaging.pub_sock(context, service_list['speedLimit'].port)

  while True:

    gps = messaging.recv_sock(gps_sock)
    if gps is not None:
      fix_ok = gps.gpsLocationExternal.flags & 1
      if fix_ok:
        lat = gps.gpsLocationExternal.latitude
        lon = gps.gpsLocationExternal.longitude

        try:
          max_speed = get_max_speed(lat, lon)

          dat = ui.SpeedLimitData.new_message()

          if max_speed:
            dat.speed = max_speed * CV.MPH_TO_MS

          speedlimit_sock.send(dat.to_bytes())
        except Exception as e:
          print(e)

def main(gctx=None):
  speedlimitd_thread()

if __name__ == "__main__":
  main()
