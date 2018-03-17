#!/usr/bin/env python
import zmq
from copy import copy
from selfdrive import messaging
from selfdrive.services import service_list
from cereal import log

from common.transformations.coordinates import geodetic2ecef

def main(gctx=None):
  context = zmq.Context()
  poller = zmq.Poller()
  gps_sock = messaging.sub_sock(context, service_list['gpsLocation'].port, poller)
  gps_ext_sock = messaging.sub_sock(context, service_list['gpsLocationExternal'].port, poller)
  app_sock = messaging.sub_sock(context, service_list['applanixLocation'].port, poller)
  loc_sock = messaging.pub_sock(context, service_list['liveLocation'].port)

  last_ext, last_gps, last_app = -1, -1, -1
  # 5 sec
  max_gap = 5*1e9
  preferred_type = None

  while 1:
    for sock, event in poller.poll(500):
      if sock is app_sock:
        msg = messaging.recv_one(sock)
        last_app = msg.logMonoTime
        this_type = 'app'
      if sock is gps_sock:
        msg = messaging.recv_one(sock)
        gps_pkt = msg.gpsLocation
        last_gps = msg.logMonoTime
        this_type = 'gps'
      if sock is gps_ext_sock:
        msg = messaging.recv_one(sock)
        gps_pkt = msg.gpsLocationExternal
        last_ext = msg.logMonoTime
        this_type = 'ext'

      last = max(last_gps, last_ext, last_app)

      if last_app > last - max_gap:
        new_preferred_type = 'app'
      elif last_ext > last - max_gap:
        new_preferred_type = 'ext'
      else:
        new_preferred_type = 'gps'

      if preferred_type != new_preferred_type:
        print "switching from %s to %s" % (preferred_type, new_preferred_type)
        preferred_type = new_preferred_type

      if this_type == preferred_type:
        new_msg = messaging.new_message()
        if this_type == 'app':
          # straight proxy the applanix
          new_msg.init('liveLocation')
          new_msg.liveLocation = copy(msg.applanixLocation)
        else:
          new_msg.logMonoTime = msg.logMonoTime
          new_msg.init('liveLocation')
          pkt = new_msg.liveLocation
          pkt.lat = gps_pkt.latitude
          pkt.lon = gps_pkt.longitude
          pkt.alt = gps_pkt.altitude
          pkt.speed = gps_pkt.speed
          pkt.heading = gps_pkt.bearing
          pkt.positionECEF = [float(x) for x in geodetic2ecef([pkt.lat, pkt.lon, pkt.alt])]
          pkt.source = log.LiveLocationData.SensorSource.dummy
        loc_sock.send(new_msg.to_bytes())

if __name__ == '__main__':
  main()
