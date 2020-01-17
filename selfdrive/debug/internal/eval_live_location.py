#!/usr/bin/env python3
import time
import sys
import argparse
import zmq
import json
import pyproj
import numpy as np
ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

import cereal.messaging as messaging
from cereal.services import service_list

poller = zmq.Poller()
ll = messaging.sub_sock("liveLocation", poller)
tll = messaging.sub_sock("testLiveLocation", poller)

l, tl = None, None

lp = time.time()

while 1:
  polld = poller.poll(timeout=1000)
  for sock, mode in polld:
    if mode != zmq.POLLIN:
      continue
    if sock == ll:
      l = messaging.recv_one(sock)
    elif sock == tll:
      tl = messaging.recv_one(sock)
  if l is None or tl is None:
    continue

  alt_err = np.abs(l.liveLocation.alt - tl.liveLocation.alt)
  l1 = pyproj.transform(lla, ecef, l.liveLocation.lon, l.liveLocation.lat, l.liveLocation.alt)
  l2 = pyproj.transform(lla, ecef, tl.liveLocation.lon, tl.liveLocation.lat, tl.liveLocation.alt)

  al1 = pyproj.transform(lla, ecef, l.liveLocation.lon, l.liveLocation.lat, l.liveLocation.alt)
  al2 = pyproj.transform(lla, ecef, tl.liveLocation.lon, tl.liveLocation.lat, l.liveLocation.alt)

  tdiff = np.abs(l.logMonoTime - tl.logMonoTime) / 1e9

  if time.time()-lp > 0.1:
    print("tm: %f   mse: %f   mse(flat): %f   alterr: %f" % (tdiff, np.mean((np.array(l1)-np.array(l2))**2), np.mean((np.array(al1)-np.array(al2))**2), alt_err))
    lp = time.time()


