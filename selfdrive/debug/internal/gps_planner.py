#!/usr/bin/env python3
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import cereal.messaging as messaging
import zmq
from common.transformations.coordinates import LocalCoord
from cereal.services import service_list

SCALE = 20.

def mpc_vwr_thread(addr="127.0.0.1"):
  plt.ion()
  fig = plt.figure(figsize=(15, 15))
  ax = fig.add_subplot(1,1,1)
  ax.set_xlim([-SCALE, SCALE])
  ax.set_ylim([-SCALE, SCALE])
  ax.grid(True)

  line, = ax.plot([0.0], [0.0], ".b")
  line2, = ax.plot([0.0], [0.0], 'r')

  ax.set_aspect('equal', 'datalim')
  plt.show()

  live_location = messaging.sub_sock('liveLocation', addr=addr, conflate=True)
  gps_planner_points = messaging.sub_sock('gpsPlannerPoints', conflate=True)
  gps_planner_plan = messaging.sub_sock('gpsPlannerPlan', conflate=True)

  last_points = messaging.recv_one(gps_planner_points)
  last_plan = messaging.recv_one(gps_planner_plan)
  while True:
    p = messaging.recv_one_or_none(gps_planner_points)
    pl = messaging.recv_one_or_none(gps_planner_plan)
    ll = messaging.recv_one(live_location).liveLocation

    if p is not None:
      last_points = p
    if pl is not None:
      last_plan = pl

    if not last_plan.gpsPlannerPlan.valid:
      time.sleep(0.1)
      line2.set_color('r')
      continue

    p0 = last_points.gpsPlannerPoints.points[0]
    p0 = np.array([p0.x, p0.y, p0.z])

    n = LocalCoord.from_geodetic(np.array([ll.lat, ll.lon, ll.alt]))
    points = []
    print(len(last_points.gpsPlannerPoints.points))
    for p in last_points.gpsPlannerPoints.points:
      ecef = np.array([p.x, p.y, p.z])
      points.append(n.ecef2ned(ecef))

    points = np.vstack(points)
    line.set_xdata(points[:, 1])
    line.set_ydata(points[:, 0])

    y = np.matrix(np.arange(-100, 100.0, 0.5))
    x = -np.matrix(np.polyval(last_plan.gpsPlannerPlan.poly, y))
    xy = np.hstack([x.T, y.T])

    cur_heading = np.radians(ll.heading - 90)
    c, s = np.cos(cur_heading), np.sin(cur_heading)
    R = np.array([[c, -s], [s, c]])
    xy = xy.dot(R)

    line2.set_xdata(xy[:, 1])
    line2.set_ydata(-xy[:, 0])
    line2.set_color('g')


    ax.set_xlim([-SCALE, SCALE])
    ax.set_ylim([-SCALE, SCALE])

    fig.canvas.draw()
    fig.canvas.flush_events()



if __name__ == "__main__":
  if len(sys.argv) > 1:
    mpc_vwr_thread(sys.argv[1])
  else:
    mpc_vwr_thread()
