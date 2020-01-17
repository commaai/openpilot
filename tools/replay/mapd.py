#!/usr/bin/env python

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
import zmq
from cereal.services import service_list
from selfdrive.config import Conversions as CV
import cereal.messaging as messaging


if __name__ == "__main__":
  live_map_sock = messaging.sub_sock(service_list['liveMapData'].port, conflate=True)
  plan_sock = messaging.sub_sock(service_list['plan'].port, conflate=True)

  plt.ion()
  fig = plt.figure(figsize=(8, 16))
  ax = fig.add_subplot(2, 1, 1)
  ax.set_title('Map')

  SCALE = 1000
  ax.set_xlim([-SCALE, SCALE])
  ax.set_ylim([-SCALE, SCALE])
  ax.set_xlabel('x [m]')
  ax.set_ylabel('y [m]')
  ax.grid(True)

  points_plt, = ax.plot([0.0], [0.0], "--xk")
  cur, = ax.plot([0.0], [0.0], "xr")

  speed_txt = ax.text(-500, 900, '')
  curv_txt = ax.text(-500, 775, '')

  ax = fig.add_subplot(2, 1, 2)
  ax.set_title('Curvature')
  curvature_plt, = ax.plot([0.0], [0.0], "--xk")
  ax.set_xlim([0, 500])
  ax.set_ylim([0, 1e-2])
  ax.set_xlabel('Distance along path [m]')
  ax.set_ylabel('Curvature [1/m]')
  ax.grid(True)

  plt.show()

  while True:
    m = messaging.recv_one_or_none(live_map_sock)
    p = messaging.recv_one_or_none(plan_sock)
    if p is not None:
      v = p.plan.vCurvature * CV.MS_TO_MPH
      speed_txt.set_text('Desired curvature speed: %.2f mph' % v)

    if m is not None:
      print("Current way id: %d" % m.liveMapData.wayId)
      curv_txt.set_text('Curvature valid: %s   Dist: %03.0f m\nSpeedlimit valid: %s   Speed: %.0f mph' %
                        (str(m.liveMapData.curvatureValid),
                          m.liveMapData.distToTurn,
                          str(m.liveMapData.speedLimitValid),
                          m.liveMapData.speedLimit * CV.MS_TO_MPH))

      points_plt.set_xdata(m.liveMapData.roadX)
      points_plt.set_ydata(m.liveMapData.roadY)
      curvature_plt.set_xdata(m.liveMapData.roadCurvatureX)
      curvature_plt.set_ydata(m.liveMapData.roadCurvature)

    fig.canvas.draw()
    fig.canvas.flush_events()
