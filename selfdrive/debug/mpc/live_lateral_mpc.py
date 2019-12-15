#!/usr/bin/env python3
import matplotlib
matplotlib.use('TkAgg')

import sys
import selfdrive.messaging as messaging
import numpy as np
import matplotlib.pyplot as plt

# debug liateral MPC by plotting its trajectory. To receive liveLongitudinalMpc packets,
# set on LOG_MPC env variable and run plannerd on a replay


def mpc_vwr_thread(addr="127.0.0.1"):

  plt.ion()
  fig = plt.figure(figsize=(15, 20))
  ax = fig.add_subplot(131)
  aa = fig.add_subplot(132, sharey=ax)
  ap = fig.add_subplot(133, sharey=ax)

  ax.set_xlim([-10, 10])
  ax.set_ylim([0., 100.])
  aa.set_xlim([-20., 20])
  ap.set_xlim([-5, 5])

  ax.set_xlabel('x [m]')
  ax.set_ylabel('y [m]')
  aa.set_xlabel('steer_angle [deg]')
  ap.set_xlabel('asset angle [deg]')
  ax.grid(True)
  aa.grid(True)
  ap.grid(True)

  path_x = np.arange(0, 100)
  mpc_path_x = np.arange(0, 49)

  p_path_y = np.zeros(100)

  l_path_y = np.zeros(100)
  r_path_y = np.zeros(100)
  mpc_path_y = np.zeros(49)
  mpc_steer_angle = np.zeros(49)
  mpc_psi = np.zeros(49)

  line1, = ax.plot(mpc_path_y, mpc_path_x)
  # line1b, = ax.plot(mpc_path_y, mpc_path_x, 'o')

  lineP, = ax.plot(p_path_y, path_x)
  lineL, = ax.plot(l_path_y, path_x)
  lineR, = ax.plot(r_path_y, path_x)
  line3, = aa.plot(mpc_steer_angle, mpc_path_x)
  line4, = ap.plot(mpc_psi, mpc_path_x)
  ax.invert_xaxis()
  aa.invert_xaxis()
  plt.show()


  # *** log ***
  livempc = messaging.sub_sock('liveMpc', addr=addr)
  model = messaging.sub_sock('model', addr=addr)
  path_plan_sock = messaging.sub_sock('pathPlan', addr=addr)

  while 1:
    lMpc = messaging.recv_sock(livempc, wait=True)
    md = messaging.recv_sock(model)
    pp = messaging.recv_sock(path_plan_sock)

    if md is not None:
      p_poly = np.array(md.model.path.poly)
      l_poly = np.array(md.model.leftLane.poly)
      r_poly = np.array(md.model.rightLane.poly)

      p_path_y = np.polyval(p_poly, path_x)
      l_path_y = np.polyval(r_poly, path_x)
      r_path_y = np.polyval(l_poly, path_x)

    if pp is not None:
      p_path_y = np.polyval(pp.pathPlan.dPoly, path_x)
      lineP.set_xdata(p_path_y)
      lineP.set_ydata(path_x)

    if lMpc is not None:
      mpc_path_x  = list(lMpc.liveMpc.x)[1:]
      mpc_path_y  = list(lMpc.liveMpc.y)[1:]
      mpc_steer_angle  = list(lMpc.liveMpc.delta)[1:]
      mpc_psi  = list(lMpc.liveMpc.psi)[1:]

      line1.set_xdata(mpc_path_y)
      line1.set_ydata(mpc_path_x)
      lineL.set_xdata(l_path_y)
      lineL.set_ydata(path_x)
      lineR.set_xdata(r_path_y)
      lineR.set_ydata(path_x)
      line3.set_xdata(np.asarray(mpc_steer_angle)*180./np.pi * 14)
      line3.set_ydata(mpc_path_x)
      line4.set_xdata(np.asarray(mpc_psi)*180./np.pi)
      line4.set_ydata(mpc_path_x)

      aa.relim()
      aa.autoscale_view(True, scaley=True, scalex=True)

      fig.canvas.draw()
      fig.canvas.flush_events()

if __name__ == "__main__":
  if len(sys.argv) > 1:
    mpc_vwr_thread(sys.argv[1])
  else:
    mpc_vwr_thread()
