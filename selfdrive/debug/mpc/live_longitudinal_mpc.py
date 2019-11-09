#!/usr/bin/env python3

import matplotlib
matplotlib.use('TkAgg')

import sys
import selfdrive.messaging as messaging
import numpy as np
import matplotlib.pyplot as plt

N = 21

# debug longitudinal MPC by plotting its trajectory. To receive liveLongitudinalMpc packets,
# set on LOG_MPC env variable and run plannerd on a replay

def plot_longitudinal_mpc(addr="127.0.0.1"):
  # *** log ***
  livempc = messaging.sub_sock('liveLongitudinalMpc', addr=addr, conflate=True)
  radarstate = messaging.sub_sock('radarState', addr=addr, conflate=True)

  plt.ion()
  fig = plt.figure()

  t = np.hstack([np.arange(0.0, 0.8, 0.2), np.arange(0.8, 10.6, 0.6)])

  p_x_ego = fig.add_subplot(3, 2, 1)
  p_v_ego = fig.add_subplot(3, 2, 3)
  p_a_ego = fig.add_subplot(3, 2, 5)
  # p_x_l = fig.add_subplot(3, 2, 2)
  # p_a_l = fig.add_subplot(3, 2, 6)
  p_d_l = fig.add_subplot(3, 2, 2)
  p_d_l_v = fig.add_subplot(3, 2, 4)
  p_d_l_vv = fig.add_subplot(3, 2, 6)

  p_v_ego.set_ylim([0, 30])
  p_a_ego.set_ylim([-4, 4])
  p_d_l.set_ylim([-1, 10])

  p_x_ego.set_title('x')
  p_v_ego.set_title('v')
  p_a_ego.set_title('a')
  p_d_l.set_title('rel dist')

  l_x_ego, = p_x_ego.plot(t, np.zeros(N))
  l_v_ego, = p_v_ego.plot(t, np.zeros(N))
  l_a_ego, = p_a_ego.plot(t, np.zeros(N))
  l_x_l, = p_x_ego.plot(t, np.zeros(N))
  l_v_l, = p_v_ego.plot(t, np.zeros(N))
  l_a_l, = p_a_ego.plot(t, np.zeros(N))
  l_d_l, = p_d_l.plot(t, np.zeros(N))
  l_d_l_v, = p_d_l_v.plot(np.zeros(N))
  l_d_l_vv, = p_d_l_vv.plot(np.zeros(N))
  p_x_ego.legend(['ego', 'l'])
  p_v_ego.legend(['ego', 'l'])
  p_a_ego.legend(['ego', 'l'])
  p_d_l_v.set_xlabel('d_rel')
  p_d_l_v.set_ylabel('v_rel')
  p_d_l_v.set_ylim([-20, 20])
  p_d_l_v.set_xlim([0, 100])
  p_d_l_vv.set_xlabel('d_rel')
  p_d_l_vv.set_ylabel('v_rel')
  p_d_l_vv.set_ylim([-5, 5])
  p_d_l_vv.set_xlim([10, 40])

  while True:
    lMpc = messaging.recv_sock(livempc, wait=True)
    rs = messaging.recv_sock(radarstate, wait=True)

    if lMpc is not None:

      if lMpc.liveLongitudinalMpc.mpcId != 1:
        continue

      x_ego = list(lMpc.liveLongitudinalMpc.xEgo)
      v_ego = list(lMpc.liveLongitudinalMpc.vEgo)
      a_ego = list(lMpc.liveLongitudinalMpc.aEgo)
      x_l = list(lMpc.liveLongitudinalMpc.xLead)
      v_l = list(lMpc.liveLongitudinalMpc.vLead)
      # a_l = list(lMpc.liveLongitudinalMpc.aLead)
      a_l = rs.radarState.leadOne.aLeadK * np.exp(-lMpc.liveLongitudinalMpc.aLeadTau * t**2 / 2)
      #print(min(a_ego), lMpc.liveLongitudinalMpc.qpIterations)

      l_x_ego.set_ydata(x_ego)
      l_v_ego.set_ydata(v_ego)
      l_a_ego.set_ydata(a_ego)

      l_x_l.set_ydata(x_l)
      l_v_l.set_ydata(v_l)
      l_a_l.set_ydata(a_l)

      l_d_l.set_ydata(np.array(x_l) - np.array(x_ego))
      l_d_l_v.set_ydata(np.array(v_l) - np.array(v_ego))
      l_d_l_v.set_xdata(np.array(x_l) - np.array(x_ego))
      l_d_l_vv.set_ydata(np.array(v_l) - np.array(v_ego))
      l_d_l_vv.set_xdata(np.array(x_l) - np.array(x_ego))

      p_x_ego.relim()
      p_x_ego.autoscale_view(True, scaley=True, scalex=True)
      fig.canvas.draw()
      fig.canvas.flush_events()




if __name__ == "__main__":
  if len(sys.argv) > 1:
    plot_longitudinal_mpc(sys.argv[1])
  else:
    plot_longitudinal_mpc()
