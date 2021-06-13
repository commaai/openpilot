#!/usr/bin/env python
# type: ignore
import sys
import matplotlib.pyplot as plt
import numpy as np
import cereal.messaging as messaging
import time

# tool to plot one or more signals live. Call ex:
#./rqplot.py log.carState.vEgo log.carState.aEgo

# TODO: can this tool consume 10x less cpu?

def recursive_getattr(x, name):
   l = name.split('.')
   if len(l) == 1:
     return getattr(x, name)
   else:
     return recursive_getattr(getattr(x, l[0]), ".".join(l[1:]) )


if __name__ == "__main__":
  poller = messaging.Poller()

  services = []
  fields = []
  subs = []
  values = []

  plt.ion()
  fig, ax = plt.subplots()
  #fig = plt.figure(figsize=(10, 15))
  #ax = fig.add_subplot(111)
  ax.grid(True)
  fig.canvas.draw()

  subs_name = sys.argv[1:]
  lines = []
  x, y = [], []
  LEN = 500

  for i, sub in enumerate(subs_name):
    sub_split = sub.split(".")
    services.append(sub_split[0])
    fields.append(".".join(sub_split[1:]))
    subs.append(messaging.sub_sock(sub_split[0], poller))

    x.append(np.ones(LEN)*np.nan)
    y.append(np.ones(LEN)*np.nan)
    lines.append(ax.plot(x[i], y[i])[0])

  for l in lines:
    l.set_marker("*")

  cur_t = 0.
  ax.legend(subs_name)
  ax.set_xlabel('time [s]')

  while 1:
    print(1./(time.time() - cur_t))
    cur_t = time.time()
    for i, s in enumerate(subs):
      msg = messaging.recv_sock(s)
      #msg = messaging.recv_one_or_none(s)
      if msg is not None:
        x[i] = np.append(x[i], getattr(msg, 'logMonoTime') / 1e9)
        x[i] = np.delete(x[i], 0)
        y[i] = np.append(y[i], recursive_getattr(msg, subs_name[i]))
        y[i] = np.delete(y[i], 0)

        lines[i].set_xdata(x[i])
        lines[i].set_ydata(y[i])

    ax.relim()
    ax.autoscale_view(True, scaley=True, scalex=True)

    fig.canvas.blit(ax.bbox)
    fig.canvas.flush_events()

    # just a bit of wait to avoid 100% CPU usage
    time.sleep(0.001)
