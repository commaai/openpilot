#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model # pylint: disable=import-error

from tools.lib.route import Route
from tools.lib.logreader import MultiLogIterator


MIN_SAMPLES = 30*100

def to_signed(n, bits):
  if n >= (1 << max((bits - 1), 0)):
    n = n - (1 << max(bits, 0))
  return n

def get_eps_factor(lr, plot=False):

  engaged = False
  torque_cmd, eps_torque = None, None
  cmds, eps = [], []

  for msg in lr:
    if msg.which() != 'can':
      continue

    for m in msg.can:
      if m.address == 0x2e4 and m.src == 128:
        engaged = bool(m.dat[0] & 1)
        torque_cmd = to_signed((m.dat[1] << 8) | m.dat[2], 16)
      elif m.address == 0x260 and m.src == 0:
        eps_torque = to_signed((m.dat[5] << 8) | m.dat[6], 16)

    if engaged and torque_cmd is not None and eps_torque is not None:
      cmds.append(torque_cmd)
      eps.append(eps_torque)
    else:
      if len(cmds) > MIN_SAMPLES:
        break
      cmds, eps = [], []

  if len(cmds) < MIN_SAMPLES:
    raise Exception("too few samples found in route")

  lm = linear_model.LinearRegression(fit_intercept=False)
  lm.fit(np.array(cmds).reshape(-1, 1), eps)
  scale_factor = 1./lm.coef_[0]

  if plot:
    plt.plot(np.array(eps)*scale_factor)
    plt.plot(cmds)
    plt.show()
  return scale_factor

if __name__ == "__main__":
  r = Route(sys.argv[1])
  lr = MultiLogIterator(r.log_paths(), wraparound=False)
  n = get_eps_factor(lr, plot="--plot" in sys.argv)
  print("EPS torque factor: ", n)
