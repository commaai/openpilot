#!/usr/bin/env python3
"""Twitch metric for lateral edits: small fast left/right moves at high speed.

Per segment, the signal is band-passed to 1-3 Hz (FFT mask), then we take the std of its frame-to-frame
change on frames with v > V_MIN. Signals: curvature (replayed curv plus the change in plan curvature) and
plan y at 1 s and 2 s. Also the p99.9 of |curvature change vs master| on the same frames.

usage: exp_cut_twitch.py      prints master vs shipped lat_readout_wlo20 (from the cache)
"""
import numpy as np

from openpilot.tools.weight_lab import fit_lat_edit as F
from openpilot.tools.weight_lab.exp_cut_data import load

LAT_T = 0.25            # same as fit_lat_edit.steer_curv
MIN_STABLE_DELAY = 0.3
MIN_SPEED = 1.0
FS = 20.0
BAND = (1.0, 3.0)
V_MIN = 20.0


def interp_rows(rows, t):
  i = np.searchsorted(F.T, t) - 1
  w = (t - F.T[i]) / (F.T[i + 1] - F.T[i])
  return rows[:, i] * (1 - w) + rows[:, i + 1] * w


def steer_curv(plan, v):
  """Vectorized fit_lat_edit.steer_curv: lat_t 0.25 s, no plan distance."""
  psi = (LAT_T / MIN_STABLE_DELAY) * interp_rows(plan[:, :, 11], MIN_STABLE_DELAY)
  vc = np.maximum(v, MIN_SPEED)
  return 2 * psi / (vc * LAT_T) - plan[:, 0, 14] / vc


def band(sig):
  f = np.fft.rfftfreq(len(sig), 1 / FS)
  spec = np.fft.rfft(sig - sig.mean())
  spec[(f < BAND[0]) | (f > BAND[1])] = 0
  return np.fft.irfft(spec, len(sig))


def signals(D, plan):
  curv = D['curv'] + steer_curv(plan, D['v']) - steer_curv(D['plan'], D['v'])
  return {'curv': curv, 'y@1s': interp_rows(plan[:, :, 1], 1.0), 'y@2s': interp_rows(plan[:, :, 1], 2.0)}


def twitch(D, plan, v_min=V_MIN):
  """Std of the frame-to-frame change of the 1-3 Hz part of each signal, on frames with v > v_min."""
  sig = signals(D, plan)
  sig['added curv'] = sig['curv'] - D['curv']  # change vs master: its 1-3 Hz part is the jitter the edit adds
  out = {k: [] for k in sig}
  for sg in np.unique(D['seg']):
    idx = np.flatnonzero(D['seg'] == sg)
    idx = idx[np.argsort(D['frame'][idx])]
    if len(idx) < 100:
      continue
    step = np.diff(D['frame'][idx]) == 1
    fast = (D['v'][idx][1:] > v_min) & (D['v'][idx][1:] < 45) & step  # v ~80 m/s frames are bad speed readings
    for k, s in sig.items():
      out[k].append(np.diff(band(s[idx]))[fast])
  res = {k: float(np.std(np.concatenate(v))) for k, v in out.items()}
  fast = (D['v'] > v_min) & (D['v'] < 45)
  res['|dcurv| p99.9'] = float(np.percentile(np.abs(steer_curv(plan, D['v']) - steer_curv(D['plan'], D['v']))[fast], 99.9))
  return res


def fmt(res):
  return ' | '.join(f'{k} {v:.2e}' for k, v in res.items())


def main():
  D = load()
  print(f'twitch at v > {V_MIN} m/s, 1-3 Hz band, std of frame-to-frame change (111 base segments, 11 routes)')
  print(f'  master : {fmt(twitch(D, D["plan"]))}')
  print(f'  shipped: {fmt(twitch(D, D["plan_ship"]))}   (lat rows = master + lat_readout_wlo20)')


if __name__ == '__main__':
  main()
