#!/usr/bin/env python3
"""Corner cutting: planned path offset from the model's own lane center, toward the inside of the turn.

inside > 0 means the plan cuts toward the inside of the curve. Model frame: y is left-positive and
curvature is left-positive, so inside = sign(curvature) * (plan_y - center_y).

usage: cut.py NPZ [FRAME_FROM FRAME_TO]
"""
import sys
import numpy as np

LANE = slice(117, 645)
LANE_PROB = slice(645, 653)
X_IDXS = np.array([192.0 * (i / 32) ** 2 for i in range(33)])
DISTS = (10.0, 20.0, 30.0)


def lane_center(raw):
  lanes = raw[:, LANE][:, :264].reshape(-1, 4, 33, 2)[..., 0]  # y of the 4 lines at X_IDXS
  probs = 1 / (1 + np.exp(-raw[:, LANE_PROB].reshape(-1, 4, 2)[..., 1]))
  return (lanes[:, 1] + lanes[:, 2]) / 2, np.minimum(probs[:, 1], probs[:, 2]), lanes[:, 2] - lanes[:, 1]


def inside_offset(d):
  center, prob, width = lane_center(d['raw'])
  plan = d['plan']
  out = np.zeros((len(plan), len(DISTS)), dtype=np.float32)
  for i in range(len(plan)):
    px, py = plan[i, :, 0], plan[i, :, 1]
    for j, x in enumerate(DISTS):
      out[i, j] = np.interp(x, px, py) - np.interp(x, X_IDXS, center[i])
  sign = np.sign(d["curv"])  # y is left-positive, curvature is left-positive
  return out * sign[:, None], out, prob, width


def main():
  d = np.load(sys.argv[1])
  inside, raw_off, prob, width = inside_offset(d)
  fr = d['frame']
  a, b = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (fr[0], fr[-1])
  for f in range(a, b + 1, 20):
    i = np.argmin(np.abs(fr - f))
    print(f"f{fr[i]:4d} v {d['v_ego'][i]:4.1f} curv {d['curv'][i]:+.4f} lat_acc {d['curv'][i] * d['v_ego'][i] ** 2:+.2f} | "
          f"inside offset @10/20/30 m {inside[i, 0]:+.2f} {inside[i, 1]:+.2f} {inside[i, 2]:+.2f} | lane prob {prob[i]:.2f} width {np.interp(20, X_IDXS, width[i]):.1f}")


if __name__ == '__main__':
  main()
