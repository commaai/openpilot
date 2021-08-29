#!/usr/bin/env python3

import os
import sys

import numpy as np

from rednose.helpers import TEMPLATE_DIR, load_code, write_code
from rednose.helpers.sympy_helpers import quat_matrix_l, rot_matrix


def sane(track):
  img_pos = track[1:, 2:4]
  diffs_x = abs(img_pos[1:, 0] - img_pos[:-1, 0])
  diffs_y = abs(img_pos[1:, 1] - img_pos[:-1, 1])
  for i in range(1, len(diffs_x)):
    if ((diffs_x[i] > 0.05 or diffs_x[i - 1] > 0.05) and
        (diffs_x[i] > 2 * diffs_x[i - 1] or
         diffs_x[i] < .5 * diffs_x[i - 1])) or \
       ((diffs_y[i] > 0.05 or diffs_y[i - 1] > 0.05) and
        (diffs_y[i] > 2 * diffs_y[i - 1] or
         diffs_y[i] < .5 * diffs_y[i - 1])):
      return False
  return True


class FeatureHandler():
  name = 'feature_handler'

  @staticmethod
  def generate_code(generated_dir, K=5):
    # Wrap c code for slow matching
    c_header = "\nvoid merge_features(double *tracks, double *features, long long *empty_idxs);"

    c_code = "#include <math.h>\n"
    c_code += "#include <string.h>\n"
    c_code += "#define K %d\n" % K
    c_code += "extern \"C\" {\n"
    c_code += "\n" + open(os.path.join(TEMPLATE_DIR, "feature_handler.c")).read()
    c_code += "\n}\n"

    filename = f"{FeatureHandler.name}_{K}"
    write_code(generated_dir, filename, c_code, c_header)

  def __init__(self, generated_dir, K=5):
    self.MAX_TRACKS = 6000
    self.K = K

    # Array of tracks, each track has K 5D features preceded
    # by 5 params that inidicate [f_idx, last_idx, updated, complete, valid]
    # f_idx: idx of current last feature in track
    # idx of of last feature in frame
    # bool for whether this track has been update
    # bool for whether this track is complete
    # bool for whether this track is valid
    self.tracks = np.zeros((self.MAX_TRACKS, K + 1, 5))
    self.tracks[:] = np.nan

    name = f"{FeatureHandler.name}_{K}"
    ffi, lib = load_code(generated_dir, name)

    def merge_features_c(tracks, features, empty_idxs):
      lib.merge_features(ffi.cast("double *", tracks.ctypes.data),
                         ffi.cast("double *", features.ctypes.data),
                         ffi.cast("long long *", empty_idxs.ctypes.data))

    # self.merge_features = self.merge_features_python
    self.merge_features = merge_features_c

  def reset(self):
    self.tracks[:] = np.nan

  def merge_features_python(self, tracks, features, empty_idxs):
    empty_idx = 0
    for f in features:
      match_idx = int(f[4])
      if tracks[match_idx, 0, 1] == match_idx and tracks[match_idx, 0, 2] == 0:
        tracks[match_idx, 0, 0] += 1
        tracks[match_idx, 0, 1] = f[1]
        tracks[match_idx, 0, 2] = 1
        tracks[match_idx, int(tracks[match_idx, 0, 0])] = f
        if tracks[match_idx, 0, 0] == self.K:
          tracks[match_idx, 0, 3] = 1
          if sane(tracks[match_idx]):
            tracks[match_idx, 0, 4] = 1
      else:
        if empty_idx == len(empty_idxs):
          print('need more empty space')
          continue
        tracks[empty_idxs[empty_idx], 0, 0] = 1
        tracks[empty_idxs[empty_idx], 0, 1] = f[1]
        tracks[empty_idxs[empty_idx], 0, 2] = 1
        tracks[empty_idxs[empty_idx], 1] = f
        empty_idx += 1

  def update_tracks(self, features):
    last_idxs = np.copy(self.tracks[:, 0, 1])
    real = np.isfinite(last_idxs)
    self.tracks[last_idxs[real].astype(int)] = self.tracks[real]

    mask = np.ones(self.MAX_TRACKS, np.bool)
    mask[last_idxs[real].astype(int)] = 0
    empty_idxs = np.arange(self.MAX_TRACKS)[mask]

    self.tracks[empty_idxs] = np.nan
    self.tracks[:, 0, 2] = 0
    self.merge_features(self.tracks, features, empty_idxs)

  def handle_features(self, features):
    self.update_tracks(features)
    valid_idxs = self.tracks[:, 0, 4] == 1
    complete_idxs = self.tracks[:, 0, 3] == 1
    stale_idxs = self.tracks[:, 0, 2] == 0
    valid_tracks = self.tracks[valid_idxs]
    self.tracks[complete_idxs] = np.nan
    self.tracks[stale_idxs] = np.nan
    return valid_tracks[:, 1:, :4].reshape((len(valid_tracks), self.K * 4))


def generate_orient_error_jac(K):
  import sympy as sp
  from rednose.helpers.sympy_helpers import quat_rotate

  x_sym = sp.MatrixSymbol('abr', 3, 1)
  dtheta = sp.MatrixSymbol('dtheta', 3, 1)
  delta_quat = sp.Matrix(np.ones(4))
  delta_quat[1:, :] = sp.Matrix(0.5 * dtheta[0:3, :])
  poses_sym = sp.MatrixSymbol('poses', 7 * K, 1)
  img_pos_sym = sp.MatrixSymbol('img_positions', 2 * K, 1)
  alpha, beta, rho = x_sym
  to_c = sp.Matrix(rot_matrix(-np.pi / 2, -np.pi / 2, 0))
  pos_0 = sp.Matrix(np.array(poses_sym[K * 7 - 7:K * 7 - 4])[:, 0])
  q = quat_matrix_l(poses_sym[K * 7 - 4:K * 7]) * delta_quat
  quat_rot = quat_rotate(*q)
  rot_g_to_0 = to_c * quat_rot.T
  rows = []
  for i in range(K):
    pos_i = sp.Matrix(np.array(poses_sym[i * 7:i * 7 + 3])[:, 0])
    q = quat_matrix_l(poses_sym[7 * i + 3:7 * i + 7]) * delta_quat
    quat_rot = quat_rotate(*q)
    rot_g_to_i = to_c * quat_rot.T
    rot_0_to_i = rot_g_to_i * (rot_g_to_0.T)
    trans_0_to_i = rot_g_to_i * (pos_0 - pos_i)
    funct_vec = rot_0_to_i * sp.Matrix([alpha, beta, 1]) + rho * trans_0_to_i
    h1, h2, h3 = funct_vec
    rows.append(h1 / h3 - img_pos_sym[i * 2 + 0])
    rows.append(h2 / h3 - img_pos_sym[i * 2 + 1])
  img_pos_residual_sym = sp.Matrix(rows)

  # sympy into c
  sympy_functions = []
  sympy_functions.append(('orient_error_jac', img_pos_residual_sym.jacobian(dtheta), [x_sym, poses_sym, img_pos_sym, dtheta]))

  return sympy_functions


if __name__ == "__main__":
  K = int(sys.argv[1].split("_")[-1])
  generated_dir = sys.argv[2]
  FeatureHandler.generate_code(generated_dir, K=K)
