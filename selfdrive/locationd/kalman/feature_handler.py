import common.transformations.orientation as orient
import numpy as np
import scipy.optimize as opt
import time
import os
from bisect import bisect_left
from common.sympy_helpers import sympy_into_c, quat_matrix_l
from common.ffi_wrapper import ffi_wrap, wrap_compiled, compile_code

EXTERNAL_PATH = os.path.dirname(os.path.abspath(__file__))


def sane(track):
  img_pos = track[1:,2:4]
  diffs_x = abs(img_pos[1:,0] - img_pos[:-1,0])
  diffs_y = abs(img_pos[1:,1] - img_pos[:-1,1])
  for i in range(1, len(diffs_x)):
    if ((diffs_x[i] > 0.05 or diffs_x[i-1] > 0.05) and \
       (diffs_x[i] > 2*diffs_x[i-1] or \
        diffs_x[i] < .5*diffs_x[i-1])) or \
       ((diffs_y[i] > 0.05 or diffs_y[i-1] > 0.05) and \
       (diffs_y[i] > 2*diffs_y[i-1] or \
        diffs_y[i] < .5*diffs_y[i-1])):
      return False
  return True

class FeatureHandler():
  def __init__(self, K):
    self.MAX_TRACKS=6000
    self.K = K
    #Array of tracks, each track
    #has K 5D features preceded
    #by 5 params that inidicate
    #[f_idx, last_idx, updated, complete, valid]
    # f_idx: idx of current last feature in track
    # idx of of last feature in frame
    # bool for whether this track has been update
    # bool for whether this track is complete
    # bool for whether this track is valid
    self.tracks = np.zeros((self.MAX_TRACKS, K+1, 5))
    self.tracks[:] = np.nan

    # Wrap c code for slow matching
    c_header = "\nvoid merge_features(double *tracks, double *features, long long *empty_idxs);"
    c_code = "#define K %d\n" % K
    c_code += "\n" + open(os.path.join(EXTERNAL_PATH, "feature_handler.c")).read()
    ffi, lib = ffi_wrap('feature_handler', c_code, c_header)
    def merge_features_c(tracks, features, empty_idxs):
      lib.merge_features(ffi.cast("double *", tracks.ctypes.data),
                    ffi.cast("double *", features.ctypes.data),
                    ffi.cast("long long *", empty_idxs.ctypes.data))

    #self.merge_features = self.merge_features_python
    self.merge_features = merge_features_c

  def reset(self):
    self.tracks[:] = np.nan

  def merge_features_python(self, tracks, features, empty_idxs):
    empty_idx = 0
    for f in features:
      match_idx = int(f[4])
      if tracks[match_idx, 0, 1] == match_idx and tracks[match_idx, 0 ,2] == 0:
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
    t0 = time.time()
    last_idxs = np.copy(self.tracks[:,0,1])
    real = np.isfinite(last_idxs)
    self.tracks[last_idxs[real].astype(int)] = self.tracks[real]
    mask = np.ones(self.MAX_TRACKS, np.bool)
    mask[last_idxs[real].astype(int)] = 0
    empty_idxs = np.arange(self.MAX_TRACKS)[mask]
    self.tracks[empty_idxs] = np.nan
    self.tracks[:,0,2] = 0
    self.merge_features(self.tracks, features, empty_idxs)

  def handle_features(self, features):
    self.update_tracks(features)
    valid_idxs = self.tracks[:,0,4] == 1
    complete_idxs = self.tracks[:,0,3] == 1
    stale_idxs = self.tracks[:,0,2] == 0
    valid_tracks = self.tracks[valid_idxs]
    self.tracks[complete_idxs] = np.nan
    self.tracks[stale_idxs] = np.nan
    return valid_tracks[:,1:,:4].reshape((len(valid_tracks), self.K*4))

def generate_residual(K):
  import sympy as sp
  from common.sympy_helpers import quat_rotate
  x_sym = sp.MatrixSymbol('abr', 3,1)
  poses_sym = sp.MatrixSymbol('poses', 7*K,1)
  img_pos_sym = sp.MatrixSymbol('img_positions', 2*K,1)
  alpha, beta, rho = x_sym
  to_c = sp.Matrix(orient.rot_matrix(-np.pi/2, -np.pi/2, 0))
  pos_0 = sp.Matrix(np.array(poses_sym[K*7-7:K*7-4])[:,0])
  q = poses_sym[K*7-4:K*7]
  quat_rot = quat_rotate(*q)
  rot_g_to_0 = to_c*quat_rot.T
  rows = []
  for i in range(K):
    pos_i = sp.Matrix(np.array(poses_sym[i*7:i*7+3])[:,0])
    q = poses_sym[7*i+3:7*i+7]
    quat_rot = quat_rotate(*q)
    rot_g_to_i = to_c*quat_rot.T
    rot_0_to_i = rot_g_to_i*(rot_g_to_0.T)
    trans_0_to_i = rot_g_to_i*(pos_0 - pos_i)
    funct_vec = rot_0_to_i*sp.Matrix([alpha, beta, 1]) + rho*trans_0_to_i
    h1, h2, h3 = funct_vec
    rows.append(h1/h3 - img_pos_sym[i*2 +0])
    rows.append(h2/h3 - img_pos_sym[i*2 + 1])
  img_pos_residual_sym = sp.Matrix(rows)

  # sympy into c
  sympy_functions = []
  sympy_functions.append(('res_fun', img_pos_residual_sym, [x_sym, poses_sym, img_pos_sym]))
  sympy_functions.append(('jac_fun', img_pos_residual_sym.jacobian(x_sym), [x_sym, poses_sym, img_pos_sym]))

  return sympy_functions


def generate_orient_error_jac(K):
  import sympy as sp
  from common.sympy_helpers import quat_rotate
  x_sym = sp.MatrixSymbol('abr', 3,1)
  dtheta = sp.MatrixSymbol('dtheta', 3,1)
  delta_quat = sp.Matrix(np.ones(4))
  delta_quat[1:,:] = sp.Matrix(0.5*dtheta[0:3,:])
  poses_sym = sp.MatrixSymbol('poses', 7*K,1)
  img_pos_sym = sp.MatrixSymbol('img_positions', 2*K,1)
  alpha, beta, rho = x_sym
  to_c = sp.Matrix(orient.rot_matrix(-np.pi/2, -np.pi/2, 0))
  pos_0 = sp.Matrix(np.array(poses_sym[K*7-7:K*7-4])[:,0])
  q = quat_matrix_l(poses_sym[K*7-4:K*7])*delta_quat
  quat_rot = quat_rotate(*q)
  rot_g_to_0 = to_c*quat_rot.T
  rows = []
  for i in range(K):
    pos_i = sp.Matrix(np.array(poses_sym[i*7:i*7+3])[:,0])
    q = quat_matrix_l(poses_sym[7*i+3:7*i+7])*delta_quat
    quat_rot = quat_rotate(*q)
    rot_g_to_i = to_c*quat_rot.T
    rot_0_to_i = rot_g_to_i*(rot_g_to_0.T)
    trans_0_to_i = rot_g_to_i*(pos_0 - pos_i)
    funct_vec = rot_0_to_i*sp.Matrix([alpha, beta, 1]) + rho*trans_0_to_i
    h1, h2, h3 = funct_vec
    rows.append(h1/h3 - img_pos_sym[i*2 +0])
    rows.append(h2/h3 - img_pos_sym[i*2 + 1])
  img_pos_residual_sym = sp.Matrix(rows)

  # sympy into c
  sympy_functions = []
  sympy_functions.append(('orient_error_jac', img_pos_residual_sym.jacobian(dtheta), [x_sym, poses_sym, img_pos_sym, dtheta]))

  return sympy_functions


class LstSqComputer():
  def __init__(self, K, MIN_DEPTH=2, MAX_DEPTH=500, debug=False):
    self.to_c = orient.rot_matrix(-np.pi/2, -np.pi/2, 0)
    self.MAX_DEPTH = MAX_DEPTH
    self.MIN_DEPTH = MIN_DEPTH
    self.debug = debug
    self.name = 'pos_computer_' + str(K)
    if debug:
      self.name += '_debug'

    try:
      dir_path = os.path.dirname(__file__)
      deps = [dir_path + '/' + 'feature_handler.py',
              dir_path + '/' + 'compute_pos.c']

      outs = [dir_path + '/' + self.name + '.o',
              dir_path + '/' + self.name + '.so',
              dir_path + '/' + self.name + '.cpp']
      out_times = list(map(os.path.getmtime, outs))
      dep_times = list(map(os.path.getmtime, deps))
      rebuild = os.getenv("REBUILD", False)
      if min(out_times) < max(dep_times) or rebuild:
        list(map(os.remove, outs))
        # raise the OSError if removing didnt
        # raise one to start the compilation
        raise OSError()
    except OSError as e:
      # gen c code for sympy functions
      sympy_functions = generate_residual(K)
      #if debug:
      #  sympy_functions.extend(generate_orient_error_jac(K))
      header, code = sympy_into_c(sympy_functions)

      # ffi wrap c code
      extra_header = "\nvoid compute_pos(double *to_c, double *in_poses, double *in_img_positions, double *param, double *pos);"
      code += "\n#define KDIM %d\n" % K
      header += "\n" + extra_header
      code += "\n" + open(os.path.join(EXTERNAL_PATH, "compute_pos.c")).read()
      compile_code(self.name, code, header, EXTERNAL_PATH)
    ffi, lib = wrap_compiled(self.name, EXTERNAL_PATH)

    # wrap c functions
    #if debug:
      #def orient_error_jac(x, poses, img_positions, dtheta):
      #  out = np.zeros(((K*2, 3)), dtype=np.float64)
      #  lib.orient_error_jac(ffi.cast("double *", x.ctypes.data),
      #    ffi.cast("double *", poses.ctypes.data),
      #    ffi.cast("double *", img_positions.ctypes.data),
      #    ffi.cast("double *", dtheta.ctypes.data),
      #    ffi.cast("double *", out.ctypes.data))
      #  return out
      #self.orient_error_jac = orient_error_jac
    def residual_jac(x, poses, img_positions):
      out = np.zeros(((K*2, 3)), dtype=np.float64)
      lib.jac_fun(ffi.cast("double *", x.ctypes.data),
        ffi.cast("double *", poses.ctypes.data),
        ffi.cast("double *", img_positions.ctypes.data),
        ffi.cast("double *", out.ctypes.data))
      return out
    def residual(x, poses, img_positions):
      out = np.zeros((K*2), dtype=np.float64)
      lib.res_fun(ffi.cast("double *", x.ctypes.data),
          ffi.cast("double *", poses.ctypes.data),
          ffi.cast("double *", img_positions.ctypes.data),
          ffi.cast("double *", out.ctypes.data))
      return out
    self.residual = residual
    self.residual_jac = residual_jac

    def compute_pos_c(poses, img_positions):
      pos = np.zeros(3, dtype=np.float64)
      param = np.zeros(3, dtype=np.float64)
      # Can't be a view for the ctype
      img_positions = np.copy(img_positions)
      lib.compute_pos(ffi.cast("double *", self.to_c.ctypes.data),
          ffi.cast("double *", poses.ctypes.data),
          ffi.cast("double *", img_positions.ctypes.data),
          ffi.cast("double *", param.ctypes.data),
          ffi.cast("double *", pos.ctypes.data))
      return pos, param
    self.compute_pos_c = compute_pos_c

  def compute_pos(self, poses, img_positions, debug=False):
    pos, param =  self.compute_pos_c(poses, img_positions)
    #pos, param =  self.compute_pos_python(poses, img_positions)
    depth = 1/param[2]
    if debug:
      if not self.debug:
        raise NotImplementedError("This is not a debug computer")
      #orient_err_jac = self.orient_error_jac(param, poses, img_positions, np.zeros(3)).reshape((-1,2,3))
      jac = self.residual_jac(param, poses, img_positions).reshape((-1,2,3))
      res = self.residual(param, poses, img_positions).reshape((-1,2))
      return pos, param, res, jac #, orient_err_jac
    elif (self.MIN_DEPTH < depth < self.MAX_DEPTH):
      return pos
    else:
      return None

  def gauss_newton(self, fun, jac, x, args):
    poses, img_positions = args
    delta = 1
    counter = 0
    while abs(np.linalg.norm(delta)) > 1e-4 and counter < 30:
      delta = np.linalg.pinv(jac(x, poses, img_positions)).dot(fun(x, poses, img_positions))
      x = x - delta
      counter += 1
    return [x]

  def compute_pos_python(self, poses, img_positions, check_quality=False):
    # This procedure is also described
    # in the MSCKF paper (Mourikis et al. 2007)
    x = np.array([img_positions[-1][0],
                  img_positions[-1][1], 0.1])
    res = opt.leastsq(self.residual, x, Dfun=self.residual_jac, args=(poses, img_positions)) # scipy opt
    #res = self.gauss_newton(self.residual, self.residual_jac, x, (poses, img_positions)) # diy gauss_newton

    alpha, beta, rho = res[0]
    rot_0_to_g = (orient.rotations_from_quats(poses[-1,3:])).dot(self.to_c.T)
    return (rot_0_to_g.dot(np.array([alpha, beta, 1])))/rho + poses[-1,:3]




'''
EXPERIMENTAL CODE
'''
def unroll_shutter(img_positions, poses, v, rot_rates, ecef_pos):
  # only speed correction for now
  t_roll = 0.016 # 16ms rolling shutter?
  vroll, vpitch, vyaw = rot_rates
  A = 0.5*np.array([[-1, -vroll, -vpitch, -vyaw],
                 [vroll, 0, vyaw, -vpitch],
                 [vpitch, -vyaw, 0, vroll],
                 [vyaw, vpitch, -vroll, 0]])
  q_dot = A.dot(poses[-1][3:7])
  v = np.append(v, q_dot)
  v = np.array([v[0], v[1], v[2],0,0,0,0])
  current_pose = poses[-1] + v*0.05
  poses = np.vstack((current_pose, poses))
  dt = -img_positions[:,1]*t_roll/0.48
  errs = project(poses, ecef_pos) - project(poses + np.atleast_2d(dt).T.dot(np.atleast_2d(v)), ecef_pos)
  return img_positions - errs

def project(poses, ecef_pos):
  img_positions = np.zeros((len(poses), 2))
  for i, p in enumerate(poses):
    cam_frame = orient.rotations_from_quats(p[3:]).T.dot(ecef_pos - p[:3])
    img_positions[i] = np.array([cam_frame[1]/cam_frame[0], cam_frame[2]/cam_frame[0]])
  return img_positions

