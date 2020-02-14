#!/usr/bin/env python3
import os
import sys

import numpy as np
import sympy as sp

import common.transformations.orientation as orient
from selfdrive.locationd.kalman.helpers import (TEMPLATE_DIR, load_code,
                                                write_code)
from selfdrive.locationd.kalman.helpers.sympy_helpers import (quat_rotate,
                                                              sympy_into_c)


def generate_residual(K):
  x_sym = sp.MatrixSymbol('abr', 3, 1)
  poses_sym = sp.MatrixSymbol('poses', 7 * K, 1)
  img_pos_sym = sp.MatrixSymbol('img_positions', 2 * K, 1)
  alpha, beta, rho = x_sym
  to_c = sp.Matrix(orient.rot_matrix(-np.pi / 2, -np.pi / 2, 0))
  pos_0 = sp.Matrix(np.array(poses_sym[K * 7 - 7:K * 7 - 4])[:, 0])
  q = poses_sym[K * 7 - 4:K * 7]
  quat_rot = quat_rotate(*q)
  rot_g_to_0 = to_c * quat_rot.T
  rows = []

  for i in range(K):
    pos_i = sp.Matrix(np.array(poses_sym[i * 7:i * 7 + 3])[:, 0])
    q = poses_sym[7 * i + 3:7 * i + 7]
    quat_rot = quat_rotate(*q)
    rot_g_to_i = to_c * quat_rot.T
    rot_0_to_i = rot_g_to_i * rot_g_to_0.T
    trans_0_to_i = rot_g_to_i * (pos_0 - pos_i)
    funct_vec = rot_0_to_i * sp.Matrix([alpha, beta, 1]) + rho * trans_0_to_i
    h1, h2, h3 = funct_vec
    rows.append(h1 / h3 - img_pos_sym[i * 2 + 0])
    rows.append(h2 / h3 - img_pos_sym[i * 2 + 1])
  img_pos_residual_sym = sp.Matrix(rows)

  # sympy into c
  sympy_functions = []
  sympy_functions.append(('res_fun', img_pos_residual_sym, [x_sym, poses_sym, img_pos_sym]))
  sympy_functions.append(('jac_fun', img_pos_residual_sym.jacobian(x_sym), [x_sym, poses_sym, img_pos_sym]))

  return sympy_functions


class LstSqComputer():
  name = 'pos_computer'

  @staticmethod
  def generate_code(K=4):
    sympy_functions = generate_residual(K)
    header, code = sympy_into_c(sympy_functions)

    code += "\n#define KDIM %d\n" % K
    code += "\n" + open(os.path.join(TEMPLATE_DIR, "compute_pos.c")).read()

    header += """
    void compute_pos(double *to_c, double *in_poses, double *in_img_positions, double *param, double *pos);
    """

    filename = f"{LstSqComputer.name}_{K}"
    write_code(filename, code, header)

  def __init__(self, K=4, MIN_DEPTH=2, MAX_DEPTH=500):
    self.to_c = orient.rot_matrix(-np.pi / 2, -np.pi / 2, 0)
    self.MAX_DEPTH = MAX_DEPTH
    self.MIN_DEPTH = MIN_DEPTH

    name = f"{LstSqComputer.name}_{K}"
    ffi, lib = load_code(name)

    # wrap c functions
    def residual_jac(x, poses, img_positions):
      out = np.zeros(((K * 2, 3)), dtype=np.float64)
      lib.jac_fun(ffi.cast("double *", x.ctypes.data),
                  ffi.cast("double *", poses.ctypes.data),
                  ffi.cast("double *", img_positions.ctypes.data),
                  ffi.cast("double *", out.ctypes.data))
      return out
    self.residual_jac = residual_jac

    def residual(x, poses, img_positions):
      out = np.zeros((K * 2), dtype=np.float64)
      lib.res_fun(ffi.cast("double *", x.ctypes.data),
                  ffi.cast("double *", poses.ctypes.data),
                  ffi.cast("double *", img_positions.ctypes.data),
                  ffi.cast("double *", out.ctypes.data))
      return out
    self.residual = residual

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
    pos, param = self.compute_pos_c(poses, img_positions)
    # pos, param = self.compute_pos_python(poses, img_positions)

    depth = 1 / param[2]
    if debug:
      if not self.debug:
        raise NotImplementedError("This is not a debug computer")

      # orient_err_jac = self.orient_error_jac(param, poses, img_positions, np.zeros(3)).reshape((-1,2,3))
      jac = self.residual_jac(param, poses, img_positions).reshape((-1, 2, 3))
      res = self.residual(param, poses, img_positions).reshape((-1, 2))
      return pos, param, res, jac  # , orient_err_jac
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
    import scipy.optimize as opt

    # This procedure is also described
    # in the MSCKF paper (Mourikis et al. 2007)
    x = np.array([img_positions[-1][0],
                  img_positions[-1][1], 0.1])
    res = opt.leastsq(self.residual, x, Dfun=self.residual_jac, args=(poses, img_positions))  # scipy opt
    # res = self.gauss_newton(self.residual, self.residual_jac, x, (poses, img_positions)) # diy gauss_newton

    alpha, beta, rho = res[0]
    rot_0_to_g = (orient.rotations_from_quats(poses[-1, 3:])).dot(self.to_c.T)
    return (rot_0_to_g.dot(np.array([alpha, beta, 1]))) / rho + poses[-1, :3]


# EXPERIMENTAL CODE
def unroll_shutter(img_positions, poses, v, rot_rates, ecef_pos):
  # only speed correction for now
  t_roll = 0.016  # 16ms rolling shutter?
  vroll, vpitch, vyaw = rot_rates
  A = 0.5 * np.array([[-1, -vroll, -vpitch, -vyaw],
                      [vroll, 0, vyaw, -vpitch],
                      [vpitch, -vyaw, 0, vroll],
                      [vyaw, vpitch, -vroll, 0]])
  q_dot = A.dot(poses[-1][3:7])
  v = np.append(v, q_dot)
  v = np.array([v[0], v[1], v[2], 0, 0, 0, 0])
  current_pose = poses[-1] + v * 0.05
  poses = np.vstack((current_pose, poses))
  dt = -img_positions[:, 1] * t_roll / 0.48
  errs = project(poses, ecef_pos) - project(poses + np.atleast_2d(dt).T.dot(np.atleast_2d(v)), ecef_pos)
  return img_positions - errs


def project(poses, ecef_pos):
  img_positions = np.zeros((len(poses), 2))
  for i, p in enumerate(poses):
    cam_frame = orient.rotations_from_quats(p[3:]).T.dot(ecef_pos - p[:3])
    img_positions[i] = np.array([cam_frame[1] / cam_frame[0], cam_frame[2] / cam_frame[0]])
  return img_positions


if __name__ == "__main__":
  K = int(sys.argv[1].split("_")[-1])
  LstSqComputer.generate_code(K=K)

