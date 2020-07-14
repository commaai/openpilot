#!/usr/bin/env python3
import sympy as sp
import numpy as np

# TODO: remove code duplication between openpilot.common.orientation
def quat2rot(quats):
  quats = np.array(quats)
  input_shape = quats.shape
  quats = np.atleast_2d(quats)
  Rs = np.zeros((quats.shape[0], 3, 3))
  q0 = quats[:, 0]
  q1 = quats[:, 1]
  q2 = quats[:, 2]
  q3 = quats[:, 3]
  Rs[:, 0, 0] = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3
  Rs[:, 0, 1] = 2 * (q1 * q2 - q0 * q3)
  Rs[:, 0, 2] = 2 * (q0 * q2 + q1 * q3)
  Rs[:, 1, 0] = 2 * (q1 * q2 + q0 * q3)
  Rs[:, 1, 1] = q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3
  Rs[:, 1, 2] = 2 * (q2 * q3 - q0 * q1)
  Rs[:, 2, 0] = 2 * (q1 * q3 - q0 * q2)
  Rs[:, 2, 1] = 2 * (q0 * q1 + q2 * q3)
  Rs[:, 2, 2] = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3

  if len(input_shape) < 2:
    return Rs[0]
  else:
    return Rs


def euler2quat(eulers):
  eulers = np.array(eulers)
  if len(eulers.shape) > 1:
    output_shape = (-1,4)
  else:
    output_shape = (4,)
  eulers = np.atleast_2d(eulers)
  gamma, theta, psi = eulers[:,0],  eulers[:,1],  eulers[:,2]

  q0 = np.cos(gamma / 2) * np.cos(theta / 2) * np.cos(psi / 2) + \
       np.sin(gamma / 2) * np.sin(theta / 2) * np.sin(psi / 2)
  q1 = np.sin(gamma / 2) * np.cos(theta / 2) * np.cos(psi / 2) - \
       np.cos(gamma / 2) * np.sin(theta / 2) * np.sin(psi / 2)
  q2 = np.cos(gamma / 2) * np.sin(theta / 2) * np.cos(psi / 2) + \
       np.sin(gamma / 2) * np.cos(theta / 2) * np.sin(psi / 2)
  q3 = np.cos(gamma / 2) * np.cos(theta / 2) * np.sin(psi / 2) - \
       np.sin(gamma / 2) * np.sin(theta / 2) * np.cos(psi / 2)

  quats = np.array([q0, q1, q2, q3]).T
  for i in range(len(quats)):
    if quats[i,0] < 0:  # pylint: disable=unsubscriptable-object
      quats[i] = -quats[i]  # pylint: disable=unsupported-assignment-operation,unsubscriptable-object
  return quats.reshape(output_shape)


def euler2rot(eulers):
  return quat2rot(euler2quat(eulers))

rotations_from_quats = quat2rot


def cross(x):
  ret = sp.Matrix(np.zeros((3, 3)))
  ret[0, 1], ret[0, 2] = -x[2], x[1]
  ret[1, 0], ret[1, 2] = x[2], -x[0]
  ret[2, 0], ret[2, 1] = -x[1], x[0]
  return ret


def rot_matrix(roll, pitch, yaw):
  cr, sr = np.cos(roll), np.sin(roll)
  cp, sp = np.cos(pitch), np.sin(pitch)
  cy, sy = np.cos(yaw), np.sin(yaw)
  rr = np.array([[1,0,0],[0, cr,-sr],[0, sr, cr]])
  rp = np.array([[cp,0,sp],[0, 1,0],[-sp, 0, cp]])
  ry = np.array([[cy,-sy,0],[sy, cy,0],[0, 0, 1]])
  return ry.dot(rp.dot(rr))


def euler_rotate(roll, pitch, yaw):
  # make symbolic rotation matrix from eulers
  matrix_roll = sp.Matrix([[1, 0, 0],
                           [0, sp.cos(roll), -sp.sin(roll)],
                           [0, sp.sin(roll), sp.cos(roll)]])
  matrix_pitch = sp.Matrix([[sp.cos(pitch), 0, sp.sin(pitch)],
                            [0, 1, 0],
                            [-sp.sin(pitch), 0, sp.cos(pitch)]])
  matrix_yaw = sp.Matrix([[sp.cos(yaw), -sp.sin(yaw), 0],
                          [sp.sin(yaw), sp.cos(yaw), 0],
                          [0, 0, 1]])
  return matrix_yaw * matrix_pitch * matrix_roll


def quat_rotate(q0, q1, q2, q3):
  # make symbolic rotation matrix from quat
  return sp.Matrix([[q0**2 + q1**2 - q2**2 - q3**2, 2 * (q1 * q2 + q0 * q3), 2 * (q1 * q3 - q0 * q2)],
                    [2 * (q1 * q2 - q0 * q3), q0**2 - q1**2 + q2**2 - q3**2, 2 * (q2 * q3 + q0 * q1)],
                    [2 * (q1 * q3 + q0 * q2), 2 * (q2 * q3 - q0 * q1), q0**2 - q1**2 - q2**2 + q3**2]]).T


def quat_matrix_l(p):
  return sp.Matrix([[p[0], -p[1], -p[2], -p[3]],
                    [p[1],  p[0], -p[3],  p[2]],
                    [p[2],  p[3],  p[0], -p[1]],
                    [p[3], -p[2],  p[1],  p[0]]])


def quat_matrix_r(p):
  return sp.Matrix([[p[0], -p[1], -p[2], -p[3]],
                    [p[1],  p[0],  p[3], -p[2]],
                    [p[2], -p[3],  p[0],  p[1]],
                    [p[3],  p[2], -p[1],  p[0]]])


def sympy_into_c(sympy_functions, global_vars=None):
  from sympy.utilities import codegen
  routines = []
  for name, expr, args in sympy_functions:
    r = codegen.make_routine(name, expr, language="C99", global_vars=global_vars)

    # argument ordering input to sympy is broken with function with output arguments
    nargs = []

    # reorder the input arguments
    for aa in args:
      if aa is None:
        nargs.append(codegen.InputArgument(sp.Symbol('unused'), dimensions=[1, 1]))
        continue
      found = False
      for a in r.arguments:
        if str(aa.name) == str(a.name):
          nargs.append(a)
          found = True
          break
      if not found:
        # [1,1] is a hack for Matrices
        nargs.append(codegen.InputArgument(aa, dimensions=[1, 1]))

    # add the output arguments
    for a in r.arguments:
      if type(a) == codegen.OutputArgument:
        nargs.append(a)

    # assert len(r.arguments) == len(args)+1
    r.arguments = nargs

    # add routine to list
    routines.append(r)

  [(_, c_code), (_, c_header)] = codegen.get_code_generator('C', 'ekf', 'C99').write(routines, "ekf")
  c_header = '\n'.join(x for x in c_header.split("\n") if len(x) > 0 and x[0] != '#')

  c_code = '\n'.join(x for x in c_code.split("\n") if len(x) > 0 and x[0] != '#')
  c_code = 'extern "C" {\n#include <math.h>\n' + c_code + "\n}\n"

  return c_header, c_code
