import numpy as np
import sympy as sp

import os
from selfdrive.locationd.kalman.helpers import ObservationKind
from selfdrive.locationd.kalman.helpers.ekf_sym import gen_code
from common.sympy_helpers import cross, euler_rotate, quat_rotate, quat_matrix_l, quat_matrix_r

def gen_model(name, dim_state, maha_test_kinds):

  # check if rebuild is needed
  try:
    dir_path = os.path.dirname(__file__)
    deps = [dir_path + '/' + 'ekf_c.c',
            dir_path + '/' + 'ekf_sym.py',
            dir_path + '/' + 'gnss_model.py',
            dir_path + '/' + 'gnss_kf.py']

    outs = [dir_path + '/' + name + '.o',
            dir_path + '/' + name + '.so',
            dir_path + '/' + name + '.cpp']
    out_times = list(map(os.path.getmtime, outs))
    dep_times = list(map(os.path.getmtime, deps))
    rebuild = os.getenv("REBUILD", False)
    if min(out_times) > max(dep_times) and not rebuild:
      return
    list(map(os.remove, outs))
  except OSError as e:
    pass

  # make functions and jacobians with sympy
  # state variables
  state_sym = sp.MatrixSymbol('state', dim_state, 1)
  state = sp.Matrix(state_sym)
  x,y,z = state[0:3,:]
  v = state[3:6,:]
  vx, vy, vz = v
  cb, cd, ca = state[6:9,:]
  glonass_bias, glonass_freq_slope = state[9:11,:]

  dt = sp.Symbol('dt')

  state_dot = sp.Matrix(np.zeros((dim_state, 1)))
  state_dot[:3,:] = v
  state_dot[6,0] = cd
  state_dot[7,0] = ca

  # Basic descretization, 1st order intergrator
  # Can be pretty bad if dt is big
  f_sym = state + dt*state_dot


  #
  # Observation functions
  #

  # extra args
  sat_pos_freq_sym = sp.MatrixSymbol('sat_pos', 4, 1)
  sat_pos_vel_sym = sp.MatrixSymbol('sat_pos_vel', 6, 1)
  sat_los_sym = sp.MatrixSymbol('sat_los', 3, 1)
  orb_epos_sym = sp.MatrixSymbol('orb_epos_sym', 3, 1)

  # expand extra args
  sat_x, sat_y, sat_z, glonass_freq = sat_pos_freq_sym
  sat_vx, sat_vy, sat_vz = sat_pos_vel_sym[3:]
  los_x, los_y, los_z = sat_los_sym
  orb_x, orb_y, orb_z = orb_epos_sym

  h_pseudorange_sym = sp.Matrix([sp.sqrt(
                                  (x - sat_x)**2 +
                                  (y - sat_y)**2 +
                                  (z - sat_z)**2) +
                                  cb])

  h_pseudorange_glonass_sym = sp.Matrix([sp.sqrt(
                                  (x - sat_x)**2 +
                                  (y - sat_y)**2 +
                                  (z - sat_z)**2) +
                                  cb + glonass_bias + glonass_freq_slope*glonass_freq])

  los_vector = (sp.Matrix(sat_pos_vel_sym[0:3]) - sp.Matrix([x, y, z]))
  los_vector = los_vector / sp.sqrt(los_vector[0]**2 + los_vector[1]**2 + los_vector[2]**2)
  h_pseudorange_rate_sym = sp.Matrix([los_vector[0]*(sat_vx - vx) +
                                         los_vector[1]*(sat_vy - vy) +
                                         los_vector[2]*(sat_vz - vz) +
                                         cd])

  obs_eqs = [[h_pseudorange_sym, ObservationKind.PSEUDORANGE_GPS, sat_pos_freq_sym],
             [h_pseudorange_glonass_sym, ObservationKind.PSEUDORANGE_GLONASS, sat_pos_freq_sym],
             [h_pseudorange_rate_sym, ObservationKind.PSEUDORANGE_RATE_GPS, sat_pos_vel_sym],
             [h_pseudorange_rate_sym, ObservationKind.PSEUDORANGE_RATE_GLONASS, sat_pos_vel_sym]]

  gen_code(name, f_sym, dt, state_sym, obs_eqs, dim_state, dim_state, maha_test_kinds=maha_test_kinds)
