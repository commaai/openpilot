import numpy as np
import sympy as sp
import os

from selfdrive.locationd.kalman.kalman_helpers import ObservationKind
from selfdrive.locationd.kalman.ekf_sym import gen_code


def gen_model(name, dim_state):

  # check if rebuild is needed
  try:
    dir_path = os.path.dirname(__file__)
    deps = [dir_path + '/' + 'ekf_c.c',
            dir_path + '/' + 'ekf_sym.py',
            dir_path + '/' + 'loc_local_model.py',
            dir_path + '/' + 'loc_local_kf.py']

    outs = [dir_path + '/' + name + '.o',
            dir_path + '/' + name + '.so',
            dir_path + '/' + name + '.cpp']
    out_times = map(os.path.getmtime, outs)
    dep_times = map(os.path.getmtime, deps)
    rebuild = os.getenv("REBUILD", False)
    if min(out_times) > max(dep_times) and not rebuild:
      return
    map(os.remove, outs)
  except OSError:
    pass

  # make functions and jacobians with sympy
  # state variables
  state_sym = sp.MatrixSymbol('state', dim_state, 1)
  state = sp.Matrix(state_sym)
  v = state[0:3,:]
  omega = state[3:6,:]
  vroll, vpitch, vyaw = omega
  vx, vy, vz = v
  roll_bias, pitch_bias, yaw_bias = state[6:9,:]
  odo_scale = state[9,:]
  accel = state[10:13,:]

  dt = sp.Symbol('dt')

  # Time derivative of the state as a function of state
  state_dot = sp.Matrix(np.zeros((dim_state, 1)))
  state_dot[:3,:] = accel

  # Basic descretization, 1st order intergrator
  # Can be pretty bad if dt is big
  f_sym = sp.Matrix(state + dt*state_dot)

  #
  # Observation functions
  #

  # extra args
  #imu_rot = euler_rotate(*imu_angles)
  #h_gyro_sym = imu_rot*sp.Matrix([vroll + roll_bias,
  #                               vpitch + pitch_bias,
  #                               vyaw + yaw_bias])
  h_gyro_sym = sp.Matrix([vroll + roll_bias,
                          vpitch + pitch_bias,
                          vyaw + yaw_bias])

  speed = vx**2 + vy**2 + vz**2
  h_speed_sym = sp.Matrix([sp.sqrt(speed)*odo_scale])

  h_relative_motion = sp.Matrix(v)
  h_phone_rot_sym = sp.Matrix([vroll,
                               vpitch,
                               vyaw])


  obs_eqs = [[h_speed_sym, ObservationKind.ODOMETRIC_SPEED, None],
             [h_gyro_sym, ObservationKind.PHONE_GYRO, None],
             [h_phone_rot_sym, ObservationKind.NO_ROT, None],
             [h_relative_motion, ObservationKind.CAMERA_ODO_TRANSLATION, None],
             [h_phone_rot_sym, ObservationKind.CAMERA_ODO_ROTATION, None]]
  gen_code(name, f_sym, dt, state_sym, obs_eqs, dim_state, dim_state)
