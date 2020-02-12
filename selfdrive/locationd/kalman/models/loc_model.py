import numpy as np
import sympy as sp
import os

from laika.constants import EARTH_GM
from selfdrive.locationd.kalman.helpers import ObservationKind
from selfdrive.locationd.kalman.helpers.ekf_sym import gen_code
from common.sympy_helpers import cross, euler_rotate, quat_rotate, quat_matrix_l, quat_matrix_r

def gen_model(name, N, dim_main, dim_main_err,
                 dim_augment, dim_augment_err,
                 dim_state, dim_state_err,
                 maha_test_kinds):


  # check if rebuild is needed
  try:
    dir_path = os.path.dirname(__file__)
    deps = [dir_path + '/' + 'ekf_c.c',
            dir_path + '/' + 'ekf_sym.py',
            dir_path + '/' + 'loc_model.py',
            dir_path + '/' + 'loc_kf.py']

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
  q = state[3:7,:]
  v = state[7:10,:]
  vx, vy, vz = v
  omega = state[10:13,:]
  vroll, vpitch, vyaw = omega
  cb, cd = state[13:15,:]
  roll_bias, pitch_bias, yaw_bias = state[15:18,:]
  odo_scale = state[18,:]
  acceleration = state[19:22,:]
  focal_scale = state[22,:]
  imu_angles= state[23:26,:]
  glonass_bias, glonass_freq_slope = state[26:28,:]
  ca = state[28,0]

  dt = sp.Symbol('dt')

  # calibration and attitude rotation matrices
  quat_rot = quat_rotate(*q)

  # Got the quat predict equations from here
  # A New Quaternion-Based Kalman Filter for
  # Real-Time Attitude Estimation Using the Two-Step
  # Geometrically-Intuitive Correction Algorithm
  A = 0.5*sp.Matrix([[0, -vroll, -vpitch, -vyaw],
                 [vroll, 0, vyaw, -vpitch],
                 [vpitch, -vyaw, 0, vroll],
                 [vyaw, vpitch, -vroll, 0]])
  q_dot = A * q

  # Time derivative of the state as a function of state
  state_dot = sp.Matrix(np.zeros((dim_state, 1)))
  state_dot[:3,:] = v
  state_dot[3:7,:] = q_dot
  state_dot[7:10,0] = quat_rot * acceleration
  state_dot[13,0] = cd
  state_dot[14,0] = ca

  # Basic descretization, 1st order intergrator
  # Can be pretty bad if dt is big
  f_sym = state + dt*state_dot

  state_err_sym = sp.MatrixSymbol('state_err',dim_state_err,1)
  state_err = sp.Matrix(state_err_sym)
  quat_err = state_err[3:6,:]
  v_err = state_err[6:9,:]
  omega_err = state_err[9:12,:]
  cd_err = state_err[13,:]
  acceleration_err = state_err[18:21,:]
  ca_err = state_err[27,:]

  # Time derivative of the state error as a function of state error and state
  quat_err_matrix = euler_rotate(quat_err[0], quat_err[1], quat_err[2])
  q_err_dot = quat_err_matrix * quat_rot * (omega + omega_err)
  state_err_dot = sp.Matrix(np.zeros((dim_state_err, 1)))
  state_err_dot[:3,:] = v_err
  state_err_dot[3:6,:] = q_err_dot
  state_err_dot[6:9,:] = quat_err_matrix * quat_rot * (acceleration + acceleration_err)
  state_err_dot[12,:] = cd_err
  state_err_dot[13,:] = ca_err
  f_err_sym = state_err + dt*state_err_dot

  # convenient indexing
  # q idxs are for quats and p idxs are for other
  q_idxs = [[3, dim_augment]] + [[dim_main + n*dim_augment + 3, dim_main + (n+1)*dim_augment] for n in range(N)]
  q_err_idxs = [[3, dim_augment_err]] + [[dim_main_err + n*dim_augment_err + 3, dim_main_err + (n+1)*dim_augment_err] for n in range(N)]
  p_idxs = [[0, 3]] + [[dim_augment, dim_main]] + [[dim_main + n*dim_augment , dim_main + n*dim_augment + 3] for n in range(N)]
  p_err_idxs = [[0, 3]] + [[dim_augment_err, dim_main_err]] + [[dim_main_err + n*dim_augment_err, dim_main_err + n*dim_augment_err + 3] for n in range(N)]

  # Observation matrix modifier
  H_mod_sym = sp.Matrix(np.zeros((dim_state, dim_state_err)))
  for p_idx, p_err_idx in zip(p_idxs, p_err_idxs):
    H_mod_sym[p_idx[0]:p_idx[1],p_err_idx[0]:p_err_idx[1]] = np.eye(p_idx[1]-p_idx[0])
  for q_idx, q_err_idx in zip(q_idxs, q_err_idxs):
    H_mod_sym[q_idx[0]:q_idx[1],q_err_idx[0]:q_err_idx[1]] = 0.5*quat_matrix_r(state[q_idx[0]:q_idx[1]])[:,1:]


  # these error functions are defined so that say there
  # is a nominal x and true x:
  # true x = err_function(nominal x, delta x)
  # delta x = inv_err_function(nominal x, true x)
  nom_x = sp.MatrixSymbol('nom_x',dim_state,1)
  true_x = sp.MatrixSymbol('true_x',dim_state,1)
  delta_x = sp.MatrixSymbol('delta_x',dim_state_err,1)

  err_function_sym = sp.Matrix(np.zeros((dim_state,1)))
  for q_idx, q_err_idx in zip(q_idxs, q_err_idxs):
    delta_quat = sp.Matrix(np.ones((4)))
    delta_quat[1:,:] = sp.Matrix(0.5*delta_x[q_err_idx[0]: q_err_idx[1],:])
    err_function_sym[q_idx[0]:q_idx[1],0] = quat_matrix_r(nom_x[q_idx[0]:q_idx[1],0])*delta_quat
  for p_idx, p_err_idx in zip(p_idxs, p_err_idxs):
    err_function_sym[p_idx[0]:p_idx[1],:] = sp.Matrix(nom_x[p_idx[0]:p_idx[1],:] + delta_x[p_err_idx[0]:p_err_idx[1],:])

  inv_err_function_sym = sp.Matrix(np.zeros((dim_state_err,1)))
  for p_idx, p_err_idx in zip(p_idxs, p_err_idxs):
    inv_err_function_sym[p_err_idx[0]:p_err_idx[1],0] = sp.Matrix(-nom_x[p_idx[0]:p_idx[1],0] + true_x[p_idx[0]:p_idx[1],0])
  for q_idx, q_err_idx in zip(q_idxs, q_err_idxs):
    delta_quat = quat_matrix_r(nom_x[q_idx[0]:q_idx[1],0]).T*true_x[q_idx[0]:q_idx[1],0]
    inv_err_function_sym[q_err_idx[0]:q_err_idx[1],0] = sp.Matrix(2*delta_quat[1:])

  eskf_params = [[err_function_sym, nom_x, delta_x],
                 [inv_err_function_sym, nom_x, true_x],
                 H_mod_sym, f_err_sym, state_err_sym]



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

  imu_rot = euler_rotate(*imu_angles)
  h_gyro_sym = imu_rot*sp.Matrix([vroll + roll_bias,
                                 vpitch + pitch_bias,
                                 vyaw + yaw_bias])

  pos = sp.Matrix([x, y, z])
  gravity = quat_rot.T * ((EARTH_GM/((x**2 + y**2 + z**2)**(3.0/2.0)))*pos)
  h_acc_sym = imu_rot*(gravity + acceleration)
  h_phone_rot_sym = sp.Matrix([vroll,
                               vpitch,
                               vyaw])
  speed = vx**2 + vy**2 + vz**2
  h_speed_sym = sp.Matrix([sp.sqrt(speed)*odo_scale])

  # orb stuff
  orb_pos_sym = sp.Matrix([orb_x - x, orb_y - y, orb_z - z])
  orb_pos_rot_sym = quat_rot.T * orb_pos_sym
  s = orb_pos_rot_sym[0]
  h_orb_point_sym = sp.Matrix([(1/s)*(orb_pos_rot_sym[1]),
                               (1/s)*(orb_pos_rot_sym[2])])

  h_pos_sym = sp.Matrix([x, y, z])
  h_imu_frame_sym = sp.Matrix(imu_angles)

  h_relative_motion = sp.Matrix(quat_rot.T * v)


  obs_eqs = [[h_speed_sym, ObservationKind.ODOMETRIC_SPEED, None],
             [h_gyro_sym, ObservationKind.PHONE_GYRO, None],
             [h_phone_rot_sym, ObservationKind.NO_ROT, None],
             [h_acc_sym, ObservationKind.PHONE_ACCEL, None],
             [h_pseudorange_sym, ObservationKind.PSEUDORANGE_GPS, sat_pos_freq_sym],
             [h_pseudorange_glonass_sym, ObservationKind.PSEUDORANGE_GLONASS, sat_pos_freq_sym],
             [h_pseudorange_rate_sym, ObservationKind.PSEUDORANGE_RATE_GPS, sat_pos_vel_sym],
             [h_pseudorange_rate_sym, ObservationKind.PSEUDORANGE_RATE_GLONASS, sat_pos_vel_sym],
             [h_pos_sym, ObservationKind.ECEF_POS, None],
             [h_relative_motion, ObservationKind.CAMERA_ODO_TRANSLATION, None],
             [h_phone_rot_sym, ObservationKind.CAMERA_ODO_ROTATION, None],
             [h_imu_frame_sym, ObservationKind.IMU_FRAME, None],
             [h_orb_point_sym, ObservationKind.ORB_POINT, orb_epos_sym]]

  # MSCKF configuration
  if N > 0:
    focal_scale =1
    # Add observation functions for orb feature tracks
    track_epos_sym = sp.MatrixSymbol('track_epos_sym', 3, 1)
    track_x, track_y, track_z = track_epos_sym
    h_track_sym = sp.Matrix(np.zeros(((1 + N)*2, 1)))
    track_pos_sym = sp.Matrix([track_x - x, track_y - y, track_z - z])
    track_pos_rot_sym = quat_rot.T * track_pos_sym
    h_track_sym[-2:,:] = sp.Matrix([focal_scale*(track_pos_rot_sym[1]/track_pos_rot_sym[0]),
                                     focal_scale*(track_pos_rot_sym[2]/track_pos_rot_sym[0])])

    h_msckf_test_sym = sp.Matrix(np.zeros(((1 + N)*3, 1)))
    h_msckf_test_sym[-3:,:] = sp.Matrix([track_x - x,track_y - y , track_z - z])

    for n in range(N):
      idx = dim_main + n*dim_augment
      err_idx = dim_main_err + n*dim_augment_err
      x, y, z = state[idx:idx+3]
      q = state[idx+3:idx+7]
      quat_rot = quat_rotate(*q)
      track_pos_sym = sp.Matrix([track_x - x, track_y - y, track_z - z])
      track_pos_rot_sym = quat_rot.T * track_pos_sym
      h_track_sym[n*2:n*2+2,:] = sp.Matrix([focal_scale*(track_pos_rot_sym[1]/track_pos_rot_sym[0]),
                                             focal_scale*(track_pos_rot_sym[2]/track_pos_rot_sym[0])])
      h_msckf_test_sym[n*3:n*3+3,:] = sp.Matrix([track_x - x, track_y - y, track_z - z])
    obs_eqs.append([h_msckf_test_sym, ObservationKind.MSCKF_TEST, track_epos_sym])
    obs_eqs.append([h_track_sym, ObservationKind.ORB_FEATURES, track_epos_sym])
    obs_eqs.append([h_track_sym, ObservationKind.FEATURE_TRACK_TEST, track_epos_sym])
    msckf_params = [dim_main, dim_augment, dim_main_err, dim_augment_err, N, [ObservationKind.MSCKF_TEST, ObservationKind.ORB_FEATURES]]
  else:
    msckf_params = None
  gen_code(name, f_sym, dt, state_sym, obs_eqs, dim_state, dim_state_err, eskf_params, msckf_params, maha_test_kinds)
