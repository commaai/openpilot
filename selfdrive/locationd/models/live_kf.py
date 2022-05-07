#!/usr/bin/env python3

import sys
import os
import numpy as np

from selfdrive.locationd.models.constants import ObservationKind

import sympy as sp
import inspect
from rednose.helpers.sympy_helpers import euler_rotate, quat_matrix_r, quat_rotate
from rednose.helpers.ekf_sym import gen_code

EARTH_GM = 3.986005e14  # m^3/s^2 (gravitational constant * mass of earth)


def numpy2eigenstring(arr):
  assert(len(arr.shape) == 1)
  arr_str = np.array2string(arr, precision=20, separator=',')[1:-1].replace(' ', '').replace('\n', '')
  return f"(Eigen::VectorXd({len(arr)}) << {arr_str}).finished()"


class States:
  ECEF_POS = slice(0, 3)  # x, y and z in ECEF in meters
  ECEF_ORIENTATION = slice(3, 7)  # quat for pose of phone in ecef
  ECEF_VELOCITY = slice(7, 10)  # ecef velocity in m/s
  ANGULAR_VELOCITY = slice(10, 13)  # roll, pitch and yaw rates in device frame in radians/s
  GYRO_BIAS = slice(13, 16)  # roll, pitch and yaw biases
  ACCELERATION = slice(16, 19)  # Acceleration in device frame in m/s**2
  ACC_BIAS = slice(19, 22)  # Acceletometer bias in m/s**2
  CLOCK_BIAS = slice(22, 23)  # clock bias in light-meters,
  CLOCK_DRIFT = slice(23, 24)  # clock drift in light-meters/s,
  GLONASS_BIAS = slice(24, 25)  # GLONASS bias in m expressed as bias + freq_num*freq_slope
  GLONASS_FREQ_SLOPE = slice(25, 26)  # GLONASS bias in m expressed as bias + freq_num*freq_slope
  CLOCK_ACCELERATION = slice(26, 27)  # clock acceleration in light-meters/s**2,

  # Error-state has different slices because it is an ESKF
  ECEF_POS_ERR = slice(0, 3)
  ECEF_ORIENTATION_ERR = slice(3, 6)  # euler angles for orientation error
  ECEF_VELOCITY_ERR = slice(6, 9)
  ANGULAR_VELOCITY_ERR = slice(9, 12)
  GYRO_BIAS_ERR = slice(12, 15)
  ACCELERATION_ERR = slice(15, 18)
  ACC_BIAS_ERR = slice(18, 21)
  CLOCK_BIAS_ERR = slice(21, 22)
  CLOCK_DRIFT_ERR = slice(22, 23)
  GLONASS_BIAS_ERR = slice(23, 24)
  GLONASS_FREQ_SLOPE_ERR = slice(24, 25)
  CLOCK_ACCELERATION_ERR = slice(25, 26)

# todo could change name to LiveLocationKalman to match usage in locationd
class LiveKalman:
  name = 'live'

  initial_x = np.array([3.88e6, -3.37e6, 3.76e6,
                        0.42254641, -0.31238054, -0.83602975, -0.15788347,  # NED [0,0,0] -> ECEF Quat
                        0, 0, 0,
                        0, 0, 0,
                        0, 0, 0,
                        0, 0, 0,
                        0, 0, 0,
                        0, 0,     # 22,23
                        0, 0,     # 24
                        0])       # 25


  # state covariance
  initial_P_diag = np.array([10**2, 10**2, 10**2,
                             0.01**2, 0.01**2, 0.01**2,
                             10**2, 10**2, 10**2,
                             1**2, 1**2, 1**2,
                             1**2, 1**2, 1**2,
                             100**2, 100**2, 100**2,
                             0.01**2, 0.01**2, 0.01**2,
                             200000**2, 100**2,
                             10**2, 1**2,
                             0.05**2])

  # state covariance when resetting midway in a segment
  reset_orientation_diag = np.array([1**2, 1**2, 1**2])

  # fake observation covariance, to ensure the uncertainty estimate of the filter is under control
  fake_gps_pos_cov_diag = np.array([1000**2, 1000**2, 1000**2])
  fake_gps_vel_cov_diag = np.array([10**2, 10**2, 10**2])

  # process noise
  Q_diag = np.array([0.03**2, 0.03**2, 0.03**2,
                     0.001**2, 0.001**2, 0.001**2,
                     0.01**2, 0.01**2, 0.01**2,
                     0.1**2, 0.1**2, 0.1**2,
                     (0.005 / 100)**2, (0.005 / 100)**2, (0.005 / 100)**2,
                     3**2, 3**2, 3**2,
                     0.005**2, 0.005**2, 0.005**2,
                     .1**2, 0.0**2,
                     .1**2, .01**2,
                     0.005**2])

  obs_noise_diag = {ObservationKind.PHONE_GYRO: np.array([0.025**2, 0.025**2, 0.025**2]),
                    ObservationKind.PHONE_ACCEL: np.array([.5**2, .5**2, .5**2]),
                    ObservationKind.CAMERA_ODO_ROTATION: np.array([0.05**2, 0.05**2, 0.05**2]),
                    ObservationKind.NO_ROT: np.array([0.005**2, 0.005**2, 0.005**2]),
                    ObservationKind.NO_ACCEL: np.array([0.05**2, 0.05**2, 0.05**2]),
                    ObservationKind.ECEF_POS: np.array([5**2, 5**2, 5**2]),
                    ObservationKind.ECEF_VEL: np.array([.5**2, .5**2, .5**2]),
                    ObservationKind.ECEF_ORIENTATION_FROM_GPS: np.array([.2**2, .2**2, .2**2, .2**2])}

  @staticmethod
  def generate_code(generated_dir):
    name = LiveKalman.name
    dim_state = LiveKalman.initial_x.shape[0]
    dim_state_err = LiveKalman.initial_P_diag.shape[0]

    state_sym = sp.MatrixSymbol('state', dim_state, 1)
    state = sp.Matrix(state_sym)
    x, y, z = state[States.ECEF_POS, :]
    q = state[States.ECEF_ORIENTATION, :]
    v = state[States.ECEF_VELOCITY, :]
    vx, vy, vz = v
    omega = state[States.ANGULAR_VELOCITY, :]
    vroll, vpitch, vyaw = omega
    roll_bias, pitch_bias, yaw_bias = state[States.GYRO_BIAS, :]
    acceleration = state[States.ACCELERATION, :]
    acc_bias = state[States.ACC_BIAS, :]

    cb = state[States.CLOCK_BIAS, :]
    cd = state[States.CLOCK_DRIFT, :]
    glonass_bias = state[States.GLONASS_BIAS, :]
    glonass_freq_slope = state[States.GLONASS_FREQ_SLOPE, :]
    ca = state[States.CLOCK_ACCELERATION, :]

    dt = sp.Symbol('dt')

    # calibration and attitude rotation matrices
    quat_rot = quat_rotate(*q)

    # Got the quat predict equations from here
    # A New Quaternion-Based Kalman Filter for
    # Real-Time Attitude Estimation Using the Two-Step
    # Geometrically-Intuitive Correction Algorithm
    A = 0.5 * sp.Matrix([[0, -vroll, -vpitch, -vyaw],
                         [vroll, 0, vyaw, -vpitch],
                         [vpitch, -vyaw, 0, vroll],
                         [vyaw, vpitch, -vroll, 0]])
    q_dot = A * q

    # Time derivative of the state as a function of state
    state_dot = sp.Matrix(np.zeros((dim_state, 1)))
    state_dot[States.ECEF_POS, :] = v
    state_dot[States.ECEF_ORIENTATION, :] = q_dot
    state_dot[States.ECEF_VELOCITY, 0] = quat_rot * acceleration
    # todo not clean
    state_dot[22, 0] = cd
    state_dot[23, 0] = ca
    #state_dot[States.CLOCK_BIAS, 0][0,0] = cd
    state_dot[States.CLOCK_DRIFT, 0][0,0] = ca

    # Basic descretization, 1st order intergrator
    # Can be pretty bad if dt is big
    f_sym = state + dt * state_dot

    state_err_sym = sp.MatrixSymbol('state_err', dim_state_err, 1)
    state_err = sp.Matrix(state_err_sym)
    quat_err = state_err[States.ECEF_ORIENTATION_ERR, :]
    v_err = state_err[States.ECEF_VELOCITY_ERR, :]
    cd_err = state_err[States.CLOCK_DRIFT_ERR, :]
    omega_err = state_err[States.ANGULAR_VELOCITY_ERR, :]
    acceleration_err = state_err[States.ACCELERATION_ERR, :]
    ca_err = state_err[States.CLOCK_ACCELERATION_ERR, :]

    # Time derivative of the state error as a function of state error and state
    quat_err_matrix = euler_rotate(quat_err[0], quat_err[1], quat_err[2])
    q_err_dot = quat_err_matrix * quat_rot * (omega + omega_err)
    state_err_dot = sp.Matrix(np.zeros((dim_state_err, 1)))
    state_err_dot[States.ECEF_POS_ERR, :] = v_err
    state_err_dot[States.ECEF_ORIENTATION_ERR, :] = q_err_dot
    state_err_dot[States.ECEF_VELOCITY_ERR, :] = quat_err_matrix * quat_rot * (acceleration + acceleration_err)
    state_err_dot[States.CLOCK_BIAS_ERR, :] = cd_err
    state_err_dot[States.CLOCK_DRIFT_ERR, :] = ca_err
    f_err_sym = state_err + dt * state_err_dot

    # Observation matrix modifier
    H_mod_sym = sp.Matrix(np.zeros((dim_state, dim_state_err)))
    H_mod_sym[States.ECEF_POS, States.ECEF_POS_ERR] = np.eye(States.ECEF_POS.stop - States.ECEF_POS.start)
    H_mod_sym[States.ECEF_ORIENTATION, States.ECEF_ORIENTATION_ERR] = 0.5 * quat_matrix_r(state[3:7])[:, 1:]
    H_mod_sym[States.ECEF_ORIENTATION.stop:, States.ECEF_ORIENTATION_ERR.stop:] = np.eye(dim_state - States.ECEF_ORIENTATION.stop)

    # these error functions are defined so that say there
    # is a nominal x and true x:
    # true x = err_function(nominal x, delta x)
    # delta x = inv_err_function(nominal x, true x)
    nom_x = sp.MatrixSymbol('nom_x', dim_state, 1)
    true_x = sp.MatrixSymbol('true_x', dim_state, 1)
    delta_x = sp.MatrixSymbol('delta_x', dim_state_err, 1)

    err_function_sym = sp.Matrix(np.zeros((dim_state, 1)))
    delta_quat = sp.Matrix(np.ones(4))
    delta_quat[1:, :] = sp.Matrix(0.5 * delta_x[States.ECEF_ORIENTATION_ERR, :])
    err_function_sym[States.ECEF_POS, :] = sp.Matrix(nom_x[States.ECEF_POS, :] + delta_x[States.ECEF_POS_ERR, :])
    err_function_sym[States.ECEF_ORIENTATION, 0] = quat_matrix_r(nom_x[States.ECEF_ORIENTATION, 0]) * delta_quat
    err_function_sym[States.ECEF_ORIENTATION.stop:, :] = sp.Matrix(nom_x[States.ECEF_ORIENTATION.stop:, :] + delta_x[States.ECEF_ORIENTATION_ERR.stop:, :])

    inv_err_function_sym = sp.Matrix(np.zeros((dim_state_err, 1)))
    inv_err_function_sym[States.ECEF_POS_ERR, 0] = sp.Matrix(-nom_x[States.ECEF_POS, 0] + true_x[States.ECEF_POS, 0])
    delta_quat = quat_matrix_r(nom_x[States.ECEF_ORIENTATION, 0]).T * true_x[States.ECEF_ORIENTATION, 0]
    inv_err_function_sym[States.ECEF_ORIENTATION_ERR, 0] = sp.Matrix(2 * delta_quat[1:])
    inv_err_function_sym[States.ECEF_ORIENTATION_ERR.stop:, 0] = sp.Matrix(-nom_x[States.ECEF_ORIENTATION.stop:, 0] + true_x[States.ECEF_ORIENTATION.stop:, 0])

    eskf_params = [[err_function_sym, nom_x, delta_x],
                   [inv_err_function_sym, nom_x, true_x],
                   H_mod_sym, f_err_sym, state_err_sym]
    #
    # Observation functions
    #

    # extra args
    sat_pos_freq_sym = sp.MatrixSymbol('sat_pos', 4, 1)
    sat_pos_vel_sym = sp.MatrixSymbol('sat_pos_vel', 6, 1)
    # sat_los_sym = sp.MatrixSymbol('sat_los', 3, 1) # todo should fix?

    # expand extra args
    sat_x, sat_y, sat_z, glonass_freq = sat_pos_freq_sym
    sat_vx, sat_vy, sat_vz = sat_pos_vel_sym[3:]

    h_pseudorange_sym = sp.Matrix([
      sp.sqrt(
        (x - sat_x) ** 2 +
        (y - sat_y) ** 2 +
        (z - sat_z) ** 2
      ) + cb[0]
    ])

    h_pseudorange_glonass_sym = sp.Matrix([
      sp.sqrt(
        (x - sat_x) ** 2 +
        (y - sat_y) ** 2 +
        (z - sat_z) ** 2
      ) + cb[0] + glonass_bias[0] + glonass_freq_slope[0] * glonass_freq
    ])

    los_vector = (sp.Matrix(sat_pos_vel_sym[:3]) - sp.Matrix([x, y, z]))
    los_vector = los_vector / sp.sqrt(los_vector[0] ** 2 + los_vector[1] ** 2 + los_vector[2] ** 2)
    h_pseudorange_rate_sym = sp.Matrix([los_vector[0] * (sat_vx - vx) +
                                        los_vector[1] * (sat_vy - vy) +
                                        los_vector[2] * (sat_vz - vz) +
                                        cd[0]])

    h_gyro_sym = sp.Matrix([
      vroll + roll_bias,
      vpitch + pitch_bias,
      vyaw + yaw_bias])

    pos = sp.Matrix([x, y, z])
    gravity = quat_rot.T * ((EARTH_GM / ((x**2 + y**2 + z**2)**(3.0 / 2.0))) * pos)
    h_acc_sym = (gravity + acceleration + acc_bias)
    h_acc_stationary_sym = acceleration
    h_phone_rot_sym = sp.Matrix([vroll, vpitch, vyaw])
    h_pos_sym = sp.Matrix([x, y, z])
    h_vel_sym = sp.Matrix([vx, vy, vz])
    h_orientation_sym = q
    h_relative_motion = sp.Matrix(quat_rot.T * v)

    obs_eqs = [[h_gyro_sym, ObservationKind.PHONE_GYRO, None],
               [h_phone_rot_sym, ObservationKind.NO_ROT, None],
               [h_acc_sym, ObservationKind.PHONE_ACCEL, None],
               [h_pseudorange_sym, ObservationKind.PSEUDORANGE_GPS, sat_pos_freq_sym],
               [h_pseudorange_glonass_sym, ObservationKind.PSEUDORANGE_GLONASS, sat_pos_freq_sym],
               [h_pseudorange_rate_sym, ObservationKind.PSEUDORANGE_RATE_GPS, sat_pos_vel_sym],
               [h_pseudorange_rate_sym, ObservationKind.PSEUDORANGE_RATE_GLONASS, sat_pos_vel_sym],
               [h_pos_sym, ObservationKind.ECEF_POS, None],
               [h_vel_sym, ObservationKind.ECEF_VEL, None],
               [h_orientation_sym, ObservationKind.ECEF_ORIENTATION_FROM_GPS, None],
               [h_relative_motion, ObservationKind.CAMERA_ODO_TRANSLATION, None],
               [h_phone_rot_sym, ObservationKind.CAMERA_ODO_ROTATION, None],
               [h_acc_stationary_sym, ObservationKind.NO_ACCEL, None]]

    # this returns a sympy routine for the jacobian of the observation function of the local vel
    in_vec = sp.MatrixSymbol('in_vec', 6, 1)  # roll, pitch, yaw, vx, vy, vz
    h = euler_rotate(in_vec[0], in_vec[1], in_vec[2]).T * (sp.Matrix([in_vec[3], in_vec[4], in_vec[5]]))
    extra_routines = [('H', h.jacobian(in_vec), [in_vec])]

    gen_code(generated_dir, name, f_sym, dt, state_sym, obs_eqs, dim_state, dim_state_err, eskf_params, extra_routines=extra_routines)

    # write constants to extra header file for use in cpp
    live_kf_header = "#pragma once\n\n"
    live_kf_header += "#include <unordered_map>\n"
    live_kf_header += "#include <eigen3/Eigen/Dense>\n\n"
    for state, slc in inspect.getmembers(States, lambda x: type(x) == slice):
      assert(slc.step is None)  # unsupported
      live_kf_header += f'#define STATE_{state}_START {slc.start}\n'
      live_kf_header += f'#define STATE_{state}_END {slc.stop}\n'
      live_kf_header += f'#define STATE_{state}_LEN {slc.stop - slc.start}\n'
    live_kf_header += "\n"

    for kind, val in inspect.getmembers(ObservationKind, lambda x: type(x) == int):
      live_kf_header += f'#define OBSERVATION_{kind} {val}\n'
    live_kf_header += "\n"

    live_kf_header += f"static const Eigen::VectorXd live_initial_x = {numpy2eigenstring(LiveKalman.initial_x)};\n"
    live_kf_header += f"static const Eigen::VectorXd live_initial_P_diag = {numpy2eigenstring(LiveKalman.initial_P_diag)};\n"
    live_kf_header += f"static const Eigen::VectorXd live_fake_gps_pos_cov_diag = {numpy2eigenstring(LiveKalman.fake_gps_pos_cov_diag)};\n"
    live_kf_header += f"static const Eigen::VectorXd live_fake_gps_vel_cov_diag = {numpy2eigenstring(LiveKalman.fake_gps_vel_cov_diag)};\n"
    live_kf_header += f"static const Eigen::VectorXd live_reset_orientation_diag = {numpy2eigenstring(LiveKalman.reset_orientation_diag)};\n"
    live_kf_header += f"static const Eigen::VectorXd live_Q_diag = {numpy2eigenstring(LiveKalman.Q_diag)};\n"
    live_kf_header += "static const std::unordered_map<int, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> live_obs_noise_diag = {\n"
    for kind, noise in LiveKalman.obs_noise_diag.items():
      live_kf_header += f"  {{ {kind}, {numpy2eigenstring(noise)} }},\n"
    live_kf_header += "};\n\n"

    open(os.path.join(generated_dir, "live_kf_constants.h"), 'w').write(live_kf_header)


if __name__ == "__main__":
  generated_dir = sys.argv[2]
  LiveKalman.generate_code(generated_dir)
