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


class States():
  ECEF_POS = slice(0, 3)  # x, y and z in ECEF in meters
  ECEF_ORIENTATION = slice(3, 7)  # quat for pose of phone in ecef
  ECEF_VELOCITY = slice(7, 10)  # ecef velocity in m/s
  ANGULAR_VELOCITY = slice(10, 13)  # roll, pitch and yaw rates in device frame in radians/s
  GYRO_BIAS = slice(13, 16)  # roll, pitch and yaw biases
  ODO_SCALE = slice(16, 17)  # odometer scale
  ACCELERATION = slice(17, 20)  # Acceleration in device frame in m/s**2
  IMU_OFFSET = slice(20, 23)  # imu offset angles in radians
  ACC_BIAS = slice(23, 26)

  # Error-state has different slices because it is an ESKF
  ECEF_POS_ERR = slice(0, 3)
  ECEF_ORIENTATION_ERR = slice(3, 6)  # euler angles for orientation error
  ECEF_VELOCITY_ERR = slice(6, 9)
  ANGULAR_VELOCITY_ERR = slice(9, 12)
  GYRO_BIAS_ERR = slice(12, 15)
  ODO_SCALE_ERR = slice(15, 16)
  ACCELERATION_ERR = slice(16, 19)
  IMU_OFFSET_ERR = slice(19, 22)
  ACC_BIAS_ERR = slice(22, 25)


class LiveKalman():
  name = 'live'

  initial_x = np.array([3.88e6, -3.37e6, 3.76e6,
                        0.42254641, -0.31238054, -0.83602975, -0.15788347,  # NED [0,0,0] -> ECEF Quat
                        0, 0, 0,
                        0, 0, 0,
                        0, 0, 0,
                        1,
                        0, 0, 0,
                        0, 0, 0,
                        0, 0, 0])

  # state covariance
  initial_P_diag = np.array([10**2, 10**2, 10**2,
                             0.01**2, 0.01**2, 0.01**2,
                             10**2, 10**2, 10**2,
                             1**2, 1**2, 1**2,
                             1**2, 1**2, 1**2,
                             0.02**2,
                             1**2, 1**2, 1**2,
                             (0.01)**2, (0.01)**2, (0.01)**2,
                             0.1**2, 0.1**2, 0.1**2])

  # process noise
  Q_diag = np.array([0.03**2, 0.03**2, 0.03**2,
                     0.001**2, 0.001**2, 0.001**2,
                     0.01**2, 0.01**2, 0.01**2,
                     0.1**2, 0.1**2, 0.1**2,
                     (0.005 / 100)**2, (0.005 / 100)**2, (0.005 / 100)**2,
                     (0.02 / 100)**2,
                     3**2, 3**2, 3**2,
                     (0.05 / 60)**2, (0.05 / 60)**2, (0.05 / 60)**2,
                     0.01**2, 0.01**2, 0.01**2])

  obs_noise_diag = {ObservationKind.ODOMETRIC_SPEED: np.array([0.2**2]),
                    ObservationKind.PHONE_GYRO: np.array([0.025**2, 0.025**2, 0.025**2]),
                    ObservationKind.PHONE_ACCEL: np.array([.5**2, .5**2, .5**2]),
                    ObservationKind.CAMERA_ODO_ROTATION: np.array([0.05**2, 0.05**2, 0.05**2]),
                    ObservationKind.IMU_FRAME: np.array([0.05**2, 0.05**2, 0.05**2]),
                    ObservationKind.NO_ROT: np.array([0.005**2, 0.005**2, 0.005**2]),
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
    imu_angles = state[States.IMU_OFFSET, :]
    acc_bias = state[States.ACC_BIAS, :]

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

    # Basic descretization, 1st order intergrator
    # Can be pretty bad if dt is big
    f_sym = state + dt * state_dot

    state_err_sym = sp.MatrixSymbol('state_err', dim_state_err, 1)
    state_err = sp.Matrix(state_err_sym)
    quat_err = state_err[States.ECEF_ORIENTATION_ERR, :]
    v_err = state_err[States.ECEF_VELOCITY_ERR, :]
    omega_err = state_err[States.ANGULAR_VELOCITY_ERR, :]
    acceleration_err = state_err[States.ACCELERATION_ERR, :]


    # Time derivative of the state error as a function of state error and state
    quat_err_matrix = euler_rotate(quat_err[0], quat_err[1], quat_err[2])
    q_err_dot = quat_err_matrix * quat_rot * (omega + omega_err)
    state_err_dot = sp.Matrix(np.zeros((dim_state_err, 1)))
    state_err_dot[States.ECEF_POS_ERR, :] = v_err
    state_err_dot[States.ECEF_ORIENTATION_ERR, :] = q_err_dot
    state_err_dot[States.ECEF_VELOCITY_ERR, :] = quat_err_matrix * quat_rot * (acceleration + acceleration_err)
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
    delta_quat = sp.Matrix(np.ones((4)))
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
    # imu_rot = euler_rotate(*imu_angles)
    h_gyro_sym = sp.Matrix([
      vroll + roll_bias,
      vpitch + pitch_bias,
      vyaw + yaw_bias])

    pos = sp.Matrix([x, y, z])
    gravity = quat_rot.T * ((EARTH_GM / ((x**2 + y**2 + z**2)**(3.0 / 2.0))) * pos)
    h_acc_sym = (gravity + acceleration + acc_bias)
    h_phone_rot_sym = sp.Matrix([vroll, vpitch, vyaw])

    speed = sp.sqrt(vx**2 + vy**2 + vz**2 + 1e-6)
    h_speed_sym = sp.Matrix([speed])

    h_pos_sym = sp.Matrix([x, y, z])
    h_vel_sym = sp.Matrix([vx, vy, vz])
    h_orientation_sym = q
    h_imu_frame_sym = sp.Matrix(imu_angles)

    h_relative_motion = sp.Matrix(quat_rot.T * v)

    obs_eqs = [[h_speed_sym, ObservationKind.ODOMETRIC_SPEED, None],
               [h_gyro_sym, ObservationKind.PHONE_GYRO, None],
               [h_phone_rot_sym, ObservationKind.NO_ROT, None],
               [h_acc_sym, ObservationKind.PHONE_ACCEL, None],
               [h_pos_sym, ObservationKind.ECEF_POS, None],
               [h_vel_sym, ObservationKind.ECEF_VEL, None],
               [h_orientation_sym, ObservationKind.ECEF_ORIENTATION_FROM_GPS, None],
               [h_relative_motion, ObservationKind.CAMERA_ODO_TRANSLATION, None],
               [h_phone_rot_sym, ObservationKind.CAMERA_ODO_ROTATION, None],
               [h_imu_frame_sym, ObservationKind.IMU_FRAME, None]]

    # this returns a sympy routine for the jacobian of the observation function of the local vel
    in_vec = sp.MatrixSymbol('in_vec', 6, 1)  # roll, pitch, yaw, vx, vy, vz
    h = euler_rotate(in_vec[0], in_vec[1], in_vec[2]).T*(sp.Matrix([in_vec[3], in_vec[4], in_vec[5]]))
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
    live_kf_header += f"static const Eigen::VectorXd live_Q_diag = {numpy2eigenstring(LiveKalman.Q_diag)};\n"
    live_kf_header += "static const std::unordered_map<int, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> live_obs_noise_diag = {\n"
    for kind, noise in LiveKalman.obs_noise_diag.items():
      live_kf_header += f"  {{ {kind}, {numpy2eigenstring(noise)} }},\n"
    live_kf_header += "};\n\n"

    open(os.path.join(generated_dir, "live_kf_constants.h"), 'w').write(live_kf_header)


if __name__ == "__main__":
  generated_dir = sys.argv[2]
  LiveKalman.generate_code(generated_dir)
