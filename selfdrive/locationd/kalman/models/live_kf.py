#!/usr/bin/env python3
import numpy as np
import sympy as sp

from laika.constants import EARTH_GM
from selfdrive.locationd.kalman.helpers import KalmanError, ObservationKind
from selfdrive.locationd.kalman.helpers.ekf_sym import EKF_sym, gen_code
from selfdrive.locationd.kalman.helpers.sympy_helpers import (euler_rotate,
                                                              quat_matrix_r,
                                                              quat_rotate)
from selfdrive.swaglog import cloudlog


class States():
  ECEF_POS = slice(0, 3)  # x, y and z in ECEF in meters
  ECEF_ORIENTATION = slice(3, 7)  # quat for pose of phone in ecef
  ECEF_VELOCITY = slice(7, 10)  # ecef velocity in m/s
  ANGULAR_VELOCITY = slice(10, 13)  # roll, pitch and yaw rates in device frame in radians/s
  GYRO_BIAS = slice(13, 16)  # roll, pitch and yaw biases
  ODO_SCALE = slice(16, 17)  # odometer scale
  ACCELERATION = slice(17, 20)  # Acceleration in device frame in m/s**2
  IMU_OFFSET = slice(20, 23)  # imu offset angles in radians

  ECEF_POS_ERR = slice(0, 3)
  ECEF_ORIENTATION_ERR = slice(3, 6)
  ECEF_VELOCITY_ERR = slice(6, 9)
  ANGULAR_VELOCITY_ERR = slice(9, 12)
  GYRO_BIAS_ERR = slice(12, 15)
  ODO_SCALE_ERR = slice(15, 16)
  ACCELERATION_ERR = slice(16, 19)
  IMU_OFFSET_ERR = slice(19, 22)


class LiveKalman():
  name = 'live'

  initial_x = np.array([-2.7e6, 4.2e6, 3.8e6,
                        1, 0, 0, 0,
                        0, 0, 0,
                        0, 0, 0,
                        0, 0, 0,
                        1,
                        0, 0, 0,
                        0, 0, 0])


  # state covariance
  initial_P_diag = np.array([10000**2, 10000**2, 10000**2,
                             10**2, 10**2, 10**2,
                             10**2, 10**2, 10**2,
                             1**2, 1**2, 1**2,
                             0.05**2, 0.05**2, 0.05**2,
                             0.02**2,
                             1**2, 1**2, 1**2,
                             (0.01)**2, (0.01)**2, (0.01)**2])

  # process noise
  Q = np.diag([0.03**2, 0.03**2, 0.03**2,
               0.0**2, 0.0**2, 0.0**2,
               0.0**2, 0.0**2, 0.0**2,
               0.1**2, 0.1**2, 0.1**2,
               (0.005/100)**2, (0.005/100)**2, (0.005/100)**2,
               (0.02/100)**2,
               3**2, 3**2, 3**2,
               (0.05/60)**2, (0.05/60)**2, (0.05/60)**2])

  @staticmethod
  def generate_code():
    name = LiveKalman.name
    dim_state = LiveKalman.initial_x.shape[0]
    dim_state_err = LiveKalman.initial_P_diag.shape[0]

    state_sym = sp.MatrixSymbol('state', dim_state, 1)
    state = sp.Matrix(state_sym)
    x,y,z = state[States.ECEF_POS,:]
    q = state[States.ECEF_ORIENTATION,:]
    v = state[States.ECEF_VELOCITY,:]
    vx, vy, vz = v
    omega = state[States.ANGULAR_VELOCITY,:]
    vroll, vpitch, vyaw = omega
    roll_bias, pitch_bias, yaw_bias = state[States.GYRO_BIAS,:]
    odo_scale = state[16,:]
    acceleration = state[States.ACCELERATION,:]
    imu_angles= state[States.IMU_OFFSET,:]

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
    state_dot[States.ECEF_POS,:] = v
    state_dot[States.ECEF_ORIENTATION,:] = q_dot
    state_dot[States.ECEF_VELOCITY,0] = quat_rot * acceleration

    # Basic descretization, 1st order intergrator
    # Can be pretty bad if dt is big
    f_sym = state + dt*state_dot

    state_err_sym = sp.MatrixSymbol('state_err',dim_state_err,1)
    state_err = sp.Matrix(state_err_sym)
    quat_err = state_err[States.ECEF_ORIENTATION_ERR,:]
    v_err = state_err[States.ECEF_VELOCITY_ERR,:]
    omega_err = state_err[States.ANGULAR_VELOCITY_ERR,:]
    acceleration_err = state_err[States.ACCELERATION_ERR,:]

    # Time derivative of the state error as a function of state error and state
    quat_err_matrix = euler_rotate(quat_err[0], quat_err[1], quat_err[2])
    q_err_dot = quat_err_matrix * quat_rot * (omega + omega_err)
    state_err_dot = sp.Matrix(np.zeros((dim_state_err, 1)))
    state_err_dot[States.ECEF_POS_ERR,:] = v_err
    state_err_dot[States.ECEF_ORIENTATION_ERR,:] = q_err_dot
    state_err_dot[States.ECEF_VELOCITY_ERR,:] = quat_err_matrix * quat_rot * (acceleration + acceleration_err)
    f_err_sym = state_err + dt*state_err_dot

    # Observation matrix modifier
    H_mod_sym = sp.Matrix(np.zeros((dim_state, dim_state_err)))
    H_mod_sym[0:3, 0:3] = np.eye(3)
    H_mod_sym[3:7,3:6] = 0.5*quat_matrix_r(state[3:7])[:,1:]
    H_mod_sym[7:, 6:] = np.eye(dim_state-7)

    # these error functions are defined so that say there
    # is a nominal x and true x:
    # true x = err_function(nominal x, delta x)
    # delta x = inv_err_function(nominal x, true x)
    nom_x = sp.MatrixSymbol('nom_x',dim_state,1)
    true_x = sp.MatrixSymbol('true_x',dim_state,1)
    delta_x = sp.MatrixSymbol('delta_x',dim_state_err,1)

    err_function_sym = sp.Matrix(np.zeros((dim_state,1)))
    delta_quat = sp.Matrix(np.ones((4)))
    delta_quat[1:,:] = sp.Matrix(0.5*delta_x[3:6,:])
    err_function_sym[0:3,:] = sp.Matrix(nom_x[0:3,:] + delta_x[0:3,:])
    err_function_sym[3:7,0] = quat_matrix_r(nom_x[3:7,0])*delta_quat
    err_function_sym[7:,:] = sp.Matrix(nom_x[7:,:] + delta_x[6:,:])

    inv_err_function_sym = sp.Matrix(np.zeros((dim_state_err,1)))
    inv_err_function_sym[0:3,0] = sp.Matrix(-nom_x[0:3,0] + true_x[0:3,0])
    delta_quat = quat_matrix_r(nom_x[3:7,0]).T*true_x[3:7,0]
    inv_err_function_sym[3:6,0] = sp.Matrix(2*delta_quat[1:])
    inv_err_function_sym[6:,0] = sp.Matrix(-nom_x[7:,0] + true_x[7:,0])

    eskf_params = [[err_function_sym, nom_x, delta_x],
                  [inv_err_function_sym, nom_x, true_x],
                  H_mod_sym, f_err_sym, state_err_sym]



    #
    # Observation functions
    #


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

    h_pos_sym = sp.Matrix([x, y, z])
    h_imu_frame_sym = sp.Matrix(imu_angles)

    h_relative_motion = sp.Matrix(quat_rot.T * v)


    obs_eqs = [[h_speed_sym, ObservationKind.ODOMETRIC_SPEED, None],
              [h_gyro_sym, ObservationKind.PHONE_GYRO, None],
              [h_phone_rot_sym, ObservationKind.NO_ROT, None],
              [h_acc_sym, ObservationKind.PHONE_ACCEL, None],
              [h_pos_sym, ObservationKind.ECEF_POS, None],
              [h_relative_motion, ObservationKind.CAMERA_ODO_TRANSLATION, None],
              [h_phone_rot_sym, ObservationKind.CAMERA_ODO_ROTATION, None],
              [h_imu_frame_sym, ObservationKind.IMU_FRAME, None]]

    gen_code(name, f_sym, dt, state_sym, obs_eqs, dim_state, dim_state_err, eskf_params)

  def __init__(self):
    self.dim_state = self.initial_x.shape[0]
    self.dim_state_err = self.initial_P_diag.shape[0]

    self.obs_noise = {ObservationKind.ODOMETRIC_SPEED: np.atleast_2d(0.2**2),
                      ObservationKind.PHONE_GYRO: np.diag([0.025**2, 0.025**2, 0.025**2]),
                      ObservationKind.PHONE_ACCEL: np.diag([.5**2, .5**2, .5*2]),
                      ObservationKind.CAMERA_ODO_ROTATION: np.diag([0.05**2, 0.05**2, 0.05**2]),
                      ObservationKind.IMU_FRAME: np.diag([0.05**2, 0.05**2, 0.05**2]),
                      ObservationKind.NO_ROT: np.diag([0.00025**2, 0.00025**2, 0.00025**2]),
                      ObservationKind.ECEF_POS: np.diag([5**2, 5**2, 5**2])}

    # init filter
    self.filter = EKF_sym(self.name, self.Q, self.initial_x, np.diag(self.initial_P_diag), self.dim_state, self.dim_state_err)

  @property
  def x(self):
    return self.filter.state()

  @property
  def t(self):
    return self.filter.filter_time

  @property
  def P(self):
    return self.filter.covs()

  def predict(self, t):
    return self.filter.predict(t)

  def rts_smooth(self, estimates):
    return self.filter.rts_smooth(estimates, norm_quats=True)

  def init_state(self, state, covs_diag=None, covs=None, filter_time=None):
    if covs_diag is not None:
      P = np.diag(covs_diag)
    elif covs is not None:
      P = covs
    else:
      P = self.filter.covs()
    self.filter.init_state(state, P, filter_time)

  def predict_and_observe(self, t, kind, data):
    if len(data) > 0:
      data = np.atleast_2d(data)
    if kind == ObservationKind.CAMERA_ODO_TRANSLATION:
      r = self.predict_and_update_odo_trans(data, t, kind)
    elif kind == ObservationKind.CAMERA_ODO_ROTATION:
      r = self.predict_and_update_odo_rot(data, t, kind)
    elif kind == ObservationKind.ODOMETRIC_SPEED:
      r = self.predict_and_update_odo_speed(data, t, kind)
    else:
      r = self.filter.predict_and_update_batch(t, kind, data, self.get_R(kind, len(data)))

    # Normalize quats
    quat_norm = np.linalg.norm(self.filter.x[3:7, 0])

    # Should not continue if the quats behave this weirdly
    if not (0.1 < quat_norm < 10):
      cloudlog.error("Kalman filter quaternions unstable")
      raise KalmanError

    self.filter.x[States.ECEF_ORIENTATION, 0] = self.filter.x[States.ECEF_ORIENTATION, 0] / quat_norm

    return r

  def get_R(self, kind, n):
    obs_noise = self.obs_noise[kind]
    dim = obs_noise.shape[0]
    R = np.zeros((n, dim, dim))
    for i in range(n):
      R[i, :, :] = obs_noise
    return R

  def predict_and_update_odo_speed(self, speed, t, kind):
    z = np.array(speed)
    R = np.zeros((len(speed), 1, 1))
    for i, _ in enumerate(z):
      R[i, :, :] = np.diag([0.2**2])
    return self.filter.predict_and_update_batch(t, kind, z, R)

  def predict_and_update_odo_trans(self, trans, t, kind):
    z = trans[:, :3]
    R = np.zeros((len(trans), 3, 3))
    for i, _ in enumerate(z):
        R[i, :, :] = np.diag(trans[i, 3:]**2)
    return self.filter.predict_and_update_batch(t, kind, z, R)

  def predict_and_update_odo_rot(self, rot, t, kind):
    z = rot[:, :3]
    R = np.zeros((len(rot), 3, 3))
    for i, _ in enumerate(z):
        R[i, :, :] = np.diag(rot[i, 3:]**2)
    return self.filter.predict_and_update_batch(t, kind, z, R)


if __name__ == "__main__":
  LiveKalman.generate_code()
