#!/usr/bin/env python3
import numpy as np
import sympy as sp
import cereal.messaging as messaging
import common.transformations.coordinates as coord
from common.transformations.orientation import ecef_euler_from_ned, \
                                               euler_from_quat, \
                                               ned_euler_from_ecef, \
                                               quat_from_euler, euler_from_rot, \
                                               rot_from_quat, rot_from_euler
from rednose.helpers import KalmanError
from selfdrive.locationd.models.live_kf import LiveKalman, States, ObservationKind
from selfdrive.locationd.models.constants import GENERATED_DIR
from selfdrive.swaglog import cloudlog

#from datetime import datetime
#from laika.gps_time import GPSTime

from sympy.utilities.lambdify import lambdify
from rednose.helpers.sympy_helpers import euler_rotate


VISION_DECIMATION = 2
SENSOR_DECIMATION = 10
POSENET_STD_HIST = 40


def to_float(arr):
  return [float(arr[0]), float(arr[1]), float(arr[2])]


def get_H():
  # this returns a function to eval the jacobian
  # of the observation function of the local vel
  roll = sp.Symbol('roll')
  pitch = sp.Symbol('pitch')
  yaw = sp.Symbol('yaw')
  vx = sp.Symbol('vx')
  vy = sp.Symbol('vy')
  vz = sp.Symbol('vz')

  h = euler_rotate(roll, pitch, yaw).T*(sp.Matrix([vx, vy, vz]))
  H = h.jacobian(sp.Matrix([roll, pitch, yaw, vx, vy, vz]))
  H_f = lambdify([roll, pitch, yaw, vx, vy, vz], H)
  return H_f


class Localizer():
  def __init__(self, disabled_logs=None, dog=None):
    if disabled_logs is None:
      disabled_logs = []

    self.kf = LiveKalman(GENERATED_DIR)
    self.reset_kalman()
    self.max_age = .1  # seconds
    self.disabled_logs = disabled_logs
    self.calib = np.zeros(3)
    self.device_from_calib = np.eye(3)
    self.calib_from_device = np.eye(3)
    self.calibrated = 0
    self.H = get_H()

    self.posenet_invalid_count = 0
    self.posenet_speed = 0
    self.car_speed = 0
    self.posenet_stds = 10*np.ones((POSENET_STD_HIST))

    self.converter = coord.LocalCoord.from_ecef(self.kf.x[States.ECEF_POS])

    self.unix_timestamp_millis = 0
    self.last_gps_fix = 0

  @staticmethod
  def msg_from_state(converter, calib_from_device, H, predicted_state, predicted_cov):
    predicted_std = np.sqrt(np.diagonal(predicted_cov))

    fix_ecef = predicted_state[States.ECEF_POS]
    fix_ecef_std = predicted_std[States.ECEF_POS_ERR]
    vel_ecef = predicted_state[States.ECEF_VELOCITY]
    vel_ecef_std = predicted_std[States.ECEF_VELOCITY_ERR]
    fix_pos_geo = coord.ecef2geodetic(fix_ecef)
    #fix_pos_geo_std = np.abs(coord.ecef2geodetic(fix_ecef + fix_ecef_std) - fix_pos_geo)
    orientation_ecef = euler_from_quat(predicted_state[States.ECEF_ORIENTATION])
    orientation_ecef_std = predicted_std[States.ECEF_ORIENTATION_ERR]
    device_from_ecef = rot_from_quat(predicted_state[States.ECEF_ORIENTATION]).T
    calibrated_orientation_ecef = euler_from_rot(calib_from_device.dot(device_from_ecef))

    acc_calib = calib_from_device.dot(predicted_state[States.ACCELERATION])
    acc_calib_std = np.sqrt(np.diagonal(calib_from_device.dot(
      predicted_cov[States.ACCELERATION_ERR, States.ACCELERATION_ERR]).dot(
        calib_from_device.T)))
    ang_vel_calib = calib_from_device.dot(predicted_state[States.ANGULAR_VELOCITY])
    ang_vel_calib_std = np.sqrt(np.diagonal(calib_from_device.dot(
      predicted_cov[States.ANGULAR_VELOCITY_ERR, States.ANGULAR_VELOCITY_ERR]).dot(
        calib_from_device.T)))

    vel_device = device_from_ecef.dot(vel_ecef)
    device_from_ecef_eul = euler_from_quat(predicted_state[States.ECEF_ORIENTATION]).T
    idxs = list(range(States.ECEF_ORIENTATION_ERR.start, States.ECEF_ORIENTATION_ERR.stop)) + \
           list(range(States.ECEF_VELOCITY_ERR.start, States.ECEF_VELOCITY_ERR.stop))
    condensed_cov = predicted_cov[idxs][:, idxs]
    HH = H(*list(np.concatenate([device_from_ecef_eul, vel_ecef])))
    vel_device_cov = HH.dot(condensed_cov).dot(HH.T)
    vel_device_std = np.sqrt(np.diagonal(vel_device_cov))

    vel_calib = calib_from_device.dot(vel_device)
    vel_calib_std = np.sqrt(np.diagonal(calib_from_device.dot(
      vel_device_cov).dot(calib_from_device.T)))

    orientation_ned = ned_euler_from_ecef(fix_ecef, orientation_ecef)
    #orientation_ned_std = ned_euler_from_ecef(fix_ecef, orientation_ecef + orientation_ecef_std) - orientation_ned
    ned_vel = converter.ecef2ned(fix_ecef + vel_ecef) - converter.ecef2ned(fix_ecef)
    #ned_vel_std = self.converter.ecef2ned(fix_ecef + vel_ecef + vel_ecef_std) - self.converter.ecef2ned(fix_ecef + vel_ecef)

    fix = messaging.log.LiveLocationKalman.new_message()
    fix.positionGeodetic.value = to_float(fix_pos_geo)
    fix.positionGeodetic.std = to_float(np.nan*np.zeros(3))
    fix.positionGeodetic.valid = True
    fix.positionECEF.value = to_float(fix_ecef)
    fix.positionECEF.std = to_float(fix_ecef_std)
    fix.positionECEF.valid = True
    fix.velocityECEF.value = to_float(vel_ecef)
    fix.velocityECEF.std = to_float(vel_ecef_std)
    fix.velocityECEF.valid = True
    fix.velocityNED.value = to_float(ned_vel)
    fix.velocityNED.std = to_float(np.nan*np.zeros(3))
    fix.velocityNED.valid = True
    fix.velocityDevice.value = to_float(vel_device)
    fix.velocityDevice.std = to_float(vel_device_std)
    fix.velocityDevice.valid = True
    fix.accelerationDevice.value = to_float(predicted_state[States.ACCELERATION])
    fix.accelerationDevice.std = to_float(predicted_std[States.ACCELERATION_ERR])
    fix.accelerationDevice.valid = True

    fix.orientationECEF.value = to_float(orientation_ecef)
    fix.orientationECEF.std = to_float(orientation_ecef_std)
    fix.orientationECEF.valid = True
    fix.calibratedOrientationECEF.value = to_float(calibrated_orientation_ecef)
    fix.calibratedOrientationECEF.std = to_float(np.nan*np.zeros(3))
    fix.calibratedOrientationECEF.valid = True
    fix.orientationNED.value = to_float(orientation_ned)
    fix.orientationNED.std = to_float(np.nan*np.zeros(3))
    fix.orientationNED.valid = True
    fix.angularVelocityDevice.value = to_float(predicted_state[States.ANGULAR_VELOCITY])
    fix.angularVelocityDevice.std = to_float(predicted_std[States.ANGULAR_VELOCITY_ERR])
    fix.angularVelocityDevice.valid = True

    fix.velocityCalibrated.value = to_float(vel_calib)
    fix.velocityCalibrated.std = to_float(vel_calib_std)
    fix.velocityCalibrated.valid = True
    fix.angularVelocityCalibrated.value = to_float(ang_vel_calib)
    fix.angularVelocityCalibrated.std = to_float(ang_vel_calib_std)
    fix.angularVelocityCalibrated.valid = True
    fix.accelerationCalibrated.value = to_float(acc_calib)
    fix.accelerationCalibrated.std = to_float(acc_calib_std)
    fix.accelerationCalibrated.valid = True
    return fix

  def liveLocationMsg(self):
    fix = self.msg_from_state(self.converter, self.calib_from_device, self.H, self.kf.x, self.kf.P)

    #if abs(self.posenet_speed - self.car_speed) > max(0.4 * self.car_speed, 5.0):
    #  self.posenet_invalid_count += 1
    #else:
    #  self.posenet_invalid_count = 0
    #fix.posenetOK = self.posenet_invalid_count < 4

    # experimentally found these values
    old_mean, new_mean = np.mean(self.posenet_stds[:POSENET_STD_HIST//2]), np.mean(self.posenet_stds[POSENET_STD_HIST//2:])
    std_spike = new_mean/old_mean > 4 and new_mean > 5

    if std_spike and self.car_speed > 5:
      fix.posenetOK = False
    else:
      fix.posenetOK = True

    #fix.gpsWeek = self.time.week
    #fix.gpsTimeOfWeek = self.time.tow
    fix.unixTimestampMillis = self.unix_timestamp_millis

    if np.linalg.norm(fix.positionECEF.std) < 50 and self.calibrated:
      fix.status = 'valid'
    elif np.linalg.norm(fix.positionECEF.std) < 50:
      fix.status = 'uncalibrated'
    else:
      fix.status = 'uninitialized'
    return fix

  def update_kalman(self, time, kind, meas, R=None):
    try:
      self.kf.predict_and_observe(time, kind, meas, R)
    except KalmanError:
      cloudlog.error("Error in predict and observe, kalman reset")
      self.reset_kalman()

  def handle_gps(self, current_time, log):
    # ignore the message if the fix is invalid
    if log.flags % 2 == 0:
      return

    self.last_gps_fix = current_time

    self.converter = coord.LocalCoord.from_geodetic([log.latitude, log.longitude, log.altitude])
    ecef_pos = self.converter.ned2ecef([0, 0, 0])
    ecef_vel = self.converter.ned2ecef(np.array(log.vNED)) - ecef_pos
    ecef_pos_R = np.diag([(3*log.verticalAccuracy)**2]*3)
    ecef_vel_R = np.diag([(log.speedAccuracy)**2]*3)

    #self.time = GPSTime.from_datetime(datetime.utcfromtimestamp(log.timestamp*1e-3))
    self.unix_timestamp_millis = log.timestamp
    gps_est_error = np.sqrt((self.kf.x[0] - ecef_pos[0])**2 +
                            (self.kf.x[1] - ecef_pos[1])**2 +
                            (self.kf.x[2] - ecef_pos[2])**2)

    orientation_ecef = euler_from_quat(self.kf.x[States.ECEF_ORIENTATION])
    orientation_ned = ned_euler_from_ecef(ecef_pos, orientation_ecef)
    orientation_ned_gps = np.array([0, 0, np.radians(log.bearing)])
    orientation_error = np.mod(orientation_ned - orientation_ned_gps - np.pi, 2*np.pi) - np.pi
    if np.linalg.norm(ecef_vel) > 5 and np.linalg.norm(orientation_error) > 1:
      cloudlog.error("Locationd vs ubloxLocation orientation difference too large, kalman reset")
      initial_pose_ecef_quat = quat_from_euler(ecef_euler_from_ned(ecef_pos, orientation_ned_gps))
      self.reset_kalman(init_orient=initial_pose_ecef_quat)
      self.update_kalman(current_time, ObservationKind.ECEF_ORIENTATION_FROM_GPS, initial_pose_ecef_quat)
    elif gps_est_error > 50:
      cloudlog.error("Locationd vs ubloxLocation position difference too large, kalman reset")
      self.reset_kalman()

    self.update_kalman(current_time, ObservationKind.ECEF_POS, ecef_pos, R=ecef_pos_R)
    self.update_kalman(current_time, ObservationKind.ECEF_VEL, ecef_vel, R=ecef_vel_R)

  def handle_car_state(self, current_time, log):
    self.speed_counter += 1

    if self.speed_counter % SENSOR_DECIMATION == 0:
      self.update_kalman(current_time, ObservationKind.ODOMETRIC_SPEED, [log.vEgo])
      self.car_speed = abs(log.vEgo)
      if log.vEgo == 0:
        self.update_kalman(current_time, ObservationKind.NO_ROT, [0, 0, 0])

  def handle_cam_odo(self, current_time, log):
    self.cam_counter += 1

    if self.cam_counter % VISION_DECIMATION == 0:
      rot_device = self.device_from_calib.dot(log.rot)
      rot_device_std = self.device_from_calib.dot(log.rotStd)
      self.update_kalman(current_time,
                         ObservationKind.CAMERA_ODO_ROTATION,
                         np.concatenate([rot_device, 10*rot_device_std]))
      trans_device = self.device_from_calib.dot(log.trans)
      trans_device_std = self.device_from_calib.dot(log.transStd)
      self.posenet_speed = np.linalg.norm(trans_device)
      self.posenet_stds[:-1] = self.posenet_stds[1:]
      self.posenet_stds[-1] = trans_device_std[0]
      self.update_kalman(current_time,
                         ObservationKind.CAMERA_ODO_TRANSLATION,
                         np.concatenate([trans_device, 10*trans_device_std]))

  def handle_sensors(self, current_time, log):
    # TODO does not yet account for double sensor readings in the log
    for sensor_reading in log:
      # Gyro Uncalibrated
      if sensor_reading.sensor == 5 and sensor_reading.type == 16:
        self.gyro_counter += 1
        if self.gyro_counter % SENSOR_DECIMATION == 0:
          v = sensor_reading.gyroUncalibrated.v
          self.update_kalman(current_time, ObservationKind.PHONE_GYRO, [-v[2], -v[1], -v[0]])

      # Accelerometer
      if sensor_reading.sensor == 1 and sensor_reading.type == 1:
        self.acc_counter += 1
        if self.acc_counter % SENSOR_DECIMATION == 0:
          v = sensor_reading.acceleration.v
          self.update_kalman(current_time, ObservationKind.PHONE_ACCEL, [-v[2], -v[1], -v[0]])

  def handle_live_calib(self, current_time, log):
    if len(log.rpyCalib):
      self.calib = log.rpyCalib
      self.device_from_calib = rot_from_euler(self.calib)
      self.calib_from_device = self.device_from_calib.T
      self.calibrated = log.calStatus == 1

  def reset_kalman(self, current_time=None, init_orient=None):
    self.filter_time = current_time
    init_x = LiveKalman.initial_x.copy()
    # too nonlinear to init on completely wrong
    if init_orient is not None:
      init_x[3:7] = init_orient
    self.kf.init_state(init_x, covs=np.diag(LiveKalman.initial_P_diag), filter_time=current_time)

    self.observation_buffer = []

    self.gyro_counter = 0
    self.acc_counter = 0
    self.speed_counter = 0
    self.cam_counter = 0


def locationd_thread(sm, pm, disabled_logs=None):
  if disabled_logs is None:
    disabled_logs = []

  if sm is None:
    socks = ['gpsLocationExternal', 'sensorEvents', 'cameraOdometry', 'liveCalibration', 'carState']
    sm = messaging.SubMaster(socks, ignore_alive=['gpsLocationExternal'])
  if pm is None:
    pm = messaging.PubMaster(['liveLocationKalman'])

  localizer = Localizer(disabled_logs=disabled_logs)

  while True:
    sm.update()

    for sock, updated in sm.updated.items():
      if updated and sm.valid[sock]:
        t = sm.logMonoTime[sock] * 1e-9
        if sock == "sensorEvents":
          localizer.handle_sensors(t, sm[sock])
        elif sock == "gpsLocationExternal":
          localizer.handle_gps(t, sm[sock])
        elif sock == "carState":
          localizer.handle_car_state(t, sm[sock])
        elif sock == "cameraOdometry":
          localizer.handle_cam_odo(t, sm[sock])
        elif sock == "liveCalibration":
          localizer.handle_live_calib(t, sm[sock])

    if sm.updated['cameraOdometry']:
      t = sm.logMonoTime['cameraOdometry']
      msg = messaging.new_message('liveLocationKalman')
      msg.logMonoTime = t

      msg.liveLocationKalman = localizer.liveLocationMsg()
      msg.liveLocationKalman.inputsOK = sm.all_alive_and_valid()
      msg.liveLocationKalman.sensorsOK = sm.alive['sensorEvents'] and sm.valid['sensorEvents']

      gps_age = (t / 1e9) - localizer.last_gps_fix
      msg.liveLocationKalman.gpsOK = gps_age < 1.0
      pm.send('liveLocationKalman', msg)


def main(sm=None, pm=None):
  locationd_thread(sm, pm)


if __name__ == "__main__":
  import os
  os.environ["OMP_NUM_THREADS"] = "1"
  main()
