#!/usr/bin/env python3
import math

import numpy as np

import cereal.messaging as messaging
import common.transformations.coordinates as coord
from common.transformations.orientation import (ecef_euler_from_ned,
                                                euler_from_quat,
                                                ned_euler_from_ecef,
                                                quat_from_euler,
                                                rot_from_quat, rot_from_euler)
from selfdrive.locationd.kalman.helpers import ObservationKind, KalmanError
from selfdrive.locationd.kalman.models.live_kf import LiveKalman, States
from selfdrive.swaglog import cloudlog
#from datetime import datetime
#from laika.gps_time import GPSTime


VISION_DECIMATION = 2
SENSOR_DECIMATION = 10


def to_float(arr):
  return [float(arr[0]), float(arr[1]), float(arr[2])]


class Localizer():
  def __init__(self, disabled_logs=[], dog=None):
    self.kf = LiveKalman()
    self.reset_kalman()
    self.max_age = .2  # seconds
    self.disabled_logs = disabled_logs
    self.calib = np.zeros(3)
    self.device_from_calib = np.eye(3)
    self.calib_from_device = np.eye(3)
    self.calibrated = 0

  def liveLocationMsg(self, time):
    predicted_state = self.kf.x
    predicted_std = np.sqrt(np.diagonal(self.kf.P))

    fix_ecef = predicted_state[States.ECEF_POS]
    fix_ecef_std = predicted_std[States.ECEF_POS_ERR]
    vel_ecef = predicted_state[States.ECEF_VELOCITY]
    vel_ecef_std = predicted_std[States.ECEF_VELOCITY_ERR]
    fix_pos_geo = coord.ecef2geodetic(fix_ecef)
    fix_pos_geo_std = coord.ecef2geodetic(fix_ecef + fix_ecef_std) - fix_pos_geo
    ned_vel = self.converter.ecef2ned(fix_ecef + vel_ecef) - self.converter.ecef2ned(fix_ecef)
    ned_vel_std = self.converter.ecef2ned(fix_ecef + vel_ecef + vel_ecef_std) - self.converter.ecef2ned(fix_ecef + vel_ecef)
    device_from_ecef = rot_from_quat(predicted_state[States.ECEF_ORIENTATION]).T
    vel_device = device_from_ecef.dot(vel_ecef)
    vel_device_std = device_from_ecef.dot(vel_ecef_std)
    orientation_ecef = euler_from_quat(predicted_state[States.ECEF_ORIENTATION])
    orientation_ecef_std = predicted_std[States.ECEF_ORIENTATION_ERR]
    orientation_ned = ned_euler_from_ecef(fix_ecef, orientation_ecef)
    orientation_ned_std = ned_euler_from_ecef(fix_ecef, orientation_ecef + orientation_ecef_std) - orientation_ned
    vel_calib = self.calib_from_device.dot(vel_device)
    vel_calib_std = self.calib_from_device.dot(vel_device_std)
    acc_calib = self.calib_from_device.dot(predicted_state[States.ACCELERATION])
    acc_calib_std = self.calib_from_device.dot(predicted_std[States.ACCELERATION_ERR])
    ang_vel_calib = self.calib_from_device.dot(predicted_state[States.ANGULAR_VELOCITY])
    ang_vel_calib_std = self.calib_from_device.dot(predicted_std[States.ANGULAR_VELOCITY_ERR])


    fix = messaging.log.LiveLocationKalman.new_message()
    fix.positionGeodetic.value = to_float(fix_pos_geo)
    fix.positionGeodetic.std = to_float(fix_pos_geo_std)
    fix.positionGeodetic.valid = True
    fix.positionECEF.value = to_float(fix_ecef)
    fix.positionECEF.std = to_float(fix_ecef_std)
    fix.positionECEF.valid = True
    fix.velocityECEF.value = to_float(vel_ecef)
    fix.velocityECEF.std = to_float(vel_ecef_std)
    fix.velocityECEF.valid = True
    fix.velocityNED.value = to_float(ned_vel)
    fix.velocityNED.std = to_float(ned_vel_std)
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
    fix.orientationNED.value = to_float(orientation_ned)
    fix.orientationNED.std = to_float(orientation_ned_std)
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

    #fix.gpsWeek = self.time.week
    #fix.gpsTimeOfWeek = self.time.tow
    fix.unixTimestampMillis = self.unix_timestamp_millis

    if self.filter_ready and self.calibrated:
      fix.status = 'valid'
    elif self.filter_ready:
      fix.status = 'uncalibrated'
    else:
      fix.status = 'uninitialized'
    return fix

  def update_kalman(self, time, kind, meas):
    if self.filter_ready:
      try:
        self.kf.predict_and_observe(time, kind, meas)
      except KalmanError:
        cloudlog.error("Error in predict and observe, kalman reset")
        self.reset_kalman()
    #idx = bisect_right([x[0] for x in self.observation_buffer], time)
    #self.observation_buffer.insert(idx, (time, kind, meas))
    #while len(self.observation_buffer) > 0 and self.observation_buffer[-1][0] - self.observation_buffer[0][0] > self.max_age:
    #  else:
    #    self.observation_buffer.pop(0)

  def handle_gps(self, current_time, log):
    self.converter = coord.LocalCoord.from_geodetic([log.latitude, log.longitude, log.altitude])
    fix_ecef = self.converter.ned2ecef([0, 0, 0])

    #self.time = GPSTime.from_datetime(datetime.utcfromtimestamp(log.timestamp*1e-3))
    self.unix_timestamp_millis = log.timestamp

    # TODO initing with bad bearing not allowed, maybe not bad?
    if not self.filter_ready and log.speed > 5:
      self.filter_ready = True
      initial_ecef = fix_ecef
      gps_bearing = math.radians(log.bearing)
      initial_pose_ecef = ecef_euler_from_ned(initial_ecef, [0, 0, gps_bearing])
      initial_pose_ecef_quat = quat_from_euler(initial_pose_ecef)
      gps_speed = log.speed
      quat_uncertainty = 0.2**2
      initial_pose_ecef_quat = quat_from_euler(initial_pose_ecef)

      initial_state = LiveKalman.initial_x
      initial_covs_diag = LiveKalman.initial_P_diag

      initial_state[States.ECEF_POS] = initial_ecef
      initial_state[States.ECEF_ORIENTATION] = initial_pose_ecef_quat
      initial_state[States.ECEF_VELOCITY] = rot_from_quat(initial_pose_ecef_quat).dot(np.array([gps_speed, 0, 0]))

      initial_covs_diag[States.ECEF_POS_ERR] = 10**2
      initial_covs_diag[States.ECEF_ORIENTATION_ERR] = quat_uncertainty
      initial_covs_diag[States.ECEF_VELOCITY_ERR] = 1**2
      self.kf.init_state(initial_state, covs=np.diag(initial_covs_diag), filter_time=current_time)
      cloudlog.info("Filter initialized")
    elif self.filter_ready:
      self.update_kalman(current_time, ObservationKind.ECEF_POS, fix_ecef)
      gps_est_error = np.sqrt((self.kf.x[0] - fix_ecef[0])**2 +
                              (self.kf.x[1] - fix_ecef[1])**2 +
                              (self.kf.x[2] - fix_ecef[2])**2)
      if gps_est_error > 50:
        cloudlog.error("Locationd vs ubloxLocation difference too large, kalman reset")
        self.reset_kalman()

  def handle_car_state(self, current_time, log):
    self.speed_counter += 1

    if self.speed_counter % SENSOR_DECIMATION == 0:
      self.update_kalman(current_time, ObservationKind.ODOMETRIC_SPEED, [log.vEgo])
      if log.vEgo == 0:
        self.update_kalman(current_time, ObservationKind.NO_ROT, [0, 0, 0])

  def handle_cam_odo(self, current_time, log):
    self.cam_counter += 1

    if self.cam_counter % VISION_DECIMATION == 0:
      rot_device = self.device_from_calib.dot(log.rot)
      rot_device_std = self.device_from_calib.dot(log.rotStd)
      self.update_kalman(current_time,
                         ObservationKind.CAMERA_ODO_ROTATION,
                         np.concatenate([rot_device, rot_device_std]))
      trans_device = self.device_from_calib.dot(log.trans)
      trans_device_std = self.device_from_calib.dot(log.transStd)
      self.update_kalman(current_time,
                         ObservationKind.CAMERA_ODO_TRANSLATION,
                         np.concatenate([trans_device, trans_device_std]))

  def handle_sensors(self, current_time, log):
    # TODO does not yet account for double sensor readings in the log
    for sensor_reading in log:
      # Gyro Uncalibrated
      if sensor_reading.sensor == 5 and sensor_reading.type == 16:
        self.gyro_counter += 1
        if self.gyro_counter % SENSOR_DECIMATION == 0:
          if max(abs(self.kf.x[States.IMU_OFFSET])) > 0.07:
            cloudlog.info('imu frame angles exceeded, correcting')
            self.update_kalman(current_time, ObservationKind.IMU_FRAME, [0, 0, 0])

          v = sensor_reading.gyroUncalibrated.v
          self.update_kalman(current_time, ObservationKind.PHONE_GYRO, [-v[2], -v[1], -v[0]])

      # Accelerometer
      if sensor_reading.sensor == 1 and sensor_reading.type == 1:
        self.acc_counter += 1
        if self.acc_counter % SENSOR_DECIMATION == 0:
          v = sensor_reading.acceleration.v
          self.update_kalman(current_time, ObservationKind.PHONE_ACCEL, [-v[2], -v[1], -v[0]])

  def handle_live_calib(self, current_time, log):
    self.calib = log.rpyCalib
    self.device_from_calib = rot_from_euler(self.calib)
    self.calib_from_device = self.device_from_calib.T
    self.calibrated = log.calStatus == 1

  def reset_kalman(self):
    self.filter_time = None
    self.filter_ready = False
    self.observation_buffer = []

    self.gyro_counter = 0
    self.acc_counter = 0
    self.speed_counter = 0
    self.cam_counter = 0


def locationd_thread(sm, pm, disabled_logs=[]):
  if sm is None:
    sm = messaging.SubMaster(['gpsLocationExternal', 'sensorEvents', 'cameraOdometry', 'liveCalibration'])
  if pm is None:
    pm = messaging.PubMaster(['liveLocationKalman'])

  localizer = Localizer(disabled_logs=disabled_logs)

  while True:
    sm.update()

    for sock, updated in sm.updated.items():
      if updated:
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

    if localizer.filter_ready and sm.updated['gpsLocationExternal']:
      t = sm.logMonoTime['gpsLocationExternal']
      msg = messaging.new_message('liveLocationKalman')
      msg.logMonoTime = t

      msg.liveLocationKalman = localizer.liveLocationMsg(t * 1e-9)
      pm.send('liveLocationKalman', msg)


def main(sm=None, pm=None):
  locationd_thread(sm, pm)


if __name__ == "__main__":
  import os
  os.environ["OMP_NUM_THREADS"] = "1"
  main()
