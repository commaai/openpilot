#!/usr/bin/env python3
import math
from bisect import bisect_right

import numpy as np

import cereal.messaging as messaging
import common.transformations.coordinates as coord
from common.transformations.orientation import (ecef_euler_from_ned,
                                                euler2quat,
                                                ned_euler_from_ecef,
                                                quat2euler,
                                                rotations_from_quats)
from selfdrive.locationd.kalman.helpers import ObservationKind, KalmanError
from selfdrive.locationd.kalman.models.live_kf import LiveKalman, States
from selfdrive.swaglog import cloudlog

SENSOR_DECIMATION = 1  # No decimation


class Localizer():
  def __init__(self, disabled_logs=[], dog=None):
    self.kf = LiveKalman()
    self.reset_kalman()
    self.max_age = .2  # seconds
    self.disabled_logs = disabled_logs

  def liveLocationMsg(self, time):
    fix = messaging.log.LiveLocationData.new_message()

    predicted_state = self.kf.x

    fix_ecef = predicted_state[States.ECEF_POS]
    fix_pos_geo = coord.ecef2geodetic(fix_ecef)
    fix.lat = float(fix_pos_geo[0])
    fix.lon = float(fix_pos_geo[1])
    fix.alt = float(fix_pos_geo[2])

    fix.speed = float(np.linalg.norm(predicted_state[States.ECEF_VELOCITY]))

    orientation_ned_euler = ned_euler_from_ecef(fix_ecef, quat2euler(predicted_state[States.ECEF_ORIENTATION]))
    fix.roll = math.degrees(orientation_ned_euler[0])
    fix.pitch = math.degrees(orientation_ned_euler[1])
    fix.heading = math.degrees(orientation_ned_euler[2])

    fix.gyro = [float(predicted_state[10]), float(predicted_state[11]), float(predicted_state[12])]
    fix.accel = [float(predicted_state[19]), float(predicted_state[20]), float(predicted_state[21])]

    local_vel = rotations_from_quats(predicted_state[States.ECEF_ORIENTATION]).T.dot(predicted_state[States.ECEF_VELOCITY])
    fix.pitchCalibration = math.degrees(math.atan2(local_vel[2], local_vel[0]))
    fix.yawCalibration = math.degrees(math.atan2(local_vel[1], local_vel[0]))

    #fix.imuFrame = [(180/np.pi)*float(predicted_state[23]), (180/np.pi)*float(predicted_state[24]), (180/np.pi)*float(predicted_state[25])]
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
    converter = coord.LocalCoord.from_geodetic([log.latitude, log.longitude, log.altitude])
    fix_ecef = converter.ned2ecef([0, 0, 0])

    # TODO initing with bad bearing not allowed, maybe not bad?
    if not self.filter_ready and log.speed > 5:
      self.filter_ready = True
      initial_ecef = fix_ecef
      gps_bearing = math.radians(log.bearing)
      initial_pose_ecef = ecef_euler_from_ned(initial_ecef, [0, 0, gps_bearing])
      initial_pose_ecef_quat = euler2quat(initial_pose_ecef)
      gps_speed = log.speed
      quat_uncertainty = 0.2**2
      initial_pose_ecef_quat = euler2quat(initial_pose_ecef)

      initial_state = LiveKalman.initial_x
      initial_covs_diag = LiveKalman.initial_P_diag

      initial_state[States.ECEF_POS] = initial_ecef
      initial_state[States.ECEF_ORIENTATION] = initial_pose_ecef_quat
      initial_state[States.ECEF_VELOCITY] = rotations_from_quats(initial_pose_ecef_quat).dot(np.array([gps_speed, 0, 0]))

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
    self.update_kalman(current_time,
                       ObservationKind.CAMERA_ODO_ROTATION,
                       np.concatenate([log.rot, log.rotStd]))
    self.update_kalman(current_time,
                       ObservationKind.CAMERA_ODO_TRANSLATION,
                       np.concatenate([log.trans, log.transStd]))

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

  def reset_kalman(self):
    self.filter_time = None
    self.filter_ready = False
    self.observation_buffer = []

    self.gyro_counter = 0
    self.acc_counter = 0
    self.speed_counter = 0


def locationd_thread(sm, pm, disabled_logs=[]):
  if sm is None:
    sm = messaging.SubMaster(['gpsLocationExternal', 'sensorEvents', 'cameraOdometry'])
  if pm is None:
    pm = messaging.PubMaster(['liveLocation'])

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

    if localizer.filter_ready and sm.updated['gpsLocationExternal']:
      t = sm.logMonoTime['gpsLocationExternal']
      msg = messaging.new_message()
      msg.logMonoTime = t

      msg.init('liveLocation')
      msg.liveLocation = localizer.liveLocationMsg(t * 1e-9)

      pm.send('liveLocation', msg)


def main(sm=None, pm=None):
  locationd_thread(sm, pm)


if __name__ == "__main__":
  import os
  os.environ["OMP_NUM_THREADS"] = "1"
  main()
