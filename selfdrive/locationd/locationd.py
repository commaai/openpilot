#!/usr/bin/env python3
import os
import zmq
import numpy as np
from bisect import bisect_right
import cereal.messaging as messaging
from selfdrive.swaglog import cloudlog
from cereal.services import service_list
from common.transformations.orientation import rotations_from_quats, ecef_euler_from_ned, euler2quat, ned_euler_from_ecef, quat2euler
import common.transformations.coordinates as coord

import laika.raw_gnss as gnss
from laika.astro_dog import AstroDog

from selfdrive.locationd.kalman.loc_kf import LocKalman
from selfdrive.locationd.kalman.kalman_helpers import ObservationKind
os.environ["OMP_NUM_THREADS"] = "1"


class Localizer():
  def __init__(self, disabled_logs=[], dog=None):
    self.kf = LocKalman(0)
    self.reset_kalman()
    if dog:
      self.dog = dog
    else:
      self.dog = AstroDog(auto_update=True)
    self.max_age = .2  # seconds
    self.disabled_logs = disabled_logs
    self.week = None

  def liveLocationMsg(self, time):
    fix = messaging.log.LiveLocationData.new_message()
    predicted_state = self.kf.x
    fix_ecef = predicted_state[0:3]
    fix_pos_geo = coord.ecef2geodetic(fix_ecef)
    fix.lat = float(fix_pos_geo[0])
    fix.lon = float(fix_pos_geo[1])
    fix.alt = float(fix_pos_geo[2])

    fix.speed = float(np.linalg.norm(predicted_state[7:10]))

    orientation_ned_euler = ned_euler_from_ecef(fix_ecef, quat2euler(predicted_state[3:7]))
    fix.roll = float(orientation_ned_euler[0]*180/np.pi)
    fix.pitch = float(orientation_ned_euler[1]*180/np.pi)
    fix.heading = float(orientation_ned_euler[2]*180/np.pi)

    fix.gyro = [float(predicted_state[10]), float(predicted_state[11]), float(predicted_state[12])]
    fix.accel = [float(predicted_state[19]), float(predicted_state[20]), float(predicted_state[21])]
    local_vel = rotations_from_quats(predicted_state[3:7]).T.dot(predicted_state[7:10])
    fix.pitchCalibration = float((np.arctan2(local_vel[2], local_vel[0]))*180/np.pi)
    fix.yawCalibration = float((np.arctan2(local_vel[1], local_vel[0]))*180/np.pi)
    fix.imuFrame = [(180/np.pi)*float(predicted_state[23]), (180/np.pi)*float(predicted_state[24]), (180/np.pi)*float(predicted_state[25])]
    return fix


  def update_kalman(self, time, kind, meas):
    idx = bisect_right([x[0] for x in self.observation_buffer], time)
    self.observation_buffer.insert(idx, (time, kind, meas))
    #print len(self.observation_buffer), idx, self.kf.filter.filter_time, time
    while self.observation_buffer[-1][0] - self.observation_buffer[0][0] > self.max_age:
      if self.filter_ready:
        self.kf.predict_and_observe(*self.observation_buffer.pop(0))
      else:
        self.observation_buffer.pop(0)

  def handle_gps(self, log, current_time):
    self.converter = coord.LocalCoord.from_geodetic([log.gpsLocationExternal.latitude, log.gpsLocationExternal.longitude, log.gpsLocationExternal.altitude])
    fix_ecef = self.converter.ned2ecef([0,0,0])
    # initing with bad bearing allowed, maybe bad?
    if not self.filter_ready and len(list(self.dog.orbits.keys())) >6:  # and log.gpsLocationExternal.speed > 5:
      self.filter_ready = True
      initial_ecef = fix_ecef
      initial_state = np.zeros(29)
      gps_bearing = log.gpsLocationExternal.bearing*(np.pi/180)
      initial_pose_ecef = ecef_euler_from_ned(initial_ecef, [0, 0, gps_bearing])
      initial_pose_ecef_quat = euler2quat(initial_pose_ecef)
      gps_speed = log.gpsLocationExternal.speed
      quat_uncertainty = 0.2**2
      initial_pose_ecef_quat = euler2quat(initial_pose_ecef)
      initial_state[:3] = initial_ecef
      initial_state[3:7] = initial_pose_ecef_quat
      initial_state[7:10] = rotations_from_quats(initial_pose_ecef_quat).dot(np.array([gps_speed, 0, 0]))
      initial_state[18] = 1
      initial_state[22] = 1
      covs_diag = np.array([10**2,10**2,10**2,
                     quat_uncertainty, quat_uncertainty, quat_uncertainty,
                     2**2, 2**2, 2**2,
                     1, 1, 1,
                     20000000**2, 100**2,
                     0.01**2, 0.01**2, 0.01**2,
                     0.02**2,
                     2**2, 2**2, 2**2,
                     .01**2,
                     0.01**2, 0.01**2, 0.01**2,
                     10**2, 1**2,
                     0.2**2])
      self.kf.init_state(initial_state, covs=np.diag(covs_diag), filter_time=current_time)
      print("Filter initialized")
    elif self.filter_ready:
      #self.update_kalman(current_time, ObservationKind.ECEF_POS, fix_ecef)
      gps_est_error = np.sqrt((self.kf.x[0] - fix_ecef[0])**2 +
                              (self.kf.x[1] - fix_ecef[1])**2 +
                              (self.kf.x[2] - fix_ecef[2])**2)
      if gps_est_error > 50:
        cloudlog.info("Locationd vs ubloxLocation difference too large, kalman reset")
        self.reset_kalman()

  def handle_car_state(self, log, current_time):
    self.speed_counter += 1
    if self.speed_counter % 5==0:
      self.update_kalman(current_time, ObservationKind.ODOMETRIC_SPEED, log.carState.vEgo)
      if log.carState.vEgo == 0:
        self.update_kalman(current_time, ObservationKind.NO_ROT, [0, 0, 0])

  def handle_ublox_gnss(self, log, current_time):
    if hasattr(log.ubloxGnss, 'measurementReport'):
      self.raw_gnss_counter += 1
      if True or self.raw_gnss_counter % 3==0:
        processed_raw = gnss.process_measurements(gnss.read_raw_ublox(log.ubloxGnss.measurementReport), dog=self.dog)
        corrected_raw = gnss.correct_measurements(processed_raw, self.kf.x[:3], dog=self.dog)
        corrected_raw = np.array([c.as_array() for c in corrected_raw]).reshape((-1,14))
        self.update_kalman(current_time, ObservationKind.PSEUDORANGE_GPS, corrected_raw)
        self.update_kalman(current_time, ObservationKind.PSEUDORANGE_RATE_GPS, corrected_raw)
    #elif hasattr(log.ubloxGnss, 'ephemeris'):
    #  self.dog.add_ublox_ephems([log])
    #  if len(self.dog.orbits.keys()) < 6:
    #    print 'Added ublox ephem now has ', len(self.dog.orbits.keys())

  def handle_qcom_gnss(self, log, current_time):
    if hasattr(log.qcomGnss, 'drSvPoly') and self.week is not None:
      self.dog.add_qcom_ephems([log], self.week)
      if len(list(self.dog.orbits.keys())) < 6:
        print('Added qcom ephem now has ', len(list(self.dog.orbits.keys())))
    if hasattr(log.qcomGnss, 'drMeasurementReport') and log.qcomGnss.drMeasurementReport.source == "gps":
      self.week = log.qcomGnss.drMeasurementReport.gpsWeek

  def handle_cam_odo(self, log, current_time):
    self.update_kalman(current_time, ObservationKind.CAMERA_ODO_ROTATION, np.concatenate([log.cameraOdometry.rot,
                                                                                          log.cameraOdometry.rotStd]))
    self.update_kalman(current_time, ObservationKind.CAMERA_ODO_TRANSLATION, np.concatenate([log.cameraOdometry.trans,
                                                                                             log.cameraOdometry.transStd]))
    pass

  def handle_sensors(self, log, current_time):
    for sensor_reading in log.sensorEvents:
      # TODO does not yet account for double sensor readings in the log
      if sensor_reading.type == 4:
        self.gyro_counter += 1
        if True or self.gyro_counter % 5==0:
          if max(abs(self.kf.x[23:26])) > 0.07:
            print('imu frame angles exceeded, correcting')
            self.update_kalman(current_time, ObservationKind.IMU_FRAME, [0, 0, 0])
          self.update_kalman(current_time, ObservationKind.PHONE_GYRO, [-sensor_reading.gyro.v[2], -sensor_reading.gyro.v[1], -sensor_reading.gyro.v[0]])
      if sensor_reading.type == 1:
        self.acc_counter += 1
        if True or self.acc_counter % 5==0:
          self.update_kalman(current_time, ObservationKind.PHONE_ACCEL, [-sensor_reading.acceleration.v[2], -sensor_reading.acceleration.v[1], -sensor_reading.acceleration.v[0]])

  def handle_log(self, log):
    current_time = 1e-9*log.logMonoTime
    typ = log.which
    if typ in self.disabled_logs:
      return
    if typ == "sensorEvents":
      self.handle_sensors(log, current_time)
    elif typ == "gpsLocationExternal":
      self.handle_gps(log, current_time)
    elif typ == "carState":
      self.handle_car_state(log, current_time)
    elif typ == "ubloxGnss":
      self.handle_ublox_gnss(log, current_time)
    elif typ == "qcomGnss":
      self.handle_qcom_gnss(log, current_time)
    elif typ == "cameraOdometry":
      self.handle_cam_odo(log, current_time)

  def reset_kalman(self):
    self.filter_time = None
    self.filter_ready = False
    self.observation_buffer = []
    self.converter = None
    self.gyro_counter = 0
    self.acc_counter = 0
    self.raw_gnss_counter = 0
    self.speed_counter = 0


def locationd_thread(gctx, addr, disabled_logs):
  poller = zmq.Poller()

  #carstate = messaging.sub_sock('carState', poller, addr=addr, conflate=True)
  gpsLocationExternal = messaging.sub_sock('gpsLocationExternal', poller, addr=addr, conflate=True)
  ubloxGnss = messaging.sub_sock('ubloxGnss', poller, addr=addr, conflate=True)
  qcomGnss = messaging.sub_sock('qcomGnss', poller, addr=addr, conflate=True)
  sensorEvents = messaging.sub_sock('sensorEvents', poller, addr=addr, conflate=True)

  liveLocation = messaging.pub_sock('liveLocation')

  localizer = Localizer(disabled_logs=disabled_logs)
  print("init done")

  # buffer with all the messages that still need to be input into the kalman
  while 1:
    polld = poller.poll(timeout=1000)
    for sock, mode in polld:
      if mode != zmq.POLLIN:
        continue
      logs = messaging.drain_sock(sock)
      for log in logs:
        localizer.handle_log(log)

    if localizer.filter_ready and log.which == 'ubloxGnss':
      msg = messaging.new_message()
      msg.logMonoTime = log.logMonoTime
      msg.init('liveLocation')
      msg.liveLocation = localizer.liveLocationMsg(log.logMonoTime*1e-9)
      liveLocation.send(msg.to_bytes())


def main(gctx=None, addr="127.0.0.1"):
  IN_CAR = os.getenv("IN_CAR", False)
  disabled_logs = os.getenv("DISABLED_LOGS", "").split(",")
  # No speed for now
  disabled_logs.append('carState')
  if IN_CAR:
    addr = "192.168.5.11"
  locationd_thread(gctx, addr, disabled_logs)


if __name__ == "__main__":
  main()
