#!/usr/bin/env python3
import os
import zmq
import math
import json

import numpy as np
from bisect import bisect_right

from cereal import car
from common.params import Params
from common.numpy_fast import clip
import cereal.messaging as messaging
from selfdrive.swaglog import cloudlog
from selfdrive.controls.lib.vehicle_model import VehicleModel
from cereal.services import service_list
from selfdrive.locationd.kalman.loc_local_kf import LocLocalKalman
from selfdrive.locationd.kalman.kalman_helpers import ObservationKind
from selfdrive.locationd.params_learner import ParamsLearner

DEBUG = False
kf = LocLocalKalman()  # Make sure that model is generated on import time


LEARNING_RATE = 3


class Localizer():
  def __init__(self, disabled_logs=None, dog=None):
    self.kf = LocLocalKalman()
    self.reset_kalman()

    self.sensor_data_t = 0.0
    self.max_age = .2  # seconds
    self.calibration_valid = False

    if disabled_logs is None:
      self.disabled_logs = list()
    else:
      self.disabled_logs = disabled_logs

  def reset_kalman(self):
    self.filter_time = None
    self.observation_buffer = []
    self.converter = None
    self.speed_counter = 0
    self.sensor_counter = 0

  def liveLocationMsg(self, time):
    fix = messaging.log.KalmanOdometry.new_message()

    predicted_state = self.kf.x
    fix.trans = [float(predicted_state[0]), float(predicted_state[1]), float(predicted_state[2])]
    fix.rot = [float(predicted_state[3]), float(predicted_state[4]), float(predicted_state[5])]

    return fix

  def update_kalman(self, time, kind, meas):
    idx = bisect_right([x[0] for x in self.observation_buffer], time)
    self.observation_buffer.insert(idx, (time, kind, meas))
    while self.observation_buffer[-1][0] - self.observation_buffer[0][0] > self.max_age:
      self.kf.predict_and_observe(*self.observation_buffer.pop(0))

  def handle_cam_odo(self, log, current_time):
    self.update_kalman(current_time, ObservationKind.CAMERA_ODO_ROTATION, np.concatenate([log.cameraOdometry.rot,
                                                                                          log.cameraOdometry.rotStd]))
    self.update_kalman(current_time, ObservationKind.CAMERA_ODO_TRANSLATION, np.concatenate([log.cameraOdometry.trans,
                                                                                             log.cameraOdometry.transStd]))

  def handle_controls_state(self, log, current_time):
    self.speed_counter += 1
    if self.speed_counter % 5 == 0:
      self.update_kalman(current_time, ObservationKind.ODOMETRIC_SPEED, np.array([log.controlsState.vEgo]))

  def handle_sensors(self, log, current_time):
    for sensor_reading in log.sensorEvents:
      # TODO does not yet account for double sensor readings in the log
      if sensor_reading.type == 4:
        self.sensor_counter += 1
        if self.sensor_counter % LEARNING_RATE == 0:
          self.update_kalman(current_time, ObservationKind.PHONE_GYRO, [-sensor_reading.gyro.v[2], -sensor_reading.gyro.v[1], -sensor_reading.gyro.v[0]])

  def handle_log(self, log):
    current_time = 1e-9 * log.logMonoTime
    typ = log.which
    if typ in self.disabled_logs:
      return
    if typ == "sensorEvents":
      self.sensor_data_t = current_time
      self.handle_sensors(log, current_time)
    elif typ == "controlsState":
      self.handle_controls_state(log, current_time)
    elif typ == "cameraOdometry":
      self.handle_cam_odo(log, current_time)


def locationd_thread(gctx, addr, disabled_logs):
  poller = zmq.Poller()

  controls_state_socket = messaging.sub_sock('controlsState', poller, addr=addr, conflate=True)
  sensor_events_socket = messaging.sub_sock('sensorEvents', poller, addr=addr, conflate=True)
  camera_odometry_socket = messaging.sub_sock('cameraOdometry', poller, addr=addr, conflate=True)

  kalman_odometry_socket = messaging.pub_sock('kalmanOdometry')
  live_parameters_socket = messaging.pub_sock('liveParameters')

  params_reader = Params()
  cloudlog.info("Parameter learner is waiting for CarParams")
  CP = car.CarParams.from_bytes(params_reader.get("CarParams", block=True))
  VM = VehicleModel(CP)
  cloudlog.info("Parameter learner got CarParams: %s" % CP.carFingerprint)

  params = params_reader.get("LiveParameters")

  # Check if car model matches
  if params is not None:
    params = json.loads(params)
    if (params.get('carFingerprint', None) != CP.carFingerprint) or (params.get('carVin', CP.carVin) != CP.carVin):
      cloudlog.info("Parameter learner found parameters for wrong car.")
      params = None

  if params is None:
    params = {
      'carFingerprint': CP.carFingerprint,
      'carVin': CP.carVin,
      'angleOffsetAverage': 0.0,
      'stiffnessFactor': 1.0,
      'steerRatio': VM.sR,
    }
    params_reader.put("LiveParameters", json.dumps(params))
    cloudlog.info("Parameter learner resetting to default values")

  cloudlog.info("Parameter starting with: %s" % str(params))
  localizer = Localizer(disabled_logs=disabled_logs)

  learner = ParamsLearner(VM,
                          angle_offset=params['angleOffsetAverage'],
                          stiffness_factor=params['stiffnessFactor'],
                          steer_ratio=params['steerRatio'],
                          learning_rate=LEARNING_RATE)

  i = 1
  while True:
    for socket, event in poller.poll(timeout=1000):
      log = messaging.recv_one(socket)
      localizer.handle_log(log)

      if socket is controls_state_socket:
        if not localizer.kf.t:
          continue

        if i % LEARNING_RATE == 0:
          # controlsState is not updating the Kalman Filter, so update KF manually
          localizer.kf.predict(1e-9 * log.logMonoTime)

          predicted_state = localizer.kf.x
          yaw_rate = -float(predicted_state[5])

          steering_angle = math.radians(log.controlsState.angleSteers)
          params_valid = learner.update(yaw_rate, log.controlsState.vEgo, steering_angle)

          log_t = 1e-9 * log.logMonoTime
          sensor_data_age = log_t - localizer.sensor_data_t

          params = messaging.new_message()
          params.init('liveParameters')
          params.liveParameters.valid = bool(params_valid)
          params.liveParameters.sensorValid = bool(sensor_data_age < 5.0)
          params.liveParameters.angleOffset = float(math.degrees(learner.ao))
          params.liveParameters.angleOffsetAverage = float(math.degrees(learner.slow_ao))
          params.liveParameters.stiffnessFactor = float(learner.x)
          params.liveParameters.steerRatio = float(learner.sR)
          live_parameters_socket.send(params.to_bytes())

        if i % 6000 == 0:   # once a minute
          params = learner.get_values()
          params['carFingerprint'] = CP.carFingerprint
          params['carVin'] = CP.carVin
          params_reader.put("LiveParameters", json.dumps(params))
          params_reader.put("ControlsParams", json.dumps({'angle_model_bias': log.controlsState.angleModelBias}))

        i += 1
      elif socket is camera_odometry_socket:
        msg = messaging.new_message()
        msg.init('kalmanOdometry')
        msg.logMonoTime = log.logMonoTime
        msg.kalmanOdometry = localizer.liveLocationMsg(log.logMonoTime * 1e-9)
        kalman_odometry_socket.send(msg.to_bytes())
      elif socket is sensor_events_socket:
        pass


def main(gctx=None, addr="127.0.0.1"):
  IN_CAR = os.getenv("IN_CAR", False)
  disabled_logs = os.getenv("DISABLED_LOGS", "").split(",")

  # No speed for now
  disabled_logs.append('controlsState')
  if IN_CAR:
    addr = "192.168.5.11"

  locationd_thread(gctx, addr, disabled_logs)


if __name__ == "__main__":
  main()
