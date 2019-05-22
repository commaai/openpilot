#!/usr/bin/env python
import os
import zmq
import math
import json

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from bisect import bisect_right

from cereal import car
from common.params import Params
from common.numpy_fast import clip
import selfdrive.messaging as messaging
from selfdrive.swaglog import cloudlog
from selfdrive.controls.lib.vehicle_model import VehicleModel
from selfdrive.services import service_list
from selfdrive.locationd.kalman.loc_local_kf import LocLocalKalman
from selfdrive.locationd.kalman.kalman_helpers import ObservationKind

DEBUG = False
kf = LocLocalKalman()  # Make sure that model is generated on import time

MAX_ANGLE_OFFSET = math.radians(10.)
MAX_ANGLE_OFFSET_TH = math.radians(9.)
MIN_STIFFNESS = 0.5
MAX_STIFFNESS = 2.0
MIN_SR = 0.5
MAX_SR = 2.0
MIN_SR_TH = 0.55
MAX_SR_TH = 1.9

LEARNING_RATE = 3

class Localizer(object):
  def __init__(self, disabled_logs=None, dog=None):
    self.kf = LocLocalKalman()
    self.reset_kalman()

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

  def handle_live100(self, log, current_time):
    self.speed_counter += 1
    if self.speed_counter % 5 == 0:
      self.update_kalman(current_time, ObservationKind.ODOMETRIC_SPEED, np.array([log.live100.vEgo]))

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
      self.handle_sensors(log, current_time)
    elif typ == "live100":
      self.handle_live100(log, current_time)
    elif typ == "cameraOdometry":
      self.handle_cam_odo(log, current_time)


class ParamsLearner(object):
  def __init__(self, VM, angle_offset=0., stiffness_factor=1.0, steer_ratio=None, learning_rate=1.0):
    self.VM = VM

    self.ao = math.radians(angle_offset)
    self.slow_ao = math.radians(angle_offset)
    self.x = stiffness_factor
    self.sR = VM.sR if steer_ratio is None else steer_ratio
    self.MIN_SR = MIN_SR * self.VM.sR
    self.MAX_SR = MAX_SR * self.VM.sR
    self.MIN_SR_TH = MIN_SR_TH * self.VM.sR
    self.MAX_SR_TH = MAX_SR_TH * self.VM.sR

    self.alpha1 = 0.01 * learning_rate
    self.alpha2 = 0.0005 * learning_rate
    self.alpha3 = 0.1 * learning_rate
    self.alpha4 = 1.0 * learning_rate

  def get_values(self):
    return {
      'angleOffsetAverage': math.degrees(self.slow_ao),
      'stiffnessFactor': self.x,
      'steerRatio': self.sR,
    }

  def update(self, psi, u, sa):
    cF0 = self.VM.cF
    cR0 = self.VM.cR
    aR = self.VM.aR
    aF = self.VM.aF
    l = self.VM.l
    m = self.VM.m

    x = self.x
    ao = self.ao
    sR = self.sR

    # Gradient descent:  learn angle offset, tire stiffness and steer ratio.
    if u > 10.0 and abs(math.degrees(sa)) < 15.:
      self.ao -= self.alpha1 * 2.0*cF0*cR0*l*u*x*(1.0*cF0*cR0*l*u*x*(ao - sa) + psi*sR*(cF0*cR0*l**2*x - m*u**2*(aF*cF0 - aR*cR0)))/(sR**2*(cF0*cR0*l**2*x - m*u**2*(aF*cF0 - aR*cR0))**2)

      ao = self.slow_ao
      self.slow_ao -= self.alpha2 * 2.0*cF0*cR0*l*u*x*(1.0*cF0*cR0*l*u*x*(ao - sa) + psi*sR*(cF0*cR0*l**2*x - m*u**2*(aF*cF0 - aR*cR0)))/(sR**2*(cF0*cR0*l**2*x - m*u**2*(aF*cF0 - aR*cR0))**2)

      self.x -= self.alpha3 * -2.0*cF0*cR0*l*m*u**3*(ao - sa)*(aF*cF0 - aR*cR0)*(1.0*cF0*cR0*l*u*x*(ao - sa) + psi*sR*(cF0*cR0*l**2*x - m*u**2*(aF*cF0 - aR*cR0)))/(sR**2*(cF0*cR0*l**2*x - m*u**2*(aF*cF0 - aR*cR0))**3)

      self.sR -= self.alpha4 * -2.0*cF0*cR0*l*u*x*(ao - sa)*(1.0*cF0*cR0*l*u*x*(ao - sa) + psi*sR*(cF0*cR0*l**2*x - m*u**2*(aF*cF0 - aR*cR0)))/(sR**3*(cF0*cR0*l**2*x - m*u**2*(aF*cF0 - aR*cR0))**2)

    if DEBUG:
      # s1 = "Measured yaw rate % .6f" % psi
      # ao = 0.
      # s2 = "Uncompensated yaw % .6f" % (1.0*u*(-ao + sa)/(l*sR*(1 - m*u**2*(aF*cF0*x - aR*cR0*x)/(cF0*cR0*l**2*x**2))))
      # instant_ao = aF*m*psi*sR*u/(cR0*l*x) - aR*m*psi*sR*u/(cF0*l*x) - l*psi*sR/u + sa
      s4 = "Instant AO: % .2f Avg. AO % .2f" % (math.degrees(self.ao), math.degrees(self.slow_ao))
      s5 = "Stiffnes: % .3f x" % self.x
      print("{0} {1}".format(s4, s5))


    self.ao = clip(self.ao, -MAX_ANGLE_OFFSET, MAX_ANGLE_OFFSET)
    self.slow_ao = clip(self.slow_ao, -MAX_ANGLE_OFFSET, MAX_ANGLE_OFFSET)
    self.x = clip(self.x, MIN_STIFFNESS, MAX_STIFFNESS)
    self.sR = clip(self.sR, self.MIN_SR, self.MAX_SR)

    # don't check stiffness for validity, as it can change quickly if sR is off
    valid = abs(self.slow_ao) < MAX_ANGLE_OFFSET_TH and \
      self.sR > self.MIN_SR_TH and self.sR < self.MAX_SR_TH

    return valid


def locationd_thread(gctx, addr, disabled_logs):
  ctx = zmq.Context()
  poller = zmq.Poller()

  live100_socket = messaging.sub_sock(ctx, service_list['live100'].port, poller, addr=addr, conflate=True)
  sensor_events_socket = messaging.sub_sock(ctx, service_list['sensorEvents'].port, poller, addr=addr, conflate=True)
  camera_odometry_socket = messaging.sub_sock(ctx, service_list['cameraOdometry'].port, poller, addr=addr, conflate=True)

  kalman_odometry_socket = messaging.pub_sock(ctx, service_list['kalmanOdometry'].port)
  live_parameters_socket = messaging.pub_sock(ctx, service_list['liveParameters'].port)

  params_reader = Params()
  cloudlog.info("Parameter learner is waiting for CarParams")
  CP = car.CarParams.from_bytes(params_reader.get("CarParams", block=True))
  VM = VehicleModel(CP)
  cloudlog.info("Parameter learner got CarParams: %s" % CP.carFingerprint)

  params = params_reader.get("LiveParameters")

  # Check if car model matches
  if params is not None:
    params = json.loads(params)
    if params.get('carFingerprint', None) != CP.carFingerprint:
      cloudlog.info("Parameter learner found parameters for wrong car.")
      params = None

  if params is None:
    params = {
      'carFingerprint': CP.carFingerprint,
      'angleOffsetAverage': 0.0,
      'stiffnessFactor': 1.0,
      'steerRatio': VM.sR,
    }
    cloudlog.info("Parameter learner resetting to default values")

  cloudlog.info("Parameter starting with: %s" % str(params))
  localizer = Localizer(disabled_logs=disabled_logs)

  learner = ParamsLearner(VM,
                          angle_offset=params['angleOffsetAverage'],
                          stiffness_factor=params['stiffnessFactor'],
                          steer_ratio=params['steerRatio'],
                          learning_rate=LEARNING_RATE)

  i = 0
  while True:
    for socket, event in poller.poll(timeout=1000):
      log = messaging.recv_one(socket)
      localizer.handle_log(log)

      if socket is live100_socket:
        if not localizer.kf.t:
          continue

        if i % LEARNING_RATE == 0:
          # live100 is not updating the Kalman Filter, so update KF manually
          localizer.kf.predict(1e-9 * log.logMonoTime)

          predicted_state = localizer.kf.x
          yaw_rate = -float(predicted_state[5])

          steering_angle = math.radians(log.live100.angleSteers)
          params_valid = learner.update(yaw_rate, log.live100.vEgo, steering_angle)

          params = messaging.new_message()
          params.init('liveParameters')
          params.liveParameters.valid = bool(params_valid)
          params.liveParameters.angleOffset = float(math.degrees(learner.ao))
          params.liveParameters.angleOffsetAverage = float(math.degrees(learner.slow_ao))
          params.liveParameters.stiffnessFactor = float(learner.x)
          params.liveParameters.steerRatio = float(learner.sR)
          live_parameters_socket.send(params.to_bytes())

        if i % 6000 == 0:   # once a minute
          params = learner.get_values()
          params['carFingerprint'] = CP.carFingerprint
          params_reader.put("LiveParameters", json.dumps(params))
          params_reader.put("ControlsParams", json.dumps({'angle_model_bias': log.live100.angleModelBias}))

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
  disabled_logs.append('live100')
  if IN_CAR:
    addr = "192.168.5.11"

  locationd_thread(gctx, addr, disabled_logs)


if __name__ == "__main__":
  main()
