#!/usr/bin/env python3
from cereal import car
from selfdrive.config import Conversions as CV
from selfdrive.swaglog import cloudlog
import cereal.messaging as messaging
from selfdrive.car import gen_empty_fingerprint
from selfdrive.car.interfaces import CarInterfaceBase

# mocked car interface to work with chffrplus
TS = 0.01  # 100Hz
YAW_FR = 0.2 # ~0.8s time constant on yaw rate filter
# low pass gain
LPG = 2 * 3.1415 * YAW_FR * TS / (1 + 2 * 3.1415 * YAW_FR * TS)


class CarInterface(CarInterfaceBase):
  def __init__(self, CP, CarController):
    self.CP = CP
    self.CC = CarController

    cloudlog.debug("Using Mock Car Interface")

    # TODO: subscribe to phone sensor
    self.sensor = messaging.sub_sock('sensorEvents')
    self.gps = messaging.sub_sock('gpsLocation')

    self.speed = 0.
    self.prev_speed = 0.
    self.yaw_rate = 0.
    self.yaw_rate_meas = 0.

  @staticmethod
  def compute_gb(accel, speed):
    return accel

  @staticmethod
  def get_params(candidate, fingerprint=gen_empty_fingerprint(), vin="", has_relay=False):

    ret = car.CarParams.new_message()

    ret.carName = "mock"
    ret.carFingerprint = candidate

    ret.safetyModel = car.CarParams.SafetyModel.noOutput
    ret.openpilotLongitudinalControl = False

    # FIXME: hardcoding honda civic 2016 touring params so they can be used to
    # scale unknown params for other cars
    ret.mass = 1700.
    ret.rotationalInertia = 2500.
    ret.wheelbase = 2.70
    ret.centerToFront = ret.wheelbase * 0.5
    ret.steerRatio = 13. # reasonable
    ret.tireStiffnessFront = 1e6    # very stiff to neglect slip
    ret.tireStiffnessRear = 1e6     # very stiff to neglect slip
    ret.steerRatioRear = 0.

    ret.steerMaxBP = [0.]
    ret.steerMaxV = [0.]  # 2/3rd torque allowed above 45 kph
    ret.gasMaxBP = [0.]
    ret.gasMaxV = [0.]
    ret.brakeMaxBP = [0.]
    ret.brakeMaxV = [0.]

    ret.longitudinalTuning.kpBP = [0.]
    ret.longitudinalTuning.kpV = [0.]
    ret.longitudinalTuning.kiBP = [0.]
    ret.longitudinalTuning.kiV = [0.]
    ret.longitudinalTuning.deadzoneBP = [0.]
    ret.longitudinalTuning.deadzoneV = [0.]
    ret.steerActuatorDelay = 0.

    return ret

  # returns a car.CarState
  def update(self, c, can_strings):
    # get basic data from phone and gps since CAN isn't connected
    sensors = messaging.recv_sock(self.sensor)
    if sensors is not None:
      for sensor in sensors.sensorEvents:
        if sensor.type == 4:  # gyro
          self.yaw_rate_meas = -sensor.gyro.v[0]

    gps = messaging.recv_sock(self.gps)
    if gps is not None:
      self.prev_speed = self.speed
      self.speed = gps.gpsLocation.speed

    # create message
    ret = car.CarState.new_message()

    # speeds
    ret.vEgo = self.speed
    ret.vEgoRaw = self.speed
    a = self.speed - self.prev_speed

    ret.aEgo = a
    ret.brakePressed = a < -0.5

    self.yawRate = LPG * self.yaw_rate_meas + (1. - LPG) * self.yaw_rate
    ret.yawRate = self.yaw_rate
    ret.standstill = self.speed < 0.01
    ret.wheelSpeeds.fl = self.speed
    ret.wheelSpeeds.fr = self.speed
    ret.wheelSpeeds.rl = self.speed
    ret.wheelSpeeds.rr = self.speed
    curvature = self.yaw_rate / max(self.speed, 1.)
    ret.steeringAngle = curvature * self.CP.steerRatio * self.CP.wheelbase * CV.RAD_TO_DEG

    events = []
    ret.events = events

    return ret.as_reader()

  def apply(self, c):
    # in mock no carcontrols
    return []
