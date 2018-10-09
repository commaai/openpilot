#!/usr/bin/env python
from common.realtime import sec_since_boot
from cereal import car, log
from selfdrive.swaglog import cloudlog
from selfdrive.config import Conversions as CV
from selfdrive.controls.lib.drive_helpers import EventTypes as ET, create_event
from selfdrive.controls.lib.vehicle_model import VehicleModel
from selfdrive.car.ford.carstate import CarState, get_can_parser
from selfdrive.car.ford.fordcan import MAX_ANGLE

try:
  from selfdrive.car.ford.carcontroller import CarController
except ImportError:
  CarController = None


class CarInterface(object):
  def __init__(self, CP, sendcan=None):
    self.CP = CP
    self.VM = VehicleModel(CP)

    self.frame = 0
    self.can_invalid_count = 0
    self.gas_pressed_prev = False
    self.brake_pressed_prev = False
    self.cruise_enabled_prev = False

    # *** init the major players ***
    self.CS = CarState(CP)

    self.cp = get_can_parser(CP)

    # sending if read only is False
    if sendcan is not None:
      self.sendcan = sendcan
      self.CC = CarController(self.cp.dbc_name, CP.enableCamera, self.VM)

  @staticmethod
  def compute_gb(accel, speed):
    return float(accel) / 3.0

  @staticmethod
  def calc_accel_override(a_ego, a_target, v_ego, v_target):
    return 1.0

  @staticmethod
  def get_params(candidate, fingerprint):

    # kg of standard extra cargo to count for drive, gas, etc...
    std_cargo = 136

    ret = car.CarParams.new_message()

    ret.carName = "ford"
    ret.carFingerprint = candidate

    ret.safetyModel = car.CarParams.SafetyModels.ford

    # pedal
    ret.enableCruise = True

    # FIXME: hardcoding honda civic 2016 touring params so they can be used to
    # scale unknown params for other cars
    mass_civic = 2923. * CV.LB_TO_KG + std_cargo
    wheelbase_civic = 2.70
    centerToFront_civic = wheelbase_civic * 0.4
    centerToRear_civic = wheelbase_civic - centerToFront_civic
    rotationalInertia_civic = 2500
    tireStiffnessFront_civic = 85400
    tireStiffnessRear_civic = 90000

    ret.steerKiBP, ret.steerKpBP = [[0.], [0.]]
    ret.wheelbase = 2.85
    ret.steerRatio = 14.8
    ret.mass = 3045. * CV.LB_TO_KG + std_cargo
    ret.steerKpV, ret.steerKiV = [[0.01], [0.005]]     # TODO: tune this
    ret.steerKf = 1. / MAX_ANGLE   # MAX Steer angle to normalize FF
    ret.steerActuatorDelay = 0.1  # Default delay, not measured yet
    ret.steerRateCost = 1.0

    f = 1.2
    tireStiffnessFront_civic *= f
    tireStiffnessRear_civic *= f

    ret.centerToFront = ret.wheelbase * 0.44

    ret.longPidDeadzoneBP = [0., 9.]
    ret.longPidDeadzoneV = [0., .15]

    # min speed to enable ACC. if car can do stop and go, then set enabling speed
    # to a negative value, so it won't matter.
    ret.minEnableSpeed = -1.

    centerToRear = ret.wheelbase - ret.centerToFront
    # TODO: get actual value, for now starting with reasonable value for
    # civic and scaling by mass and wheelbase
    ret.rotationalInertia = rotationalInertia_civic * \
                            ret.mass * ret.wheelbase**2 / (mass_civic * wheelbase_civic**2)

    # TODO: start from empirically derived lateral slip stiffness for the civic and scale by
    # mass and CG position, so all cars will have approximately similar dyn behaviors
    ret.tireStiffnessFront = tireStiffnessFront_civic * \
                             ret.mass / mass_civic * \
                             (centerToRear / ret.wheelbase) / (centerToRear_civic / wheelbase_civic)
    ret.tireStiffnessRear = tireStiffnessRear_civic * \
                            ret.mass / mass_civic * \
                            (ret.centerToFront / ret.wheelbase) / (centerToFront_civic / wheelbase_civic)

    # no rear steering, at least on the listed cars above
    ret.steerRatioRear = 0.
    ret.steerControlType = car.CarParams.SteerControlType.angle

    # steer, gas, brake limitations VS speed
    ret.steerMaxBP = [0.]  # breakpoints at 1 and 40 kph
    ret.steerMaxV = [1.0]  # 2/3rd torque allowed above 45 kph
    ret.gasMaxBP = [0.]
    ret.gasMaxV = [0.5]
    ret.brakeMaxBP = [5., 20.]
    ret.brakeMaxV = [1., 0.8]

    ret.enableCamera = not any(x for x in [970, 973, 984] if x in fingerprint)
    cloudlog.warn("ECU Camera Simulated: %r", ret.enableCamera)

    ret.steerLimitAlert = False
    ret.stoppingControl = False
    ret.startAccel = 0.0

    ret.longitudinalKpBP = [0., 5., 35.]
    ret.longitudinalKpV = [3.6, 2.4, 1.5]
    ret.longitudinalKiBP = [0., 35.]
    ret.longitudinalKiV = [0.54, 0.36]

    return ret

  # returns a car.CarState
  def update(self, c):
    # ******************* do can recv *******************
    canMonoTimes = []

    self.cp.update(int(sec_since_boot() * 1e9), False)

    self.CS.update(self.cp)

    # create message
    ret = car.CarState.new_message()

    # speeds
    ret.vEgo = self.CS.v_ego
    ret.vEgoRaw = self.CS.v_ego_raw
    ret.standstill = self.CS.standstill
    ret.wheelSpeeds.fl = self.CS.v_wheel_fl
    ret.wheelSpeeds.fr = self.CS.v_wheel_fr
    ret.wheelSpeeds.rl = self.CS.v_wheel_rl
    ret.wheelSpeeds.rr = self.CS.v_wheel_rr

    # steering wheel
    ret.steeringAngle = self.CS.angle_steers
    ret.steeringPressed = self.CS.steer_override

    # gas pedal
    ret.gas = self.CS.user_gas / 100.
    ret.gasPressed = self.CS.user_gas > 0.0001
    ret.brakePressed = self.CS.brake_pressed
    ret.brakeLights = self.CS.brake_lights

    ret.cruiseState.enabled = not (self.CS.pcm_acc_status in [0, 3])
    ret.cruiseState.speed = self.CS.v_cruise_pcm
    ret.cruiseState.available = self.CS.pcm_acc_status != 0

    ret.genericToggle = self.CS.generic_toggle

    # events
    events = []
    if not self.CS.can_valid:
      self.can_invalid_count += 1
      if self.can_invalid_count >= 5:
        events.append(create_event('commIssue', [ET.NO_ENTRY, ET.IMMEDIATE_DISABLE]))
    else:
      self.can_invalid_count = 0

    if self.CS.steer_error:
      events.append(create_event('steerUnavailable', [ET.NO_ENTRY, ET.IMMEDIATE_DISABLE, ET.PERMANENT]))

    # enable request in prius is simple, as we activate when Toyota is active (rising edge)
    if ret.cruiseState.enabled and not self.cruise_enabled_prev:
      events.append(create_event('pcmEnable', [ET.ENABLE]))
    elif not ret.cruiseState.enabled:
      events.append(create_event('pcmDisable', [ET.USER_DISABLE]))

    # disable on pedals rising edge or when brake is pressed and speed isn't zero
    if (ret.gasPressed and not self.gas_pressed_prev) or \
       (ret.brakePressed and (not self.brake_pressed_prev or ret.vEgo > 0.001)):
      events.append(create_event('pedalPressed', [ET.NO_ENTRY, ET.USER_DISABLE]))

    if ret.gasPressed:
      events.append(create_event('pedalPressed', [ET.PRE_ENABLE]))

    if self.CS.lkas_state not in [2, 3] and ret.vEgo > 13.* CV.MPH_TO_MS and ret.cruiseState.enabled:
      events.append(create_event('steerTempUnavailableMute', [ET.WARNING]))

    ret.events = events
    ret.canMonoTimes = canMonoTimes

    self.gas_pressed_prev = ret.gasPressed
    self.brake_pressed_prev = ret.brakePressed
    self.cruise_enabled_prev = ret.cruiseState.enabled

    return ret.as_reader()

  # pass in a car.CarControl
  # to be called @ 100hz
  def apply(self, c, perception_state=log.Live20Data.new_message()):

    self.CC.update(self.sendcan, c.enabled, self.CS, self.frame, c.actuators,
                   c.hudControl.visualAlert, c.cruiseControl.cancel)

    self.frame += 1
    return False
