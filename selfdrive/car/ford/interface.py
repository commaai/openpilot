#!/usr/bin/env python3
from cereal import car
from selfdrive.swaglog import cloudlog
from selfdrive.config import Conversions as CV
from selfdrive.controls.lib.drive_helpers import EventTypes as ET, create_event
from selfdrive.car.ford.values import MAX_ANGLE, Ecu, ECU_FINGERPRINT, FINGERPRINTS
from selfdrive.car import STD_CARGO_KG, scale_rot_inertia, scale_tire_stiffness, is_ecu_disconnected, gen_empty_fingerprint
from selfdrive.car.interfaces import CarInterfaceBase


class CarInterface(CarInterfaceBase):

  @staticmethod
  def compute_gb(accel, speed):
    return float(accel) / 3.0

  @staticmethod
  def get_params(candidate, fingerprint=gen_empty_fingerprint(), has_relay=False, car_fw=[]):
    ret = CarInterfaceBase.get_std_params(candidate, fingerprint, has_relay)
    ret.carName = "ford"
    ret.communityFeature = True                              
    ret.safetyModel = car.CarParams.SafetyModel.ford
    ret.dashcamOnly = False

    ret.wheelbase = 2.85
    ret.steerRatio = 14.8
    ret.mass = 3045. * CV.LB_TO_KG + STD_CARGO_KG
    ret.lateralTuning.pid.kiBP, ret.lateralTuning.pid.kpBP = [[0.], [0.]]
    ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.03], [0.005]]     # TODO: tune this
    ret.lateralTuning.pid.kf = 1. / MAX_ANGLE   # MAX Steer angle to normalize FF
    ret.steerActuatorDelay = 0.5  # Default delay, not measured yet
    ret.steerLimitTimer = 0.8
    ret.steerRateCost = 2.0
    ret.centerToFront = ret.wheelbase * 0.44
    tire_stiffness_factor = 0.5328

    # TODO: get actual value, for now starting with reasonable value for
    # civic and scaling by mass and wheelbase
    ret.rotationalInertia = scale_rot_inertia(ret.mass, ret.wheelbase)

    # TODO: start from empirically derived lateral slip stiffness for the civic and scale by
    # mass and CG position, so all cars will have approximately similar dyn behaviors
    ret.tireStiffnessFront, ret.tireStiffnessRear = scale_tire_stiffness(ret.mass, ret.wheelbase, ret.centerToFront,
                                                                         tire_stiffness_factor=tire_stiffness_factor)

    ret.steerControlType = car.CarParams.SteerControlType.angle

    ret.enableCamera = is_ecu_disconnected(fingerprint[0], FINGERPRINTS, ECU_FINGERPRINT, candidate, Ecu.fwdCamera) or has_relay
    cloudlog.warning("ECU Camera Simulated: %r", ret.enableCamera)

    return ret

  # returns a car.CarState
  def update(self, c, can_strings):
    # ******************* do can recv *******************
    self.cp.update_strings(can_strings)

    ret = self.CS.update(self.cp)

    #ret = car.CarState.new_message()               
    ret.canValid = self.cp.can_valid

    # speeds
    #ret.vEgo = self.CS.v_ego
    #ret.aEgo = self.CS.a_ego
    #ret.vEgoRaw = self.CS.v_ego_raw
    #ret.yawRate = self.VM.yaw_rate(self.CS.angle_steers * CV.DEG_TO_RAD, self.CS.v_ego)
    #ret.standstill = self.CS.standstill
    #ret.wheelSpeeds.fl = self.CS.v_wheel_fl
    #ret.wheelSpeeds.fr = self.CS.v_wheel_fr
    #ret.wheelSpeeds.rl = self.CS.v_wheel_rl
    #ret.wheelSpeeds.rr = self.CS.v_wheel_rr

    # steering wheel
    #ret.steeringAngle = self.CS.angle_steers
    #ret.steeringPressed = self.CS.steer_override

    # gas pedal
    #ret.gas = self.CS.user_gas / 100.
    #ret.gasPressed = self.CS.user_gas > 0.0001
    #ret.brakePressed = self.CS.brake_pressed
    #ret.brakeLights = self.CS.brake_lights

    #ret.cruiseState.enabled = not (self.CS.pcm_acc_status in [0, 3])
    #ret.cruiseState.speed = self.CS.v_cruise_pcm
    #ret.cruiseState.available = self.CS.pcm_acc_status != 0
    #ret.cruiseState.speedOffset = 0.

    #ret.genericToggle = self.CS.generic_toggle
    
    # blinkers
    #ret.leftBlinker = self.CS.left_blinker_on
    #ret.rightBlinker = self.CS.right_blinker_on

    # doors
    #ret.doorOpen = self.CS.door_open

    # button events
    buttonEvents = []

    # blinkers
    #if self.CS.left_blinker_on != self.CS.prev_left_blinker_on:
    #  be = car.CarState.ButtonEvent.new_message()
    #  be.type = 'leftBlinker'
    #  be.pressed = self.CS.left_blinker_on
    #  buttonEvents.append(be)

    #if self.CS.right_blinker_on != self.CS.prev_right_blinker_on:
    #  be = car.CarState.ButtonEvent.new_message()
    #  be.type = 'rightBlinker'
    #  be.pressed = self.CS.right_blinker_on
    #  buttonEvents.append(be)

    ret.buttonEvents = buttonEvents        
    # events
    events = self.create_common_events(ret)

    if self.CS.lkas_state not in [2, 3] and ret.vEgo > 13.* CV.MPH_TO_MS and ret.cruiseState.enabled:
      events.append(create_event('steerTempUnavailableMute', [ET.WARNING]))
      print ("steerTempUnavailableMute!!!")
    ret.events = events

    self.CS.out = ret.as_reader()
    return self.CS.out

  # pass in a car.CarControl
  # to be called @ 100hz
  def apply(self, c):

    can_sends = self.CC.update(c.enabled, self.CS, self.frame, c.actuators,
                               c.hudControl.visualAlert, c.cruiseControl.cancel)

    self.frame += 1
    return can_sends
