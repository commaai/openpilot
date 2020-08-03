#!/usr/bin/env python3
from cereal import car
from selfdrive.config import Conversions as CV
from selfdrive.car.volvo.values import CAR, PLATFORM, BUTTON_STATES
from selfdrive.car import STD_CARGO_KG, scale_rot_inertia, scale_tire_stiffness, gen_empty_fingerprint
from selfdrive.car.interfaces import CarInterfaceBase

EventName = car.CarEvent.EventName

class CarInterface(CarInterfaceBase):
  def __init__(self, CP, CarController, CarState):
    super().__init__(CP, CarController, CarState)
    
    # Create variables
    self.cruiseState_enabled_prev = False
    self.buttonStatesPrev = BUTTON_STATES.copy()
    
  @staticmethod
  def compute_gb(accel, speed):
    return float(accel) / 4.0

  @staticmethod
  def get_params(candidate, fingerprint=gen_empty_fingerprint(), has_relay=False, car_fw=[]):
    ret = CarInterfaceBase.get_std_params(candidate, fingerprint, has_relay)
    
    # Volvo port is a community feature, since we don't own one to test
    ret.communityFeature = True

    if candidate in PLATFORM.C1:
      ret.safetyModel = car.CarParams.SafetyModel.volvoC1
      
      if candidate == CAR.V40:
        # Technical specifications
        ret.mass = 1610 + STD_CARGO_KG
        ret.wheelbase = 2.647
        ret.centerToFront = ret.wheelbase * 0.44
        ret.steerRatio = 14.7 

    elif candidate in PLATFORM.EUCD:
      ret.safetyModel = car.CarParams.SafetyModel.volvoEUCD
      
      if candidate == CAR.V60:
        ret.mass = 1750 + STD_CARGO_KG  # All data found at https://www.media.volvocars.com/global/en-gb/models/old-v60/2014/specifications
        ret.wheelbase = 2.776 
        ret.centerToFront = ret.wheelbase * 0.44
        ret.steerRatio = 15

    # Common parameters
    ret.carName = "volvo"
    ret.enableCamera = True         # Will not set safety mode if not True
    ret.radarOffCan = True          # No radar objects on can

    # Steering settings - tuning parameters for lateral control.
    #ret.steerLimitAlert = True     # Do this do anything?
    ret.steerControlType = car.CarParams.SteerControlType.angle
    ret.minSteerSpeed = 1. * CV.KPH_TO_MS
    ret.steerRateCost = 1.          # Used in pathplanner for punishing? Steering derivative?
    ret.steerActuatorDelay = 0.24   # Actuator delay from input to output.
    
    # No PID control used. Set to a value, otherwise pid loop crashes.
    #ret.steerMaxBP = [0.] # m/s
    #ret.steerMaxV = [1.]
    ret.lateralTuning.pid.kpBP = [0.]
    ret.lateralTuning.pid.kiBP = [0.]
    # Tuning factors
    ret.lateralTuning.pid.kf = 0.0
    ret.lateralTuning.pid.kpV  = [0.0]
    ret.lateralTuning.pid.kiV = [0.0]
          
    # Assuming all is automatic
    ret.transmissionType = car.CarParams.TransmissionType.automatic
    
    # TODO: get actual value, for now starting with reasonable value for
    # civic and scaling by mass and wheelbase
    ret.rotationalInertia = scale_rot_inertia(ret.mass, ret.wheelbase)

    # TODO: start from empirically derived lateral slip stiffness for the civic and scale by
    # mass and CG position, so all cars will have approximately similar dyn behaviors
    ret.tireStiffnessFront, ret.tireStiffnessRear = scale_tire_stiffness(ret.mass, ret.wheelbase, ret.centerToFront)    

    return ret

  # returns a car.CarState
  def update(self, c, can_strings):
    canMonoTimes = []
    buttonEvents = []
   
    # Process the most recent CAN message traffic, and check for validity
    self.cp.update_strings(can_strings)
    self.cp_cam.update_strings(can_strings)
   
    ret = self.CS.update(self.cp, self.cp_cam)
    ret.canValid = self.cp.can_valid and self.cp_cam.can_valid
    
    # Check for and process state-change events (button press or release) from
    # the turn stalk switch or ACC steering wheel/control stalk buttons.
    for button in self.CS.buttonStates:
      if self.CS.buttonStates[button] != self.buttonStatesPrev[button]:
        be = car.CarState.ButtonEvent.new_message()
        be.type = button
        be.pressed = self.CS.buttonStates[button]
        buttonEvents.append(be)
    
    # Events 
    events = self.create_common_events(ret)
    
    # Engagement and longitudinal control using stock ACC. Make sure OP is
    # disengaged if stock ACC is disengaged.
    #if not ret.cruiseState.enabled:
    #  events.add(EventName.)
    # Attempt OP engagement only on rising edge of stock ACC engagement.
    #elif not self.cruiseState_enabled_prev:
    #  events.add(EventName.)

    ret.events = events.to_msg()
    ret.buttonEvents = buttonEvents
    ret.canMonoTimes = canMonoTimes

    # update previous values 
    self.gas_pressed_prev = ret.gasPressed
    self.cruiseState_enabled_prev = ret.cruiseState.enabled
    self.buttonStatesPrev = self.CS.buttonStates.copy()

    # cast to reader so it can't be modified
    self.CS.out = ret.as_reader()
    return self.CS.out

  def apply(self, c):
    can_sends = self.CC.update(c.enabled, self.CS, self.frame,
                               c.actuators, 
                               c.hudControl.visualAlert, c.hudControl.leftLaneVisible,
                               c.hudControl.rightLaneVisible, c.hudControl.leadVisible,
                               c.hudControl.leftLaneDepart, c.hudControl.rightLaneDepart)
    self.frame += 1
    return can_sends
