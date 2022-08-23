from opendbc.can.packer import CANPacker
from common.realtime import DT_CTRL
from selfdrive.car import apply_toyota_steer_torque_limits
from selfdrive.car.chrysler.chryslercan import create_lkas_hud, create_lkas_command, create_cruise_buttons, acc_command
from selfdrive.car.chrysler.values import CAR, RAM_CARS, RAM_DT, RAM_HD, CarControllerParams
from cereal import car
from common.conversions import Conversions as CV
from common.numpy_fast import clip
from common.op_params import opParams, ACCEL_FACTOR, DECEL_FACTOR

GearShifter = car.CarState.GearShifter

# braking
BRAKE_CHANGE = 0.06
ACCEL_MIN = -3.5

class CarController:
  def __init__(self, dbc_name, CP, VM, OP=None):
    self.CP = CP
    self.apply_steer_last = 0
    self.frame = 0

    self.hud_count = 0
    self.last_lkas_falling_edge = 0
    self.lkas_control_bit_prev = False
    self.last_button_frame = 0

    self.packer = CANPacker(dbc_name)
    self.params = CarControllerParams(CP)

    # long
    self.last_brake = None
    self.vehicleMass = CP.mass
    self.max_gear = None
    self.last_torque = 0
    self.spoofspeed = 0

    if OP is None:
      OP = opParams()
    self.op_params = OP
    self.factor = self.op_params.get(ACCEL_FACTOR)


  def update(self, CC, CS):
    can_sends = []

    lkas_active = CC.latActive and self.lkas_control_bit_prev

    # cruise buttons
    if (CS.button_counter != self.last_button_frame):
      das_bus = 2 if self.CP.carFingerprint in RAM_CARS else 0
      self.last_button_frame = CS.button_counter
      if self.CP.carFingerprint in RAM_CARS:
        can_sends.append(create_cruise_buttons(self.packer, CS.button_counter, das_bus, CS.cruise_buttons, cancel=CC.cruiseControl.cancel, resume=CC.cruiseControl.resume))

       # ACC cancellation
      elif CC.cruiseControl.cancel:
        can_sends.append(create_cruise_buttons(self.packer, CS.button_counter+1, das_bus, CS.cruise_buttons, cancel=True))

      # ACC resume from standstill
      elif CC.cruiseControl.resume:
        can_sends.append(create_cruise_buttons(self.packer, CS.button_counter+1, das_bus, CS.cruise_buttons, resume=True))

    # HUD alerts
    if self.frame % 25 == 0:
      if CS.lkas_car_model != -1:
        can_sends.append(create_lkas_hud(self.packer, self.CP, lkas_active, CC.hudControl.visualAlert, self.hud_count, CS.lkas_car_model, CS.auto_high_beam))
        self.hud_count += 1

    # steering
    if self.frame % 2 == 0:

      # TODO: can we make this more sane? why is it different for all the cars?
      lkas_control_bit = self.lkas_control_bit_prev
      if self.CP.carFingerprint in RAM_DT:
        if CS.out.vEgo >= self.CP.minEnableSpeed:
          lkas_control_bit = True
        if (self.CP.minEnableSpeed >= 14.5)  and (CS.out.gearShifter != GearShifter.drive) :
          lkas_control_bit = False
      elif CS.out.vEgo > self.CP.minSteerSpeed:
        lkas_control_bit = True
      elif self.CP.carFingerprint in (CAR.PACIFICA_2019_HYBRID, CAR.PACIFICA_2020, CAR.JEEP_CHEROKEE_2019):
        if CS.out.vEgo < (self.CP.minSteerSpeed - 3.0):
          lkas_control_bit = False
      elif self.CP.carFingerprint in RAM_HD:
        if CS.out.vEgo < (self.CP.minSteerSpeed - 0.5):
          lkas_control_bit = False

      # EPS faults if LKAS re-enables too quickly
      lkas_control_bit = lkas_control_bit and (self.frame - self.last_lkas_falling_edge > 200) and not CS.out.steerFaultTemporary and not CS.out.steerFaultPermanent

      if not lkas_control_bit and self.lkas_control_bit_prev:
        self.last_lkas_falling_edge = self.frame
      self.lkas_control_bit_prev = lkas_control_bit

      # steer torque
      new_steer = int(round(CC.actuators.steer * self.params.STEER_MAX))
      apply_steer = apply_toyota_steer_torque_limits(new_steer, self.apply_steer_last, CS.out.steeringTorqueEps, self.params)
      if not lkas_active or not lkas_control_bit:
        apply_steer = 0
      self.apply_steer_last = apply_steer

      can_sends.append(create_lkas_command(self.packer, self.CP, int(apply_steer), lkas_control_bit))

      #LONG
      das_3_counter = CS.das_3['COUNTER']
      max_gear = 0

      if not CC.enabled:
        self.last_brake = None
      
      max_gear = 8
      if CC.actuators.accel <= 0:
        accel_req = False
        decel_req = False
        torque = None
        decel = self.acc_brake(CC.actuators.accel)
        max_gear = 6
        delta_accel = 0
        self.last_torque = CS.engineTorque
      else:
        self.last_brake = None
        accel_req = True
        decel_req = False
        delta_accel = CC.actuators.accel - CS.out.aEgo
        
        # adding a factor to the velocity. 1.0 multiplied by the delta_accel assumes that we are telling the formula we want to be x m/s faster than we currently
        # are 1 second from now. I add the ability to modify this in real time to dial it in better. 
        # For example if comma is requesting 2.0m/s2 acceleration and we use a 1.0 multiplier, then we are saying in 1 second we want to be traveling 2.0m/s faster. 
        # if we put a 1.5 multiplier then it becomes 3.0 m/s faster 1 second from now. 
        if delta_accel < 0: 
          velocity = clip((self.op_params.get(DECEL_FACTOR) * abs(delta_accel)),  -3.5, 2.0)
        else:
          velocity = clip((self.op_params.get(ACCEL_FACTOR) * abs(delta_accel)),  -3.5, 2.0)

        torque = (self.vehicleMass * delta_accel * velocity)  / (.105 *  CS.engineRpm)
        if CS.engineTorque < 0 and torque > 0:
          torque +=200 #A truck will idle between 100-200nm So, If torque is very negative we need to get back to positive quickly. 
        else:
          torque += CS.engineTorque

        decel = None

      self.last_torque = torque

      can_sends.append(acc_command(self.packer, das_3_counter, CC.enabled,
                                   accel_req,
                                   torque,
                                   max_gear,
                                   decel_req,
                                   decel,
                                   CS.das_3))

    self.frame += 1

    new_actuators = CC.actuators.copy()
    new_actuators.steer = self.apply_steer_last / self.params.STEER_MAX

    return new_actuators, can_sends

  def acc_brake(self, aTarget):
    brake_target = max(ACCEL_MIN, round(aTarget, 2))
    if self.last_brake is None:
      self.last_brake = min(0., brake_target / 2)
    else:
      tBrake = brake_target
      lBrake = self.last_brake
      if tBrake < lBrake:
        diff = min(BRAKE_CHANGE, (lBrake - tBrake) / 2)
        self.last_brake = max(lBrake - diff, tBrake)
      elif tBrake - lBrake > 0.01:  # don't let up unless it's a big enough jump
        diff = min(BRAKE_CHANGE, (tBrake - lBrake) / 2)
        self.last_brake = min(lBrake + diff, tBrake)
    return self.last_brake
