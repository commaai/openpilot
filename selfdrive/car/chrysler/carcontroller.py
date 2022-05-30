from cereal import car
from opendbc.can.packer import CANPacker
from selfdrive.car import apply_toyota_steer_torque_limits
from selfdrive.car.chrysler.chryslercan import create_lkas_hud, create_lkas_command, create_wheel_buttons
from selfdrive.car.chrysler.values import CAR, CarControllerParams, STEER_MAX_LOOKUP, STEER_DELTA_UP, STEER_DELTA_DOWN


class CarController:
  def __init__(self, dbc_name, CP, VM):
    self.CP = CP
    self.apply_steer_last = 0
    self.frame = 0
    self.prev_lkas_frame = -1
    self.hud_count = 0
    self.car_fingerprint = CP.carFingerprint
    self.gone_fast_yet = False
    self.steer_rate_limited = False
    self.lkasdisabled = 0
    self.lkaslast_frame = 0.
    self.gone_fast_yet_previous = False
    #self.CarControllerParams = CarControllerParams
    CarControllerParams.STEER_MAX = STEER_MAX_LOOKUP.get(CP.carFingerprint, 1.)
    CarControllerParams.STEER_DELTA_UP = STEER_DELTA_UP.get(CP.carFingerprint, 1.) 
    CarControllerParams.STEER_DELTA_DOWN = STEER_DELTA_DOWN.get(CP.carFingerprint, 1.) 

    self.packer = CANPacker(dbc_name)

  def update(self, CC, CS):
    # this seems needed to avoid steering faults and to force the sync with the EPS counter
    if self.prev_lkas_frame == CS.lkas_counter:
      new_actuators = CC.actuators.copy()
      new_actuators.steer = self.apply_steer_last / CarControllerParams.STEER_MAX
      return new_actuators, []

    actuators = CC.actuators

    # steer torque
    new_steer = int(round(actuators.steer * CarControllerParams.STEER_MAX))
    apply_steer = apply_toyota_steer_torque_limits(new_steer, self.apply_steer_last,
                                                   CS.out.steeringTorqueEps, CarControllerParams)
    self.steer_rate_limited = new_steer != apply_steer

    #moving_fast = CS.out.vEgo > self.CP.minSteerSpeed  # for status message

    if self.car_fingerprint not in (CAR.RAM_1500, CAR.RAM_2500):
      if CS.out.vEgo > (self.CP.minSteerSpeed - 0.5):  # for command high bit
        self.gone_fast_yet = 1 #2 means LKAS enabled
      elif self.car_fingerprint in (CAR.PACIFICA_2019_HYBRID, CAR.PACIFICA_2020, CAR.JEEP_CHEROKEE_2019):
        if CS.out.vEgo < (self.CP.minSteerSpeed - 3.0):
          self.gone_fast_yet = 0  # < 14.5m/s stock turns off this bit, but fine down to 13.5
          
    elif self.car_fingerprint in (CAR.RAM_1500, CAR.RAM_2500):
      if CS.out.vEgo > (self.CP.minSteerSpeed):  # for command high bit
        self.gone_fast_yet = 2 #2 means LKAS enabled
      elif CS.out.vEgo < (self.CP.minSteerSpeed - 0.5):
        self.gone_fast_yet = 0   
      #self.gone_fast_yet = CS.out.vEgo > self.CP.minSteerSpeed

    if self.gone_fast_yet_previous > 0 and self.gone_fast_yet == 0:
        self.lkaslast_frame = self.frame

    #lkas_active = moving_fast and CC.enabled

    #if CS.out.steerError is True: #possible fix for LKAS error Plan to test
    #  gone_fast_yet = False

    if (CS.out.steerFaultPermanent is True) or (CS.lkasdisabled is 1) or (self.frame-self.lkaslast_frame<400):#If the LKAS Control bit is toggled too fast it can create and LKAS error
      self.gone_fast_yet = 0

    lkas_active = self.gone_fast_yet and CC.enabled

    if not lkas_active or self.gone_fast_yet_previous == 0:
      apply_steer = 0

    self.apply_steer_last = apply_steer

    self.gone_fast_yet_previous = self.gone_fast_yet

    can_sends = []

    #*** control msgs ***

    if CC.cruiseControl.cancel:
      can_sends.append(create_wheel_buttons(self.packer, CS.button_counter + 1, self.car_fingerprint, cancel=True, acc_resume = False))
    elif CS.out.cruiseState.standstill:
      can_sends.append(create_wheel_buttons(self.packer, CS.button_counter + 1, self.car_fingerprint, cancel=False, acc_resume = True))

    # LKAS_HEARTBIT is forwarded by Panda so no need to send it here.
    # frame is 50Hz (0.02s period) #Becuase we skip every other frame
    if self.frame % 12 == 0:  # 0.25s period #must be 12 to acheive .25s instead of 25 because we skip every other frame
      if CS.lkas_car_model != -1:
        can_sends.append(create_lkas_hud(self.packer, lkas_active, CC.hudControl.visualAlert, self.hud_count, CS, self.car_fingerprint))
        self.hud_count += 1

    can_sends.append(create_lkas_command(self.packer, int(apply_steer), self.gone_fast_yet, CS.lkas_counter))

    self.frame += 1
    self.prev_lkas_frame = CS.lkas_counter

    new_actuators = actuators.copy()
    new_actuators.steer = apply_steer / CarControllerParams.STEER_MAX

    return new_actuators, can_sends
