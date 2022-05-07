from opendbc.can.packer import CANPacker
from selfdrive.car import apply_std_steer_torque_limits
from selfdrive.car.subaru import subarucan
from selfdrive.car.subaru.values import DBC, PREGLOBAL_CARS, CarControllerParams


class CarController:
  def __init__(self, dbc_name, CP, VM):
    self.CP = CP
    self.apply_steer_last = 0
    self.es_distance_cnt = -1
    self.es_lkas_cnt = -1
    self.throttle_cnt = -1
    self.cruise_button_prev = 0
    self.steer_rate_limited = False
    self.frame = 0
    self.sng_acc_resume = False
    self.sng_acc_resume_cnt = -1
    self.manual_hold = False
    self.prev_cruise_state = 0
    self.prev_close_distance = 0

    self.p = CarControllerParams(CP)
    self.packer = CANPacker(DBC[CP.carFingerprint]['pt'])

  def update(self, CC, CS):
    actuators = CC.actuators
    hud_control = CC.hudControl
    pcm_cancel_cmd = CC.cruiseControl.cancel

    can_sends = []

    # *** steering ***
    if (self.frame % self.p.STEER_STEP) == 0:

      apply_steer = int(round(actuators.steer * self.p.STEER_MAX))

      # limits due to driver torque

      new_steer = int(round(apply_steer))
      apply_steer = apply_std_steer_torque_limits(new_steer, self.apply_steer_last, CS.out.steeringTorque, self.p)
      self.steer_rate_limited = new_steer != apply_steer

      if not CC.latActive:
        apply_steer = 0

      if self.CP.carFingerprint in PREGLOBAL_CARS:
        can_sends.append(subarucan.create_preglobal_steering_control(self.packer, apply_steer, self.frame, self.p.STEER_STEP))
      else:
        can_sends.append(subarucan.create_steering_control(self.packer, apply_steer, self.frame, self.p.STEER_STEP))

      self.apply_steer_last = apply_steer

    # *** stop and go ***

    throttle_cmd = False

    if CS.CP.carFingerprint in PREGLOBAL_CARS:
      if (c.enabled                                            # ACC active
          and CS.car_follow == 1                             # lead car
          and CS.out.standstill                              # must be standing still
          and CS.close_distance > 3                          # acc resume min trigger threshold (m)
          and CS.close_distance < 4.5                        # acc resume max trigger threshold (m)
          and CS.close_distance > self.prev_close_distance): # distance with lead car is increasing
        self.sng_acc_resume = True
    elif CS.CP.carFingerprint not in PREGLOBAL_CARS:
      # Record manual hold set while in standstill and no car in front
      if CS.out.standstill and self.prev_cruise_state == 1 and CS.cruise_state == 3 and CS.car_follow == 0:
        self.manual_hold = True
      # Cancel manual hold when car starts moving
      if not CS.out.standstill:
        self.manual_hold = False
      if (c.enabled                                            # ACC active
          and not self.manual_hold
          and CS.car_follow == 1                             # lead car
          and CS.cruise_state == 3                           # ACC HOLD (only with EPB)
          and CS.out.standstill                              # must be standing still
          and CS.close_distance > 3                          # acc resume min trigger threshold (m)
          and CS.close_distance < 4.5                        # acc resume max trigger threshold (m)
          and CS.close_distance > self.prev_close_distance): # distance with lead car is increasing
        self.sng_acc_resume = True
      self.prev_cruise_state = CS.cruise_state

    if self.sng_acc_resume:
      if self.sng_acc_resume_cnt < 5:
        throttle_cmd = True
        self.sng_acc_resume_cnt += 1
      else:
        self.sng_acc_resume = False
        self.sng_acc_resume_cnt = -1

    # Cancel ACC if stopped, brake pressed and not stopped behind another car
    if c.enabled and CS.out.brakePressed and CS.car_follow == 0 and CS.out.standstill:
      pcm_cancel_cmd = True

    self.prev_close_distance = CS.close_distance

    # *** alerts and pcm cancel ***

    if self.CP.carFingerprint in PREGLOBAL_CARS:
      if self.es_distance_cnt != CS.es_distance_msg["COUNTER"]:
        # 1 = main, 2 = set shallow, 3 = set deep, 4 = resume shallow, 5 = resume deep
        # disengage ACC when OP is disengaged
        if pcm_cancel_cmd:
          cruise_button = 1
        # turn main on if off and past start-up state
        elif not CS.out.cruiseState.available and CS.ready:
          cruise_button = 1
        else:
          cruise_button = CS.cruise_button

        # unstick previous mocked button press
        if cruise_button == 1 and self.cruise_button_prev == 1:
          cruise_button = 0
        self.cruise_button_prev = cruise_button

        can_sends.append(subarucan.create_preglobal_es_distance(self.packer, cruise_button, CS.es_distance_msg))
        self.es_distance_cnt = CS.es_distance_msg["COUNTER"]

      if self.throttle_cnt != CS.throttle_msg["COUNTER"]:
         can_sends.append(subarucan.create_preglobal_throttle(self.packer, CS.throttle_msg, throttle_cmd))
         self.throttle_cnt = CS.throttle_msg["COUNTER"]

    else:
      if self.es_distance_cnt != CS.es_distance_msg["COUNTER"]:
        can_sends.append(subarucan.create_es_distance(self.packer, CS.es_distance_msg, pcm_cancel_cmd))
        self.es_distance_cnt = CS.es_distance_msg["COUNTER"]

      if self.es_lkas_cnt != CS.es_lkas_msg["COUNTER"]:
        can_sends.append(subarucan.create_es_lkas(self.packer, CS.es_lkas_msg, CC.enabled, hud_control.visualAlert, hud_control.leftLaneVisible, hud_control.rightLaneVisible, hud_control.leftLaneDepart, hud_control.rightLaneDepart))
        self.es_lkas_cnt = CS.es_lkas_msg["COUNTER"]

      if self.throttle_cnt != CS.throttle_msg["COUNTER"]:
         can_sends.append(subarucan.create_throttle(self.packer, CS.throttle_msg, throttle_cmd))
         self.throttle_cnt = CS.throttle_msg["COUNTER"]

    new_actuators = actuators.copy()
    new_actuators.steer = self.apply_steer_last / self.p.STEER_MAX

    self.frame += 1
    return new_actuators, can_sends
