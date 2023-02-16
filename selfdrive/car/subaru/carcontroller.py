from opendbc.can.packer import CANPacker
from selfdrive.car import apply_std_steer_torque_limits
from selfdrive.car.subaru import subarucan
from selfdrive.car.subaru.values import DBC, GLOBAL_GEN2, PREGLOBAL_CARS, CarControllerParams


class CarController:
  def __init__(self, dbc_name, CP, VM):
    self.CP = CP
    self.apply_steer_last = 0
    self.frame = 0

    self.es_lkas_cnt = -1
    self.es_distance_cnt = -1
    self.es_dashstatus_cnt = -1
    self.cruise_button_prev = 0
    self.last_cancel_frame = 0

    self.p = CarControllerParams(CP)
    self.packer = CANPacker(DBC[CP.carFingerprint]['pt'])

  def update(self, CC, CS, now_nanos):
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

      if not CC.latActive:
        apply_steer = 0

      if self.CP.carFingerprint in PREGLOBAL_CARS:
        can_sends.append(subarucan.create_preglobal_steering_control(self.packer, apply_steer))
      else:
        can_sends.append(subarucan.create_steering_control(self.packer, apply_steer))

      self.apply_steer_last = apply_steer


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

    else:
      if pcm_cancel_cmd and (self.frame - self.last_cancel_frame) > 0.2:
        bus = 1 if self.CP.carFingerprint in GLOBAL_GEN2 else 0
        can_sends.append(subarucan.create_es_distance(self.packer, CS.es_distance_msg, bus, pcm_cancel_cmd))
        self.last_cancel_frame = self.frame

      if self.es_dashstatus_cnt != CS.es_dashstatus_msg["COUNTER"]:
        can_sends.append(subarucan.create_es_dashstatus(self.packer, CS.es_dashstatus_msg))
        self.es_dashstatus_cnt = CS.es_dashstatus_msg["COUNTER"]

      if self.es_lkas_cnt != CS.es_lkas_msg["COUNTER"]:
        can_sends.append(subarucan.create_es_lkas(self.packer, CS.es_lkas_msg, CC.enabled, hud_control.visualAlert,
                                                  hud_control.leftLaneVisible, hud_control.rightLaneVisible,
                                                  hud_control.leftLaneDepart, hud_control.rightLaneDepart))
        self.es_lkas_cnt = CS.es_lkas_msg["COUNTER"]

    new_actuators = actuators.copy()
    new_actuators.steer = self.apply_steer_last / self.p.STEER_MAX
    new_actuators.steerOutputCan = self.apply_steer_last

    self.frame += 1
    return new_actuators, can_sends
