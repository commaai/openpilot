from common.numpy_fast import clip
from opendbc.can.packer import CANPacker
from selfdrive.car import apply_driver_steer_torque_limits, apply_hysteresis
from selfdrive.car.subaru import subarucan
from selfdrive.car.subaru.values import DBC, GLOBAL_GEN2, PREGLOBAL_CARS, CanBus, CarControllerParams, SubaruFlags
from selfdrive.controls.lib.drive_helpers import rate_limit


def compute_gb(accel):
  return clip(accel/4.0, 0.0, 1.0), clip(-accel/4.0, 0.0, 1.0)

class CarController:
  def __init__(self, dbc_name, CP, VM):
    self.CP = CP
    self.apply_steer_last = 0
    self.frame = 0

    self.cruise_button_prev = 0
    self.cruise_rpm_last = 0
    self.cruise_throttle_last = 0
    self.rpm_steady = 0
    self.throttle_steady = 0

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
      apply_steer = apply_driver_steer_torque_limits(new_steer, self.apply_steer_last, CS.out.steeringTorque, self.p)

      if not CC.latActive:
        apply_steer = 0

      if self.CP.carFingerprint in PREGLOBAL_CARS:
        can_sends.append(subarucan.create_preglobal_steering_control(self.packer, apply_steer, CC.latActive))
      else:
        can_sends.append(subarucan.create_steering_control(self.packer, apply_steer, CC.latActive))

      self.apply_steer_last = apply_steer

    ### LONG ###

    cruise_rpm = 0
    cruise_throttle = 0

    brake_cmd = False
    brake_value = 0

    if self.CP.openpilotLongitudinalControl:

      gas, brake = compute_gb(actuators.accel)

      if CC.longActive and brake > 0:
        brake_value = clip(int(brake * CarControllerParams.BRAKE_SCALE), CarControllerParams.BRAKE_MIN, CarControllerParams.BRAKE_MAX)
        brake_cmd = True

      # AEB passthrough
      if CC.enabled and CS.out.stockAeb:
        brake_cmd = False

      if CC.longActive and gas > 0:
        # calculate desired values
        cruise_throttle = int(CarControllerParams.THROTTLE_BASE + (gas * CarControllerParams.THROTTLE_SCALE))
        cruise_rpm = int(CarControllerParams.RPM_BASE + (gas * CarControllerParams.RPM_SCALE))

        # limit min and max values
        cruise_throttle = clip(cruise_throttle, CarControllerParams.THROTTLE_MIN, CarControllerParams.THROTTLE_MAX)
        cruise_rpm = clip(cruise_rpm, CarControllerParams.RPM_MIN, CarControllerParams.RPM_MAX)

        # hysteresis
        cruise_throttle = apply_hysteresis(cruise_throttle, self.throttle_steady, CarControllerParams.THROTTLE_RPM_HYST)
        cruise_rpm = apply_hysteresis(cruise_rpm, self.rpm_steady, CarControllerParams.THROTTLE_RPM_HYST)

        self.throttle_steady = cruise_throttle
        self.rpm_steady = cruise_rpm

        # rate limiting
        cruise_throttle = rate_limit(cruise_throttle, self.cruise_throttle_last, CarControllerParams.THROTTLE_DELTA, CarControllerParams.THROTTLE_DELTA)
        cruise_rpm = rate_limit(cruise_rpm, self.cruise_rpm_last, CarControllerParams.RPM_DELTA, CarControllerParams.RPM_DELTA)

        self.cruise_throttle_last = cruise_throttle
        self.cruise_rpm_last = cruise_rpm

    # *** alerts and pcm cancel ***
    if self.CP.carFingerprint in PREGLOBAL_CARS:
      if self.frame % 5 == 0:
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

    else:
      if self.frame % 10 == 0:
        can_sends.append(subarucan.create_es_dashstatus(self.packer, CS.es_dashstatus_msg, CC.enabled, CC.longActive, hud_control.leadVisible))

        can_sends.append(subarucan.create_es_lkas_state(self.packer, CS.es_lkas_state_msg, CC.enabled, hud_control.visualAlert,
                                                        hud_control.leftLaneVisible, hud_control.rightLaneVisible,
                                                        hud_control.leftLaneDepart, hud_control.rightLaneDepart))

        if self.CP.flags & SubaruFlags.SEND_INFOTAINMENT:
          can_sends.append(subarucan.create_es_infotainment(self.packer, CS.es_infotainment_msg, hud_control.visualAlert))

      if self.CP.openpilotLongitudinalControl:
        if self.frame % 5 == 0:
          can_sends.append(subarucan.create_es_status(self.packer, CS.es_status_msg, CC.longActive, cruise_rpm))
          can_sends.append(subarucan.create_es_brake(self.packer, CS.es_brake_msg, CC.enabled, brake_cmd, brake_value))
          can_sends.append(subarucan.create_cruise_control(self.packer, CS.cruise_control_msg))

        if self.frame % 2 == 0:
          can_sends.append(subarucan.create_brake_status(self.packer, CS.brake_status_msg, CS.out.stockAeb))

        if self.frame % 10 == 0:
          can_sends.append(subarucan.create_es_distance(self.packer, CS.es_distance_msg, 0, pcm_cancel_cmd, CC.longActive, brake_cmd, brake_value, cruise_throttle))

      else:
        if pcm_cancel_cmd:
          bus = CanBus.alt if self.CP.carFingerprint in GLOBAL_GEN2 else CanBus.main
          can_sends.append(subarucan.create_es_distance(self.packer, CS.es_distance_msg, bus, pcm_cancel_cmd, CC.longActive, brake_cmd, brake_value, cruise_throttle))
          self.last_cancel_frame = self.frame

    new_actuators = actuators.copy()
    new_actuators.steer = self.apply_steer_last / self.p.STEER_MAX
    new_actuators.steerOutputCan = self.apply_steer_last

    self.frame += 1
    return new_actuators, can_sends
