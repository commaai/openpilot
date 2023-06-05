from common.numpy_fast import clip
from opendbc.can.packer import CANPacker
from selfdrive.car import apply_driver_steer_torque_limits
from selfdrive.car.subaru import subarucan
from selfdrive.car.subaru.values import DBC, GEN2_ES_BUTTONS_DID, GLOBAL_GEN2, PREGLOBAL_CARS, CarControllerParams, SubaruFlags

ACCEL_HYST_GAP = 10  # don't change accel command for small oscilalitons within this value

def accel_hysteresis(accel, accel_steady):

  # for small accel oscillations within ACCEL_HYST_GAP, don't change the accel command
  if accel > accel_steady + ACCEL_HYST_GAP:
    accel_steady = accel - ACCEL_HYST_GAP
  elif accel < accel_steady - ACCEL_HYST_GAP:
    accel_steady = accel + ACCEL_HYST_GAP
  accel = accel_steady

  return accel, accel_steady

def compute_gb(accel):
  return clip(accel/4.0, 0.0, 1.0), clip(-accel/4.0, 0.0, 1.0)

class CarController:
  def __init__(self, dbc_name, CP, VM):
    self.CP = CP
    self.apply_steer_last = 0
    self.frame = 0

    self.cruise_button_prev = 0
    self.steer_rate_limited = False
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

    if self.CP.flags & SubaruFlags.GEN2_DISABLE_FWD_CAMERA.value:
      addr = 0x787 # fwd camera address
      bus = 2
    
      # Tester present (keeps eyesight disabled)
      if self.frame % 100 == 0:
        can_sends.append([addr, 0, b"\x02\x3E\x80\x00\x00\x00\x00\x00", bus])
      
      # read button data request
      if self.frame % 1 == 0:
        can_sends.append([addr, 0, b'\x03\x22' + GEN2_ES_BUTTONS_DID + b'\x00\x00\x00\x00', bus])

    # *** steering ***
    if (self.frame % self.p.STEER_STEP) == 0:
      apply_steer = int(round(actuators.steer * self.p.STEER_MAX))

      # limits due to driver torque

      new_steer = int(round(apply_steer))
      apply_steer = apply_driver_steer_torque_limits(new_steer, self.apply_steer_last, CS.out.steeringTorque, self.p)

      if not CC.latActive:
        apply_steer = 0

      if self.CP.carFingerprint in PREGLOBAL_CARS:
        can_sends.append(subarucan.create_preglobal_steering_control(self.packer, apply_steer, 0))
      else:
        can_sends.append(subarucan.create_steering_control(self.packer, apply_steer, 0))

      self.apply_steer_last = apply_steer

    ### LONG ###

    cruise_rpm = 0
    cruise_throttle = 0

    brake_cmd = False
    brake_value = 0

    body_bus = 1 if self.CP.carFingerprint in GLOBAL_GEN2 else 0

    if self.CP.openpilotLongitudinalControl:

      gas, brake = compute_gb(actuators.accel)

      if CC.longActive and brake > 0:
        brake_value = clip(int(brake * CarControllerParams.BRAKE_SCALE), CarControllerParams.BRAKE_MIN, CarControllerParams.BRAKE_MAX)
        brake_cmd = True

      # AEB passthrough
      if CC.enabled and CS.out.stockAeb:
        brake_cmd = False

      if CC.longActive and gas > 0:
        # limit min and max values
        cruise_throttle = clip(int(CarControllerParams.THROTTLE_BASE + (gas * CarControllerParams.THROTTLE_SCALE)), CarControllerParams.THROTTLE_MIN, CarControllerParams.THROTTLE_MAX)
        cruise_rpm = clip(int(CarControllerParams.RPM_BASE + (gas * CarControllerParams.RPM_SCALE)), CarControllerParams.RPM_MIN, CarControllerParams.RPM_MAX)
        # hysteresis
        cruise_throttle, self.throttle_steady = accel_hysteresis(cruise_throttle, self.throttle_steady)
        cruise_rpm, self.rpm_steady = accel_hysteresis(cruise_rpm, self.rpm_steady)

        # slow down the signals change
        cruise_throttle = clip(cruise_throttle, self.cruise_throttle_last - CarControllerParams.THROTTLE_DELTA, self.cruise_throttle_last + CarControllerParams.THROTTLE_DELTA)
        cruise_rpm = clip(cruise_rpm, self.cruise_rpm_last - CarControllerParams.RPM_DELTA, self.cruise_rpm_last + CarControllerParams.RPM_DELTA)

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

        can_sends.append(subarucan.create_preglobal_es_distance(self.packer, cruise_button, CS.es_distance_msg, 0))

    else:
      if self.frame % 10 == 0:
        can_sends.append(subarucan.create_es_dashstatus(self.packer, CS.es_dashstatus_msg, CC.enabled, CC.longActive, hud_control, CS.out.cruiseState.available))
        can_sends.append(subarucan.create_es_lkas_state(self.packer, CS.es_lkas_state_msg, CC.enabled, hud_control, CS.out.cruiseState.available))

        if self.CP.flags & SubaruFlags.SEND_INFOTAINMENT:
          can_sends.append(subarucan.create_infotainmentstatus(self.packer, CS.es_infotainmentstatus_msg, hud_control))
      
      if self.frame % 10 == 0 or len(CS.buttonEvents):
        low_speed = CS.out.vEgoRaw < 9
        can_sends.append(subarucan.create_es_distance(self.packer, CS.es_distance_msg, pcm_cancel_cmd, CC.longActive, brake_cmd, brake_value, cruise_throttle, CS.cruise_buttons[-1], low_speed, body_bus))

      if self.CP.openpilotLongitudinalControl:
        if self.frame % 5 == 0:
          can_sends.append(subarucan.create_es_status(self.packer, CS.es_status_msg, CC.longActive, cruise_rpm, body_bus))
          can_sends.append(subarucan.create_es_brake(self.packer, CS.es_brake_msg, CC.enabled, brake_cmd, brake_value, body_bus))
      
        if self.frame % 2 == 0:
          can_sends.append(subarucan.create_brake_status(self.packer, CS.brake_status_msg, CS.out.stockAeb, 2))

    # TODO: what are these
    if self.frame % 5 == 0:
      can_sends.append(subarucan.create_unknown_1(self.packer, 0))

    if self.frame % 10 == 0:
      can_sends.append(subarucan.create_unknown_2(self.packer, 0))
    
    if self.frame % 2 == 0:
      can_sends.append(subarucan.create_unknown_3(self.packer, 0))

    new_actuators = actuators.copy()
    new_actuators.steer = self.apply_steer_last / self.p.STEER_MAX
    new_actuators.steerOutputCan = self.apply_steer_last

    self.frame += 1
    return new_actuators, can_sends
