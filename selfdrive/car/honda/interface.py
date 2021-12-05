#!/usr/bin/env python3
from cereal import car
from panda import Panda
from common.numpy_fast import interp
from common.params import Params
from selfdrive.car.honda.values import CarControllerParams, CruiseButtons, Flag, CAR, HONDA_BOSCH, HONDA_NIDEC_ALT_SCM_MESSAGES, HONDA_BOSCH_ALT_BRAKE_SIGNAL
from selfdrive.car import STD_CARGO_KG, CivicParams, scale_rot_inertia, scale_tire_stiffness, gen_empty_fingerprint, get_safety_config
from selfdrive.car.interfaces import CarInterfaceBase
from selfdrive.car.disable_ecu import disable_ecu
from selfdrive.config import Conversions as CV


ButtonType = car.CarState.ButtonEvent.Type
EventName = car.CarEvent.EventName
TransmissionType = car.CarParams.TransmissionType


class CarInterface(CarInterfaceBase):
  @staticmethod
  def get_pid_accel_limits(CP, current_speed, cruise_speed):
    if CP.carFingerprint in HONDA_BOSCH:
      return CarControllerParams.BOSCH_ACCEL_MIN, CarControllerParams.BOSCH_ACCEL_MAX
    else:
      # NIDECs don't allow acceleration near cruise_speed,
      # so limit limits of pid to prevent windup
      ACCEL_MAX_VALS = [CarControllerParams.NIDEC_ACCEL_MAX, 0.2]
      ACCEL_MAX_BP = [cruise_speed - 2., cruise_speed - .2]
      return CarControllerParams.NIDEC_ACCEL_MIN, interp(current_speed, ACCEL_MAX_BP, ACCEL_MAX_VALS)

  @staticmethod
  def get_params(candidate, fingerprint=gen_empty_fingerprint(), car_fw=[]):  # pylint: disable=dangerous-default-value
    ret = CarInterfaceBase.get_std_params(candidate, fingerprint)
    ret.carName = "honda"

    if candidate in HONDA_BOSCH:
      ret.safetyConfigs = [get_safety_config(car.CarParams.SafetyModel.hondaBosch)]
      ret.radarOffCan = True

      # Disable the radar and let openpilot control longitudinal
      # WARNING: THIS DISABLES AEB!
      ret.openpilotLongitudinalControl = Params().get_bool("DisableRadar")

      ret.pcmCruise = not ret.openpilotLongitudinalControl
    else:
      ret.safetyConfigs = [get_safety_config(car.CarParams.SafetyModel.hondaNidec)]
      ret.enableGasInterceptor = 0x201 in fingerprint[0]
      ret.openpilotLongitudinalControl = True

      ret.pcmCruise = not ret.enableGasInterceptor
      ret.communityFeature = ret.enableGasInterceptor

    if candidate == CAR.CRV_5G:
      ret.enableBsm = 0x12f8bfa7 in fingerprint[0]

    # Detect Bosch cars with new HUD msgs
    if any(0x33DA in f for f in fingerprint.values()):
      ret.flags |= Flag.BOSCH_EXT_HUD.value

    # Accord 1.5T CVT has different gearbox message
    if candidate == CAR.ACCORD and 0x191 in fingerprint[1]:
      ret.transmissionType = TransmissionType.cvt

    # Certain Hondas have an extra steering sensor at the bottom of the steering rack,
    # which improves controls quality as it removes the steering column torsion from feedback.
    # Tire stiffness factor fictitiously lower if it includes the steering column torsion effect.
    # For modeling details, see p.198-200 in "The Science of Vehicle Dynamics (2014), M. Guiggiani"
    ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0], [0]]
    ret.lateralTuning.pid.kiBP, ret.lateralTuning.pid.kpBP = [[0.], [0.]]
    ret.lateralTuning.pid.kf = 0.00006  # conservative feed-forward

    if candidate in HONDA_BOSCH:
      ret.longitudinalTuning.kpV = [0.25]
      ret.longitudinalTuning.kiV = [0.05]
      ret.longitudinalActuatorDelayUpperBound = 0.5 # s
    else:
      # default longitudinal tuning for all hondas
      ret.longitudinalTuning.kpBP = [0., 5., 35.]
      ret.longitudinalTuning.kpV = [1.2, 0.8, 0.5]
      ret.longitudinalTuning.kiBP = [0., 35.]
      ret.longitudinalTuning.kiV = [0.18, 0.12]

    eps_modified = False
    for fw in car_fw:
      if fw.ecu == "eps" and b"," in fw.fwVersion:
        eps_modified = True

    if candidate == CAR.CIVIC:
      stop_and_go = True
      ret.mass = CivicParams.MASS
      ret.wheelbase = CivicParams.WHEELBASE
      ret.centerToFront = CivicParams.CENTER_TO_FRONT
      ret.steerRatio = 15.38  # 10.93 is end-to-end spec
      if eps_modified:
        # stock request input values:     0x0000, 0x00DE, 0x014D, 0x01EF, 0x0290, 0x0377, 0x0454, 0x0610, 0x06EE
        # stock request output values:    0x0000, 0x0917, 0x0DC5, 0x1017, 0x119F, 0x140B, 0x1680, 0x1680, 0x1680
        # modified request output values: 0x0000, 0x0917, 0x0DC5, 0x1017, 0x119F, 0x140B, 0x1680, 0x2880, 0x3180
        # stock filter output values:     0x009F, 0x0108, 0x0108, 0x0108, 0x0108, 0x0108, 0x0108, 0x0108, 0x0108
        # modified filter output values:  0x009F, 0x0108, 0x0108, 0x0108, 0x0108, 0x0108, 0x0108, 0x0400, 0x0480
        # note: max request allowed is 4096, but request is capped at 3840 in firmware, so modifications result in 2x max
        ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 2560, 8000], [0, 2560, 3840]]
        ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.3], [0.1]]
      else:
        ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 2560], [0, 2560]]
        ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[1.1], [0.33]]
      tire_stiffness_factor = 1.

    elif candidate in (CAR.CIVIC_BOSCH, CAR.CIVIC_BOSCH_DIESEL):
      stop_and_go = True
      ret.mass = CivicParams.MASS
      ret.wheelbase = CivicParams.WHEELBASE
      ret.centerToFront = CivicParams.CENTER_TO_FRONT
      ret.steerRatio = 15.38  # 10.93 is end-to-end spec
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]  # TODO: determine if there is a dead zone at the top end
      tire_stiffness_factor = 1.
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.8], [0.24]]

    elif candidate in (CAR.ACCORD, CAR.ACCORD_2021, CAR.ACCORDH, CAR.ACCORDH_2021):
      stop_and_go = True
      ret.mass = 3279. * CV.LB_TO_KG + STD_CARGO_KG
      ret.wheelbase = 2.83
      ret.centerToFront = ret.wheelbase * 0.39
      ret.steerRatio = 16.33  # 11.82 is spec end-to-end
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]  # TODO: determine if there is a dead zone at the top end
      tire_stiffness_factor = 0.8467

      if eps_modified:
        ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.3], [0.09]]
      else:
        ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.6], [0.18]]

    elif candidate == CAR.ACURA_ILX:
      stop_and_go = False
      ret.mass = 3095. * CV.LB_TO_KG + STD_CARGO_KG
      ret.wheelbase = 2.67
      ret.centerToFront = ret.wheelbase * 0.37
      ret.steerRatio = 18.61  # 15.3 is spec end-to-end
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 3840], [0, 3840]]  # TODO: determine if there is a dead zone at the top end
      tire_stiffness_factor = 0.72
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.8], [0.24]]

    elif candidate in (CAR.CRV, CAR.CRV_EU):
      stop_and_go = False
      ret.mass = 3572. * CV.LB_TO_KG + STD_CARGO_KG
      ret.wheelbase = 2.62
      ret.centerToFront = ret.wheelbase * 0.41
      ret.steerRatio = 16.89  # as spec
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 1000], [0, 1000]]  # TODO: determine if there is a dead zone at the top end
      tire_stiffness_factor = 0.444
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.8], [0.24]]
      ret.wheelSpeedFactor = 1.025

    elif candidate == CAR.CRV_5G:
      stop_and_go = True
      ret.mass = 3410. * CV.LB_TO_KG + STD_CARGO_KG
      ret.wheelbase = 2.66
      ret.centerToFront = ret.wheelbase * 0.41
      ret.steerRatio = 16.0  # 12.3 is spec end-to-end
      if eps_modified:
        # stock request input values:     0x0000, 0x00DB, 0x01BB, 0x0296, 0x0377, 0x0454, 0x0532, 0x0610, 0x067F
        # stock request output values:    0x0000, 0x0500, 0x0A15, 0x0E6D, 0x1100, 0x1200, 0x129A, 0x134D, 0x1400
        # modified request output values: 0x0000, 0x0500, 0x0A15, 0x0E6D, 0x1100, 0x1200, 0x1ACD, 0x239A, 0x2800
        ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 2560, 10000], [0, 2560, 3840]]
        ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.21], [0.07]]
      else:
        ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 3840], [0, 3840]]
        ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.64], [0.192]]
      tire_stiffness_factor = 0.677
      ret.wheelSpeedFactor = 1.025

    elif candidate == CAR.CRV_HYBRID:
      stop_and_go = True
      ret.mass = 1667. + STD_CARGO_KG  # mean of 4 models in kg
      ret.wheelbase = 2.66
      ret.centerToFront = ret.wheelbase * 0.41
      ret.steerRatio = 16.0  # 12.3 is spec end-to-end
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]  # TODO: determine if there is a dead zone at the top end
      tire_stiffness_factor = 0.677
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.6], [0.18]]
      ret.wheelSpeedFactor = 1.025

    elif candidate == CAR.FIT:
      stop_and_go = False
      ret.mass = 2644. * CV.LB_TO_KG + STD_CARGO_KG
      ret.wheelbase = 2.53
      ret.centerToFront = ret.wheelbase * 0.39
      ret.steerRatio = 13.06
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]  # TODO: determine if there is a dead zone at the top end
      tire_stiffness_factor = 0.75
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.2], [0.05]]

    elif candidate == CAR.FREED:
      stop_and_go = False
      ret.mass = 3086. * CV.LB_TO_KG + STD_CARGO_KG
      ret.wheelbase = 2.74
      # the remaining parameters were copied from FIT
      ret.centerToFront = ret.wheelbase * 0.39
      ret.steerRatio = 13.06
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]
      tire_stiffness_factor = 0.75
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.2], [0.05]]

    elif candidate == CAR.HRV:
      stop_and_go = False
      ret.mass = 3125 * CV.LB_TO_KG + STD_CARGO_KG
      ret.wheelbase = 2.61
      ret.centerToFront = ret.wheelbase * 0.41
      ret.steerRatio = 15.2
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]
      tire_stiffness_factor = 0.5
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.16], [0.025]]
      ret.wheelSpeedFactor = 1.025

    elif candidate == CAR.ACURA_RDX:
      stop_and_go = False
      ret.mass = 3935. * CV.LB_TO_KG + STD_CARGO_KG
      ret.wheelbase = 2.68
      ret.centerToFront = ret.wheelbase * 0.38
      ret.steerRatio = 15.0  # as spec
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 1000], [0, 1000]]  # TODO: determine if there is a dead zone at the top end
      tire_stiffness_factor = 0.444
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.8], [0.24]]

    elif candidate == CAR.ACURA_RDX_3G:
      stop_and_go = True
      ret.mass = 4068. * CV.LB_TO_KG + STD_CARGO_KG
      ret.wheelbase = 2.75
      ret.centerToFront = ret.wheelbase * 0.41
      ret.steerRatio = 11.95  # as spec
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 3840], [0, 3840]]
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.6], [0.18]]
      tire_stiffness_factor = 0.677

    elif candidate == CAR.ODYSSEY:
      stop_and_go = False
      ret.mass = 4471. * CV.LB_TO_KG + STD_CARGO_KG
      ret.wheelbase = 3.00
      ret.centerToFront = ret.wheelbase * 0.41
      ret.steerRatio = 14.35  # as spec
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]  # TODO: determine if there is a dead zone at the top end
      tire_stiffness_factor = 0.82
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.28], [0.08]]

    elif candidate == CAR.ODYSSEY_CHN:
      stop_and_go = False
      ret.mass = 1849.2 + STD_CARGO_KG  # mean of 4 models in kg
      ret.wheelbase = 2.90
      ret.centerToFront = ret.wheelbase * 0.41  # from CAR.ODYSSEY
      ret.steerRatio = 14.35
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 32767], [0, 32767]]  # TODO: determine if there is a dead zone at the top end
      tire_stiffness_factor = 0.82
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.28], [0.08]]

    elif candidate in (CAR.PILOT, CAR.PILOT_2019):
      stop_and_go = False
      ret.mass = 4204. * CV.LB_TO_KG + STD_CARGO_KG  # average weight
      ret.wheelbase = 2.82
      ret.centerToFront = ret.wheelbase * 0.428
      ret.steerRatio = 17.25  # as spec
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]  # TODO: determine if there is a dead zone at the top end
      tire_stiffness_factor = 0.444
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.38], [0.11]]

    elif candidate == CAR.PASSPORT:
      stop_and_go = False
      ret.mass = 4204. * CV.LB_TO_KG + STD_CARGO_KG  # average weight
      ret.wheelbase = 2.82
      ret.centerToFront = ret.wheelbase * 0.428
      ret.steerRatio = 17.25  # as spec
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]  # TODO: determine if there is a dead zone at the top end
      tire_stiffness_factor = 0.444
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.38], [0.11]]

    elif candidate == CAR.RIDGELINE:
      stop_and_go = False
      ret.mass = 4515. * CV.LB_TO_KG + STD_CARGO_KG
      ret.wheelbase = 3.18
      ret.centerToFront = ret.wheelbase * 0.41
      ret.steerRatio = 15.59  # as spec
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]  # TODO: determine if there is a dead zone at the top end
      tire_stiffness_factor = 0.444
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.38], [0.11]]

    elif candidate == CAR.INSIGHT:
      stop_and_go = True
      ret.mass = 2987. * CV.LB_TO_KG + STD_CARGO_KG
      ret.wheelbase = 2.7
      ret.centerToFront = ret.wheelbase * 0.39
      ret.steerRatio = 15.0  # 12.58 is spec end-to-end
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]  # TODO: determine if there is a dead zone at the top end
      tire_stiffness_factor = 0.82
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.6], [0.18]]

    elif candidate == CAR.HONDA_E:
      stop_and_go = True
      ret.mass = 3338.8 * CV.LB_TO_KG + STD_CARGO_KG
      ret.wheelbase = 2.5
      ret.centerToFront = ret.wheelbase * 0.5
      ret.steerRatio = 16.71
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]  # TODO: determine if there is a dead zone at the top end
      tire_stiffness_factor = 0.82
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.6], [0.18]] # TODO: can probably use some tuning

    else:
      raise ValueError("unsupported car %s" % candidate)

    # These cars use alternate user brake msg (0x1BE)
    if candidate in HONDA_BOSCH_ALT_BRAKE_SIGNAL:
      ret.safetyConfigs[0].safetyParam |= Panda.FLAG_HONDA_ALT_BRAKE

    # These cars use alternate SCM messages (SCM_FEEDBACK AND SCM_BUTTON)
    if candidate in HONDA_NIDEC_ALT_SCM_MESSAGES:
      ret.safetyConfigs[0].safetyParam |= Panda.FLAG_HONDA_NIDEC_ALT

    if ret.openpilotLongitudinalControl and candidate in HONDA_BOSCH:
      ret.safetyConfigs[0].safetyParam |= Panda.FLAG_HONDA_BOSCH_LONG

    # min speed to enable ACC. if car can do stop and go, then set enabling speed
    # to a negative value, so it won't matter. Otherwise, add 0.5 mph margin to not
    # conflict with PCM acc
    ret.minEnableSpeed = -1. if (stop_and_go or ret.enableGasInterceptor) else 25.5 * CV.MPH_TO_MS

    # TODO: get actual value, for now starting with reasonable value for
    # civic and scaling by mass and wheelbase
    ret.rotationalInertia = scale_rot_inertia(ret.mass, ret.wheelbase)

    # TODO: start from empirically derived lateral slip stiffness for the civic and scale by
    # mass and CG position, so all cars will have approximately similar dyn behaviors
    ret.tireStiffnessFront, ret.tireStiffnessRear = scale_tire_stiffness(ret.mass, ret.wheelbase, ret.centerToFront,
                                                                         tire_stiffness_factor=tire_stiffness_factor)

    ret.steerActuatorDelay = 0.1
    ret.steerRateCost = 0.5
    ret.steerLimitTimer = 0.8

    return ret

  @staticmethod
  def init(CP, logcan, sendcan):
    if CP.carFingerprint in HONDA_BOSCH and CP.openpilotLongitudinalControl:
      disable_ecu(logcan, sendcan, bus=1, addr=0x18DAB0F1, com_cont_req=b'\x28\x83\x03')

  # returns a car.CarState
  def update(self, c, can_strings):
    # ******************* do can recv *******************
    self.cp.update_strings(can_strings)
    self.cp_cam.update_strings(can_strings)
    if self.cp_body:
      self.cp_body.update_strings(can_strings)

    ret = self.CS.update(self.cp, self.cp_cam, self.cp_body)

    ret.canValid = self.cp.can_valid and self.cp_cam.can_valid and (self.cp_body is None or self.cp_body.can_valid)
    ret.yawRate = self.VM.yaw_rate(ret.steeringAngleDeg * CV.DEG_TO_RAD, ret.vEgo)

    buttonEvents = []

    if self.CS.cruise_buttons != self.CS.prev_cruise_buttons:
      be = car.CarState.ButtonEvent.new_message()
      be.type = ButtonType.unknown
      if self.CS.cruise_buttons != 0:
        be.pressed = True
        but = self.CS.cruise_buttons
      else:
        be.pressed = False
        but = self.CS.prev_cruise_buttons
      if but == CruiseButtons.RES_ACCEL:
        be.type = ButtonType.accelCruise
      elif but == CruiseButtons.DECEL_SET:
        be.type = ButtonType.decelCruise
      elif but == CruiseButtons.CANCEL:
        be.type = ButtonType.cancel
      elif but == CruiseButtons.MAIN:
        be.type = ButtonType.altButton3
      buttonEvents.append(be)

    if self.CS.cruise_setting != self.CS.prev_cruise_setting:
      be = car.CarState.ButtonEvent.new_message()
      be.type = ButtonType.unknown
      if self.CS.cruise_setting != 0:
        be.pressed = True
        but = self.CS.cruise_setting
      else:
        be.pressed = False
        but = self.CS.prev_cruise_setting
      if but == 1:
        be.type = ButtonType.altButton1
      # TODO: more buttons?
      buttonEvents.append(be)
    ret.buttonEvents = buttonEvents

    # events
    events = self.create_common_events(ret, pcm_enable=False)
    if self.CS.brake_error:
      events.add(EventName.brakeUnavailable)
    if self.CS.park_brake:
      events.add(EventName.parkBrake)

    if self.CP.pcmCruise and ret.vEgo < self.CP.minEnableSpeed:
      events.add(EventName.belowEngageSpeed)

    if self.CP.pcmCruise:
      # we engage when pcm is active (rising edge)
      if ret.cruiseState.enabled and not self.CS.out.cruiseState.enabled:
        events.add(EventName.pcmEnable)
      elif not ret.cruiseState.enabled and (c.actuators.accel >= 0. or not self.CP.openpilotLongitudinalControl):
        # it can happen that car cruise disables while comma system is enabled: need to
        # keep braking if needed or if the speed is very low
        if ret.vEgo < self.CP.minEnableSpeed + 2.:
          # non loud alert if cruise disables below 25mph as expected (+ a little margin)
          events.add(EventName.speedTooLow)
        else:
          events.add(EventName.cruiseDisabled)
    if self.CS.CP.minEnableSpeed > 0 and ret.vEgo < 0.001:
      events.add(EventName.manualRestart)

    # handle button presses
    for b in ret.buttonEvents:

      # do enable on both accel and decel buttons
      if b.type in [ButtonType.accelCruise, ButtonType.decelCruise] and not b.pressed:
        if not self.CP.pcmCruise:
          events.add(EventName.buttonEnable)

      # do disable on button down
      if b.type == ButtonType.cancel and b.pressed:
        events.add(EventName.buttonCancel)

    ret.events = events.to_msg()

    self.CS.out = ret.as_reader()
    return self.CS.out

  # pass in a car.CarControl
  # to be called @ 100hz
  def apply(self, c):
    if c.hudControl.speedVisible:
      hud_v_cruise = c.hudControl.setSpeed * CV.MS_TO_KPH
    else:
      hud_v_cruise = 255

    can_sends = self.CC.update(c.enabled, c.active, self.CS, self.frame,
                               c.actuators,
                               c.cruiseControl.cancel,
                               hud_v_cruise,
                               c.hudControl.lanesVisible,
                               hud_show_car=c.hudControl.leadVisible,
                               hud_alert=c.hudControl.visualAlert)

    self.frame += 1
    return can_sends
