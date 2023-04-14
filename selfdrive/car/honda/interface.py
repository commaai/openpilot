#!/usr/bin/env python3
from cereal import car
from panda import Panda
from common.conversions import Conversions as CV
from common.numpy_fast import interp
from selfdrive.car.honda.values import CarControllerParams, CruiseButtons, HondaFlags, CAR, HONDA_BOSCH, HONDA_NIDEC_ALT_SCM_MESSAGES, HONDA_BOSCH_ALT_BRAKE_SIGNAL, HONDA_BOSCH_RADARLESS
from selfdrive.car import STD_CARGO_KG, CivicParams, create_button_event, scale_tire_stiffness, get_safety_config
from selfdrive.car.interfaces import CarInterfaceBase
from selfdrive.car.disable_ecu import disable_ecu


ButtonType = car.CarState.ButtonEvent.Type
EventName = car.CarEvent.EventName
TransmissionType = car.CarParams.TransmissionType
BUTTONS_DICT = {CruiseButtons.RES_ACCEL: ButtonType.accelCruise, CruiseButtons.DECEL_SET: ButtonType.decelCruise,
                CruiseButtons.MAIN: ButtonType.altButton3, CruiseButtons.CANCEL: ButtonType.cancel}


class CarInterface(CarInterfaceBase):
  @staticmethod
  def get_pid_accel_limits(CP, current_speed, cruise_speed):
    if CP.carFingerprint in HONDA_BOSCH:
      return CarControllerParams.BOSCH_ACCEL_MIN, CarControllerParams.BOSCH_ACCEL_MAX
    elif CP.enableGasInterceptor:
      return CarControllerParams.NIDEC_ACCEL_MIN, CarControllerParams.NIDEC_ACCEL_MAX
    else:
      # NIDECs don't allow acceleration near cruise_speed,
      # so limit limits of pid to prevent windup
      ACCEL_MAX_VALS = [CarControllerParams.NIDEC_ACCEL_MAX, 0.2]
      ACCEL_MAX_BP = [cruise_speed - 2., cruise_speed - .2]
      return CarControllerParams.NIDEC_ACCEL_MIN, interp(current_speed, ACCEL_MAX_BP, ACCEL_MAX_VALS)

  @staticmethod
  def _get_params(ret, candidate, fingerprint, car_fw, experimental_long):
    ret.carName = "honda"

    if candidate in HONDA_BOSCH:
      ret.safetyConfigs = [get_safety_config(car.CarParams.SafetyModel.hondaBosch)]
      ret.radarUnavailable = True
      # Disable the radar and let openpilot control longitudinal
      # WARNING: THIS DISABLES AEB!
      # If Bosch radarless, this blocks ACC messages from the camera
      ret.experimentalLongitudinalAvailable = True
      ret.openpilotLongitudinalControl = experimental_long
      ret.pcmCruise = not ret.openpilotLongitudinalControl
    else:
      ret.safetyConfigs = [get_safety_config(car.CarParams.SafetyModel.hondaNidec)]
      ret.enableGasInterceptor = 0x201 in fingerprint[0]
      ret.openpilotLongitudinalControl = True

      ret.pcmCruise = not ret.enableGasInterceptor

    if candidate == CAR.CRV_5G:
      ret.enableBsm = 0x12f8bfa7 in fingerprint[0]

    # Detect Bosch cars with new HUD msgs
    if any(0x33DA in f for f in fingerprint.values()):
      ret.flags |= HondaFlags.BOSCH_EXT_HUD.value

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
      if candidate in HONDA_BOSCH_RADARLESS:
        ret.stopAccel = CarControllerParams.BOSCH_ACCEL_MIN  # stock uses -4.0 m/s^2 once stopped but limited by safety model
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

    elif candidate in (CAR.CIVIC_BOSCH, CAR.CIVIC_BOSCH_DIESEL, CAR.CIVIC_2022):
      ret.mass = CivicParams.MASS
      ret.wheelbase = CivicParams.WHEELBASE
      ret.centerToFront = CivicParams.CENTER_TO_FRONT
      ret.steerRatio = 15.38  # 10.93 is end-to-end spec
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]  # TODO: determine if there is a dead zone at the top end
      tire_stiffness_factor = 1.
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.8], [0.24]]

    elif candidate in (CAR.ACCORD, CAR.ACCORDH):
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
      ret.mass = 3095. * CV.LB_TO_KG + STD_CARGO_KG
      ret.wheelbase = 2.67
      ret.centerToFront = ret.wheelbase * 0.37
      ret.steerRatio = 18.61  # 15.3 is spec end-to-end
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 3840], [0, 3840]]  # TODO: determine if there is a dead zone at the top end
      tire_stiffness_factor = 0.72
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.8], [0.24]]

    elif candidate in (CAR.CRV, CAR.CRV_EU):
      ret.mass = 3572. * CV.LB_TO_KG + STD_CARGO_KG
      ret.wheelbase = 2.62
      ret.centerToFront = ret.wheelbase * 0.41
      ret.steerRatio = 16.89  # as spec
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 1000], [0, 1000]]  # TODO: determine if there is a dead zone at the top end
      tire_stiffness_factor = 0.444
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.8], [0.24]]
      ret.wheelSpeedFactor = 1.025

    elif candidate == CAR.CRV_5G:
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
      ret.mass = 1667. + STD_CARGO_KG  # mean of 4 models in kg
      ret.wheelbase = 2.66
      ret.centerToFront = ret.wheelbase * 0.41
      ret.steerRatio = 16.0  # 12.3 is spec end-to-end
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]  # TODO: determine if there is a dead zone at the top end
      tire_stiffness_factor = 0.677
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.6], [0.18]]
      ret.wheelSpeedFactor = 1.025

    elif candidate == CAR.FIT:
      ret.mass = 2644. * CV.LB_TO_KG + STD_CARGO_KG
      ret.wheelbase = 2.53
      ret.centerToFront = ret.wheelbase * 0.39
      ret.steerRatio = 13.06
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]  # TODO: determine if there is a dead zone at the top end
      tire_stiffness_factor = 0.75
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.2], [0.05]]

    elif candidate == CAR.FREED:
      ret.mass = 3086. * CV.LB_TO_KG + STD_CARGO_KG
      ret.wheelbase = 2.74
      # the remaining parameters were copied from FIT
      ret.centerToFront = ret.wheelbase * 0.39
      ret.steerRatio = 13.06
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]
      tire_stiffness_factor = 0.75
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.2], [0.05]]

    elif candidate in (CAR.HRV, CAR.HRV_3G):
      ret.mass = 3125 * CV.LB_TO_KG + STD_CARGO_KG
      ret.wheelbase = 2.61
      ret.centerToFront = ret.wheelbase * 0.41
      ret.steerRatio = 15.2
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]
      tire_stiffness_factor = 0.5
      if candidate == CAR.HRV:
        ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.16], [0.025]]
        ret.wheelSpeedFactor = 1.025
      else:
        ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.8], [0.24]]  # TODO: can probably use some tuning

    elif candidate == CAR.ACURA_RDX:
      ret.mass = 3935. * CV.LB_TO_KG + STD_CARGO_KG
      ret.wheelbase = 2.68
      ret.centerToFront = ret.wheelbase * 0.38
      ret.steerRatio = 15.0  # as spec
      tire_stiffness_factor = 0.444
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 1000], [0, 1000]]  # TODO: determine if there is a dead zone at the top end
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.8], [0.24]]

    elif candidate == CAR.ACURA_RDX_3G:
      ret.mass = 4068. * CV.LB_TO_KG + STD_CARGO_KG
      ret.wheelbase = 2.75
      ret.centerToFront = ret.wheelbase * 0.41
      ret.steerRatio = 11.95  # as spec
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 3840], [0, 3840]]
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.2], [0.06]]
      tire_stiffness_factor = 0.677

    elif candidate in (CAR.ODYSSEY, CAR.ODYSSEY_CHN):
      ret.mass = 1900. + STD_CARGO_KG
      ret.wheelbase = 3.00
      ret.centerToFront = ret.wheelbase * 0.41
      ret.steerRatio = 14.35  # as spec
      tire_stiffness_factor = 0.82
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.28], [0.08]]
      if candidate == CAR.ODYSSEY_CHN:
        ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 32767], [0, 32767]]  # TODO: determine if there is a dead zone at the top end
      else:
        ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]  # TODO: determine if there is a dead zone at the top end

    elif candidate == CAR.PILOT:
      ret.mass = 4278. * CV.LB_TO_KG + STD_CARGO_KG  # average weight
      ret.wheelbase = 2.86
      ret.centerToFront = ret.wheelbase * 0.428
      ret.steerRatio = 16.0  # as spec
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]  # TODO: determine if there is a dead zone at the top end
      tire_stiffness_factor = 0.444
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.38], [0.11]]

    elif candidate == CAR.RIDGELINE:
      ret.mass = 4515. * CV.LB_TO_KG + STD_CARGO_KG
      ret.wheelbase = 3.18
      ret.centerToFront = ret.wheelbase * 0.41
      ret.steerRatio = 15.59  # as spec
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]  # TODO: determine if there is a dead zone at the top end
      tire_stiffness_factor = 0.444
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.38], [0.11]]

    elif candidate == CAR.INSIGHT:
      ret.mass = 2987. * CV.LB_TO_KG + STD_CARGO_KG
      ret.wheelbase = 2.7
      ret.centerToFront = ret.wheelbase * 0.39
      ret.steerRatio = 15.0  # 12.58 is spec end-to-end
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]  # TODO: determine if there is a dead zone at the top end
      tire_stiffness_factor = 0.82
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.6], [0.18]]

    elif candidate == CAR.HONDA_E:
      ret.mass = 3338.8 * CV.LB_TO_KG + STD_CARGO_KG
      ret.wheelbase = 2.5
      ret.centerToFront = ret.wheelbase * 0.5
      ret.steerRatio = 16.71
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]  # TODO: determine if there is a dead zone at the top end
      tire_stiffness_factor = 0.82
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.6], [0.18]] # TODO: can probably use some tuning

    else:
      raise ValueError(f"unsupported car {candidate}")

    # These cars use alternate user brake msg (0x1BE)
    if candidate in HONDA_BOSCH_ALT_BRAKE_SIGNAL:
      ret.safetyConfigs[0].safetyParam |= Panda.FLAG_HONDA_ALT_BRAKE

    # These cars use alternate SCM messages (SCM_FEEDBACK AND SCM_BUTTON)
    if candidate in HONDA_NIDEC_ALT_SCM_MESSAGES:
      ret.safetyConfigs[0].safetyParam |= Panda.FLAG_HONDA_NIDEC_ALT

    if ret.openpilotLongitudinalControl and candidate in HONDA_BOSCH:
      ret.safetyConfigs[0].safetyParam |= Panda.FLAG_HONDA_BOSCH_LONG

    if candidate in HONDA_BOSCH_RADARLESS:
      ret.safetyConfigs[0].safetyParam |= Panda.FLAG_HONDA_RADARLESS

    # min speed to enable ACC. if car can do stop and go, then set enabling speed
    # to a negative value, so it won't matter. Otherwise, add 0.5 mph margin to not
    # conflict with PCM acc
    stop_and_go = candidate in (HONDA_BOSCH | {CAR.CIVIC}) or ret.enableGasInterceptor
    ret.minEnableSpeed = -1. if stop_and_go else 25.5 * CV.MPH_TO_MS

    # TODO: start from empirically derived lateral slip stiffness for the civic and scale by
    # mass and CG position, so all cars will have approximately similar dyn behaviors
    ret.tireStiffnessFront, ret.tireStiffnessRear = scale_tire_stiffness(ret.mass, ret.wheelbase, ret.centerToFront,
                                                                         tire_stiffness_factor=tire_stiffness_factor)

    ret.steerActuatorDelay = 0.1
    ret.steerLimitTimer = 0.8

    return ret

  @staticmethod
  def init(CP, logcan, sendcan):
    if CP.carFingerprint in (HONDA_BOSCH - HONDA_BOSCH_RADARLESS) and CP.openpilotLongitudinalControl:
      disable_ecu(logcan, sendcan, bus=1, addr=0x18DAB0F1, com_cont_req=b'\x28\x83\x03')

  # returns a car.CarState
  def _update(self, c):
    ret = self.CS.update(self.cp, self.cp_cam, self.cp_body)

    buttonEvents = []

    if self.CS.cruise_buttons != self.CS.prev_cruise_buttons:
      buttonEvents.append(create_button_event(self.CS.cruise_buttons, self.CS.prev_cruise_buttons, BUTTONS_DICT))

    if self.CS.cruise_setting != self.CS.prev_cruise_setting:
      buttonEvents.append(create_button_event(self.CS.cruise_setting, self.CS.prev_cruise_setting, {1: ButtonType.altButton1}))

    ret.buttonEvents = buttonEvents

    # events
    events = self.create_common_events(ret, pcm_enable=False)
    if self.CS.brake_error:
      events.add(EventName.brakeUnavailable)

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

    ret.events = events.to_msg()

    return ret

  # pass in a car.CarControl
  # to be called @ 100hz
  def apply(self, c, now_nanos):
    return self.CC.update(c, self.CS, now_nanos)
