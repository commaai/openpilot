#!/usr/bin/env python
import os
import numpy as np
from cereal import car, log
from common.numpy_fast import clip, interp
from common.realtime import sec_since_boot
from selfdrive.swaglog import cloudlog
from selfdrive.config import Conversions as CV
from selfdrive.controls.lib.drive_helpers import create_event, EventTypes as ET, get_events
from selfdrive.controls.lib.vehicle_model import VehicleModel
from selfdrive.car.honda.carstate import CarState, get_can_parser
from selfdrive.car.honda.values import CruiseButtons, CM, BP, AH, CAR, HONDA_BOSCH
from selfdrive.controls.lib.planner import A_ACC_MAX

try:
  from selfdrive.car.honda.carcontroller import CarController
except ImportError:
  CarController = None


# msgs sent for steering controller by camera module on can 0.
# those messages are mutually exclusive on CRV and non-CRV cars
CAMERA_MSGS = [0xe4, 0x194]


def compute_gb_honda(accel, speed):
  creep_brake = 0.0
  creep_speed = 2.3
  creep_brake_value = 0.15
  if speed < creep_speed:
    creep_brake = (creep_speed - speed) / creep_speed * creep_brake_value
  return float(accel) / 4.8 - creep_brake


def get_compute_gb_acura():
  # generate a function that takes in [desired_accel, current_speed] -> [-1.0, 1.0]
  # where -1.0 is max brake and 1.0 is max gas
  # see debug/dump_accel_from_fiber.py to see how those parameters were generated
  w0 = np.array([[ 1.22056961, -0.39625418,  0.67952657],
                 [ 1.03691769,  0.78210306, -0.41343188]])
  b0 = np.array([ 0.01536703, -0.14335321, -0.26932889])
  w2 = np.array([[-0.59124422,  0.42899439,  0.38660881],
                 [ 0.79973811,  0.13178682,  0.08550351],
                 [-0.15651935, -0.44360259,  0.76910877]])
  b2 = np.array([ 0.15624429,  0.02294923, -0.0341086 ])
  w4 = np.array([[-0.31521443],
                 [-0.38626176],
                 [ 0.52667892]])
  b4 = np.array([-0.02922216])

  def compute_output(dat, w0, b0, w2, b2, w4, b4):
    m0 = np.dot(dat, w0) + b0
    m0 = leakyrelu(m0, 0.1)
    m2 = np.dot(m0, w2) + b2
    m2 = leakyrelu(m2, 0.1)
    m4 = np.dot(m2, w4) + b4
    return m4

  def leakyrelu(x, alpha):
    return np.maximum(x, alpha * x)

  def _compute_gb_acura(accel, speed):
    # linearly extrap below v1 using v1 and v2 data
    v1 = 5.
    v2 = 10.
    dat = np.array([accel, speed])
    if speed > 5.:
      m4 = compute_output(dat, w0, b0, w2, b2, w4, b4)
    else:
      dat[1] = v1
      m4v1 = compute_output(dat, w0, b0, w2, b2, w4, b4)
      dat[1] = v2
      m4v2 = compute_output(dat, w0, b0, w2, b2, w4, b4)
      m4 = (speed - v1) * (m4v2 - m4v1) / (v2 - v1) + m4v1
    return float(m4)

  return _compute_gb_acura


class CarInterface(object):
  def __init__(self, CP, sendcan=None):
    self.CP = CP

    self.frame = 0
    self.last_enable_pressed = 0
    self.last_enable_sent = 0
    self.gas_pressed_prev = False
    self.brake_pressed_prev = False
    self.can_invalid_count = 0

    self.cp = get_can_parser(CP)

    # *** init the major players ***
    self.CS = CarState(CP)
    self.VM = VehicleModel(CP)

    # sending if read only is False
    if sendcan is not None:
      self.sendcan = sendcan
      self.CC = CarController(self.cp.dbc_name, CP.enableCamera)

    if self.CS.CP.carFingerprint == CAR.ACURA_ILX:
      self.compute_gb = get_compute_gb_acura()
    else:
      self.compute_gb = compute_gb_honda

  @staticmethod
  def calc_accel_override(a_ego, a_target, v_ego, v_target):
    # limit the pcm accel cmd if:
    # - v_ego exceeds v_target, or
    # - a_ego exceeds a_target and v_ego is close to v_target

    eA = a_ego - a_target
    valuesA = [1.0, 0.1]
    bpA = [0.3, 1.1]

    eV = v_ego - v_target
    valuesV = [1.0, 0.1]
    bpV = [0.0, 0.5]

    valuesRangeV = [1., 0.]
    bpRangeV = [-1., 0.]

    # only limit if v_ego is close to v_target
    speedLimiter = interp(eV, bpV, valuesV)
    accelLimiter = max(interp(eA, bpA, valuesA), interp(eV, bpRangeV, valuesRangeV))

    # accelOverride is more or less the max throttle allowed to pcm: usually set to a constant
    # unless aTargetMax is very high and then we scale with it; this help in quicker restart

    return float(max(0.714, a_target / A_ACC_MAX)) * min(speedLimiter, accelLimiter)

  @staticmethod
  def get_params(candidate, fingerprint):

    ret = car.CarParams.new_message()
    ret.carName = "honda"
    ret.carFingerprint = candidate

    if candidate in HONDA_BOSCH:
      ret.safetyModel = car.CarParams.SafetyModels.hondaBosch
      ret.enableCamera = True
      ret.radarOffCan = True
    else:
      ret.safetyModel = car.CarParams.SafetyModels.honda
      ret.enableCamera = not any(x for x in CAMERA_MSGS if x in fingerprint)
      ret.enableGasInterceptor = 0x201 in fingerprint
    cloudlog.warn("ECU Camera Simulated: %r", ret.enableCamera)
    cloudlog.warn("ECU Gas Interceptor: %r", ret.enableGasInterceptor)

    ret.enableCruise = not ret.enableGasInterceptor

    # kg of standard extra cargo to count for drive, gas, etc...
    std_cargo = 136

    # FIXME: hardcoding honda civic 2016 touring params so they can be used to
    # scale unknown params for other cars
    mass_civic = 2923 * CV.LB_TO_KG + std_cargo
    wheelbase_civic = 2.70
    centerToFront_civic = wheelbase_civic * 0.4
    centerToRear_civic = wheelbase_civic - centerToFront_civic
    rotationalInertia_civic = 2500
    tireStiffnessFront_civic = 192150
    tireStiffnessRear_civic = 202500

    # Optimized car params: tire_stiffness_factor and steerRatio are a result of a vehicle
    # model optimization process. Certain Hondas have an extra steering sensor at the bottom
    # of the steering rack, which improves controls quality as it removes the steering column
    # torsion from feedback.
    # Tire stiffness factor fictitiously lower if it includes the steering column torsion effect.
    # For modeling details, see p.198-200 in "The Science of Vehicle Dynamics (2014), M. Guiggiani"

    ret.steerKiBP, ret.steerKpBP = [[0.], [0.]]

    ret.steerKf = 0.00006 # conservative feed-forward

    if candidate == CAR.CIVIC:
      stop_and_go = True
      ret.mass = mass_civic
      ret.wheelbase = wheelbase_civic
      ret.centerToFront = centerToFront_civic
      ret.steerRatio = 14.63  # 10.93 is end-to-end spec
      tire_stiffness_factor = 1.
      # Civic at comma has modified steering FW, so different tuning for the Neo in that car
      is_fw_modified = os.getenv("DONGLE_ID") in ['99c94dc769b5d96e']
      ret.steerKpV, ret.steerKiV = [[0.33], [0.10]] if is_fw_modified else [[0.8], [0.24]]
      if is_fw_modified:
        ret.steerKf = 0.00003
      ret.longitudinalKpBP = [0., 5., 35.]
      ret.longitudinalKpV = [3.6, 2.4, 1.5]
      ret.longitudinalKiBP = [0., 35.]
      ret.longitudinalKiV = [0.54, 0.36]

    elif candidate == CAR.CIVIC_HATCH:
      stop_and_go = True
      ret.mass = 2916. * CV.LB_TO_KG + std_cargo
      ret.wheelbase = wheelbase_civic
      ret.centerToFront = centerToFront_civic
      ret.steerRatio = 14.63  # 10.93 is spec end-to-end
      tire_stiffness_factor = 1.
      ret.steerKpV, ret.steerKiV = [[0.8], [0.24]]
      ret.longitudinalKpBP = [0., 5., 35.]
      ret.longitudinalKpV = [1.2, 0.8, 0.5]
      ret.longitudinalKiBP = [0., 35.]
      ret.longitudinalKiV = [0.18, 0.12]

    elif candidate in (CAR.ACCORD, CAR.ACCORD_15, CAR.ACCORDH):
      stop_and_go = True
      if not candidate == CAR.ACCORDH: # Hybrid uses same brake msg as hatch
        ret.safetyParam = 1 # Accord and CRV 5G use an alternate user brake msg
      ret.mass = 3279. * CV.LB_TO_KG + std_cargo
      ret.wheelbase = 2.83
      ret.centerToFront = ret.wheelbase * 0.39
      ret.steerRatio = 15.96  # 11.82 is spec end-to-end
      tire_stiffness_factor = 0.8467
      ret.steerKpV, ret.steerKiV = [[0.6], [0.18]]
      ret.longitudinalKpBP = [0., 5., 35.]
      ret.longitudinalKpV = [1.2, 0.8, 0.5]
      ret.longitudinalKiBP = [0., 35.]
      ret.longitudinalKiV = [0.18, 0.12]

    elif candidate == CAR.ACURA_ILX:
      stop_and_go = False
      ret.mass = 3095 * CV.LB_TO_KG + std_cargo
      ret.wheelbase = 2.67
      ret.centerToFront = ret.wheelbase * 0.37
      ret.steerRatio = 18.61  # 15.3 is spec end-to-end
      tire_stiffness_factor = 0.72
      # Acura at comma has modified steering FW, so different tuning for the Neo in that car
      is_fw_modified = os.getenv("DONGLE_ID") in ['ff83f397542ab647']
      ret.steerKpV, ret.steerKiV = [[0.45], [0.00]] if is_fw_modified else [[0.8], [0.24]]
      if is_fw_modified:
        ret.steerKf = 0.00003
      ret.longitudinalKpBP = [0., 5., 35.]
      ret.longitudinalKpV = [1.2, 0.8, 0.5]
      ret.longitudinalKiBP = [0., 35.]
      ret.longitudinalKiV = [0.18, 0.12]

    elif candidate == CAR.CRV:
      stop_and_go = False
      ret.mass = 3572 * CV.LB_TO_KG + std_cargo
      ret.wheelbase = 2.62
      ret.centerToFront = ret.wheelbase * 0.41
      ret.steerRatio = 15.3         # as spec
      tire_stiffness_factor = 0.444 # not optimized yet
      ret.steerKpV, ret.steerKiV = [[0.8], [0.24]]
      ret.longitudinalKpBP = [0., 5., 35.]
      ret.longitudinalKpV = [1.2, 0.8, 0.5]
      ret.longitudinalKiBP = [0., 35.]
      ret.longitudinalKiV = [0.18, 0.12]

    elif candidate == CAR.CRV_5G:
      stop_and_go = True
      ret.safetyParam = 1 # Accord and CRV 5G use an alternate user brake msg
      ret.mass = 3410. * CV.LB_TO_KG + std_cargo
      ret.wheelbase = 2.66
      ret.centerToFront = ret.wheelbase * 0.41
      ret.steerRatio = 16.0   # 12.3 is spec end-to-end
      tire_stiffness_factor = 0.677
      ret.steerKpV, ret.steerKiV = [[0.6], [0.18]]
      ret.longitudinalKpBP = [0., 5., 35.]
      ret.longitudinalKpV = [1.2, 0.8, 0.5]
      ret.longitudinalKiBP = [0., 35.]
      ret.longitudinalKiV = [0.18, 0.12]

    elif candidate == CAR.ACURA_RDX:
      stop_and_go = False
      ret.mass = 3935 * CV.LB_TO_KG + std_cargo
      ret.wheelbase = 2.68
      ret.centerToFront = ret.wheelbase * 0.38
      ret.steerRatio = 15.0         # as spec
      tire_stiffness_factor = 0.444 # not optimized yet
      ret.steerKpV, ret.steerKiV = [[0.8], [0.24]]
      ret.longitudinalKpBP = [0., 5., 35.]
      ret.longitudinalKpV = [1.2, 0.8, 0.5]
      ret.longitudinalKiBP = [0., 35.]
      ret.longitudinalKiV = [0.18, 0.12]

    elif candidate == CAR.ODYSSEY:
      stop_and_go = False
      ret.mass = 4471 * CV.LB_TO_KG + std_cargo
      ret.wheelbase = 3.00
      ret.centerToFront = ret.wheelbase * 0.41
      ret.steerRatio = 14.35        # as spec
      tire_stiffness_factor = 0.82
      ret.steerKpV, ret.steerKiV = [[0.45], [0.135]]
      ret.longitudinalKpBP = [0., 5., 35.]
      ret.longitudinalKpV = [1.2, 0.8, 0.5]
      ret.longitudinalKiBP = [0., 35.]
      ret.longitudinalKiV = [0.18, 0.12]

    elif candidate in (CAR.PILOT, CAR.PILOT_2019):
      stop_and_go = False
      ret.mass = 4303 * CV.LB_TO_KG + std_cargo
      ret.wheelbase = 2.81
      ret.centerToFront = ret.wheelbase * 0.41
      ret.steerRatio = 16.0         # as spec
      tire_stiffness_factor = 0.444 # not optimized yet
      ret.steerKpV, ret.steerKiV = [[0.38], [0.11]]
      ret.longitudinalKpBP = [0., 5., 35.]
      ret.longitudinalKpV = [1.2, 0.8, 0.5]
      ret.longitudinalKiBP = [0., 35.]
      ret.longitudinalKiV = [0.18, 0.12]

    elif candidate == CAR.RIDGELINE:
      stop_and_go = False
      ret.mass = 4515 * CV.LB_TO_KG + std_cargo
      ret.wheelbase = 3.18
      ret.centerToFront = ret.wheelbase * 0.41
      ret.steerRatio = 15.59        # as spec
      tire_stiffness_factor = 0.444 # not optimized yet
      ret.steerKpV, ret.steerKiV = [[0.38], [0.11]]
      ret.longitudinalKpBP = [0., 5., 35.]
      ret.longitudinalKpV = [1.2, 0.8, 0.5]
      ret.longitudinalKiBP = [0., 35.]
      ret.longitudinalKiV = [0.18, 0.12]

    else:
      raise ValueError("unsupported car %s" % candidate)

    ret.steerControlType = car.CarParams.SteerControlType.torque

    # min speed to enable ACC. if car can do stop and go, then set enabling speed
    # to a negative value, so it won't matter. Otherwise, add 0.5 mph margin to not
    # conflict with PCM acc
    ret.minEnableSpeed = -1. if (stop_and_go or ret.enableGasInterceptor) else 25.5 * CV.MPH_TO_MS

    centerToRear = ret.wheelbase - ret.centerToFront
    # TODO: get actual value, for now starting with reasonable value for
    # civic and scaling by mass and wheelbase
    ret.rotationalInertia = rotationalInertia_civic * \
                            ret.mass * ret.wheelbase**2 / (mass_civic * wheelbase_civic**2)

    # TODO: start from empirically derived lateral slip stiffness for the civic and scale by
    # mass and CG position, so all cars will have approximately similar dyn behaviors
    ret.tireStiffnessFront = (tireStiffnessFront_civic * tire_stiffness_factor) * \
                             ret.mass / mass_civic * \
                             (centerToRear / ret.wheelbase) / (centerToRear_civic / wheelbase_civic)
    ret.tireStiffnessRear = (tireStiffnessRear_civic * tire_stiffness_factor) * \
                            ret.mass / mass_civic * \
                            (ret.centerToFront / ret.wheelbase) / (centerToFront_civic / wheelbase_civic)

    # no rear steering, at least on the listed cars above
    ret.steerRatioRear = 0.

    # no max steer limit VS speed
    ret.steerMaxBP = [0.]  # m/s
    ret.steerMaxV = [1.]   # max steer allowed

    ret.gasMaxBP = [0.]  # m/s
    ret.gasMaxV = [0.6] if ret.enableGasInterceptor else [0.] # max gas allowed
    ret.brakeMaxBP = [5., 20.]  # m/s
    ret.brakeMaxV = [1., 0.8]   # max brake allowed

    ret.longPidDeadzoneBP = [0.]
    ret.longPidDeadzoneV = [0.]

    ret.stoppingControl = True
    ret.steerLimitAlert = True
    ret.startAccel = 0.5

    ret.steerActuatorDelay = 0.1
    ret.steerRateCost = 0.5

    return ret

  # returns a car.CarState
  def update(self, c):
    # ******************* do can recv *******************
    canMonoTimes = []

    self.cp.update(int(sec_since_boot() * 1e9), False)

    self.CS.update(self.cp)

    # create message
    ret = car.CarState.new_message()

    # speeds
    ret.vEgo = self.CS.v_ego
    ret.aEgo = self.CS.a_ego
    ret.vEgoRaw = self.CS.v_ego_raw
    ret.yawRate = self.VM.yaw_rate(self.CS.angle_steers * CV.DEG_TO_RAD, self.CS.v_ego)
    ret.standstill = self.CS.standstill
    ret.wheelSpeeds.fl = self.CS.v_wheel_fl
    ret.wheelSpeeds.fr = self.CS.v_wheel_fr
    ret.wheelSpeeds.rl = self.CS.v_wheel_rl
    ret.wheelSpeeds.rr = self.CS.v_wheel_rr

    # gas pedal
    ret.gas = self.CS.car_gas / 256.0
    if not self.CP.enableGasInterceptor:
      ret.gasPressed = self.CS.pedal_gas > 0
    else:
      ret.gasPressed = self.CS.user_gas_pressed

    # brake pedal
    ret.brake = self.CS.user_brake
    ret.brakePressed = self.CS.brake_pressed != 0
    # FIXME: read sendcan for brakelights
    brakelights_threshold = 0.02 if self.CS.CP.carFingerprint == CAR.CIVIC else 0.1
    ret.brakeLights = bool(self.CS.brake_switch or
                           c.actuators.brake > brakelights_threshold)

    # steering wheel
    ret.steeringAngle = self.CS.angle_steers
    ret.steeringRate = self.CS.angle_steers_rate

    # gear shifter lever
    ret.gearShifter = self.CS.gear_shifter

    ret.steeringTorque = self.CS.steer_torque_driver
    ret.steeringPressed = self.CS.steer_override

    # cruise state
    ret.cruiseState.enabled = self.CS.pcm_acc_status != 0
    ret.cruiseState.speed = self.CS.v_cruise_pcm * CV.KPH_TO_MS
    ret.cruiseState.available = bool(self.CS.main_on)
    ret.cruiseState.speedOffset = self.CS.cruise_speed_offset
    ret.cruiseState.standstill = False

    # TODO: button presses
    buttonEvents = []
    ret.leftBlinker = bool(self.CS.left_blinker_on)
    ret.rightBlinker = bool(self.CS.right_blinker_on)

    ret.doorOpen = not self.CS.door_all_closed
    ret.seatbeltUnlatched = not self.CS.seatbelt

    if self.CS.left_blinker_on != self.CS.prev_left_blinker_on:
      be = car.CarState.ButtonEvent.new_message()
      be.type = 'leftBlinker'
      be.pressed = self.CS.left_blinker_on != 0
      buttonEvents.append(be)

    if self.CS.right_blinker_on != self.CS.prev_right_blinker_on:
      be = car.CarState.ButtonEvent.new_message()
      be.type = 'rightBlinker'
      be.pressed = self.CS.right_blinker_on != 0
      buttonEvents.append(be)

    if self.CS.cruise_buttons != self.CS.prev_cruise_buttons:
      be = car.CarState.ButtonEvent.new_message()
      be.type = 'unknown'
      if self.CS.cruise_buttons != 0:
        be.pressed = True
        but = self.CS.cruise_buttons
      else:
        be.pressed = False
        but = self.CS.prev_cruise_buttons
      if but == CruiseButtons.RES_ACCEL:
        be.type = 'accelCruise'
      elif but == CruiseButtons.DECEL_SET:
        be.type = 'decelCruise'
      elif but == CruiseButtons.CANCEL:
        be.type = 'cancel'
      elif but == CruiseButtons.MAIN:
        be.type = 'altButton3'
      buttonEvents.append(be)

    if self.CS.cruise_setting != self.CS.prev_cruise_setting:
      be = car.CarState.ButtonEvent.new_message()
      be.type = 'unknown'
      if self.CS.cruise_setting != 0:
        be.pressed = True
        but = self.CS.cruise_setting
      else:
        be.pressed = False
        but = self.CS.prev_cruise_setting
      if but == 1:
        be.type = 'altButton1'
      # TODO: more buttons?
      buttonEvents.append(be)
    ret.buttonEvents = buttonEvents
    ret.gasbuttonstatus = self.CS.cstm_btns.get_button_status("gas")
    # events
    # TODO: I don't like the way capnp does enums
    # These strings aren't checked at compile time
    events = []
    if not self.CS.can_valid:
      self.can_invalid_count += 1
      if self.can_invalid_count >= 5:
        events.append(create_event('commIssue', [ET.NO_ENTRY, ET.IMMEDIATE_DISABLE]))
    else:
      self.can_invalid_count = 0
    if self.CS.steer_error:
      events.append(create_event('steerUnavailable', [ET.NO_ENTRY, ET.IMMEDIATE_DISABLE, ET.PERMANENT]))
    elif self.CS.steer_warning:
      events.append(create_event('steerTempUnavailable', [ET.WARNING]))
    if self.CS.brake_error:
      events.append(create_event('brakeUnavailable', [ET.NO_ENTRY, ET.IMMEDIATE_DISABLE, ET.PERMANENT]))
    if not ret.gearShifter == 'drive':
      events.append(create_event('wrongGear', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
    if ret.doorOpen:
      events.append(create_event('doorOpen', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
    if ret.seatbeltUnlatched:
      events.append(create_event('seatbeltNotLatched', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
    if self.CS.esp_disabled:
      events.append(create_event('espDisabled', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
    if not self.CS.main_on:
      events.append(create_event('wrongCarMode', [ET.NO_ENTRY, ET.USER_DISABLE]))
    if ret.gearShifter == 'reverse':
      events.append(create_event('reverseGear', [ET.NO_ENTRY, ET.IMMEDIATE_DISABLE]))
    if self.CS.brake_hold and self.CS.CP.carFingerprint not in HONDA_BOSCH:
      events.append(create_event('brakeHold', [ET.NO_ENTRY, ET.USER_DISABLE]))
    if self.CS.park_brake:
      events.append(create_event('parkBrake', [ET.NO_ENTRY, ET.USER_DISABLE]))

    if self.CP.enableCruise and ret.vEgo < self.CP.minEnableSpeed:
      events.append(create_event('speedTooLow', [ET.NO_ENTRY]))

    # disable on pedals rising edge or when brake is pressed and speed isn't zero
    if (ret.gasPressed and not self.gas_pressed_prev) or \
       (ret.brakePressed and (not self.brake_pressed_prev or ret.vEgo > 0.001)):
      events.append(create_event('pedalPressed', [ET.NO_ENTRY, ET.USER_DISABLE]))

    if ret.gasPressed:
      events.append(create_event('pedalPressed', [ET.PRE_ENABLE]))

    # it can happen that car cruise disables while comma system is enabled: need to
    # keep braking if needed or if the speed is very low
    if self.CP.enableCruise and not ret.cruiseState.enabled and c.actuators.brake <= 0.:
      # non loud alert if cruise disbales below 25mph as expected (+ a little margin)
      if ret.vEgo < self.CP.minEnableSpeed + 2.:
        events.append(create_event('speedTooLow', [ET.IMMEDIATE_DISABLE]))
      else:
        events.append(create_event("cruiseDisabled", [ET.IMMEDIATE_DISABLE]))
    if self.CS.CP.minEnableSpeed > 0 and ret.vEgo < 0.001:
      events.append(create_event('manualRestart', [ET.WARNING]))

    cur_time = sec_since_boot()
    enable_pressed = False
    # handle button presses
    for b in ret.buttonEvents:

      # do enable on both accel and decel buttons
      if b.type in ["accelCruise", "decelCruise"] and not b.pressed:
        self.last_enable_pressed = cur_time
        enable_pressed = True

      # do disable on button down
      if b.type == "cancel" and b.pressed:
        events.append(create_event('buttonCancel', [ET.USER_DISABLE]))

    if self.CP.enableCruise:
      # KEEP THIS EVENT LAST! send enable event if button is pressed and there are
      # NO_ENTRY events, so controlsd will display alerts. Also not send enable events
      # too close in time, so a no_entry will not be followed by another one.
      # TODO: button press should be the only thing that triggers enble
      if ((cur_time - self.last_enable_pressed) < 0.2 and
          (cur_time - self.last_enable_sent) > 0.2 and
          ret.cruiseState.enabled) or \
         (enable_pressed and get_events(events, [ET.NO_ENTRY])):
        events.append(create_event('buttonEnable', [ET.ENABLE]))
        self.last_enable_sent = cur_time
    elif enable_pressed:
      events.append(create_event('buttonEnable', [ET.ENABLE]))

    ret.events = events
    ret.canMonoTimes = canMonoTimes

    # update previous brake/gas pressed
    self.gas_pressed_prev = ret.gasPressed
    self.brake_pressed_prev = ret.brakePressed

    # cast to reader so it can't be modified
    return ret.as_reader()

  # pass in a car.CarControl
  # to be called @ 100hz
  def apply(self, c, perception_state=log.Live20Data.new_message()):
    if c.hudControl.speedVisible:
      hud_v_cruise = c.hudControl.setSpeed * CV.MS_TO_KPH
    else:
      hud_v_cruise = 255

    hud_alert = {
      "none": AH.NONE,
      "fcw": AH.FCW,
      "steerRequired": AH.STEER,
      "brakePressed": AH.BRAKE_PRESSED,
      "wrongGear": AH.GEAR_NOT_D,
      "seatbeltUnbuckled": AH.SEATBELT,
      "speedTooHigh": AH.SPEED_TOO_HIGH}[str(c.hudControl.visualAlert)]

    snd_beep, snd_chime = {
      "none": (BP.MUTE, CM.MUTE),
      "beepSingle": (BP.SINGLE, CM.MUTE),
      "beepTriple": (BP.TRIPLE, CM.MUTE),
      "beepRepeated": (BP.REPEATED, CM.MUTE),
      "chimeSingle": (BP.MUTE, CM.SINGLE),
      "chimeDouble": (BP.MUTE, CM.DOUBLE),
      "chimeRepeated": (BP.MUTE, CM.REPEATED),
      "chimeContinuous": (BP.MUTE, CM.CONTINUOUS)}[str(c.hudControl.audibleAlert)]

    pcm_accel = int(clip(c.cruiseControl.accelOverride,0,1)*0xc6)

    self.CC.update(self.sendcan, c.enabled, self.CS, self.frame, \
      c.actuators, \
      c.cruiseControl.speedOverride, \
      c.cruiseControl.override, \
      c.cruiseControl.cancel, \
      pcm_accel, \
      perception_state.radarErrors, \
      hud_v_cruise, c.hudControl.lanesVisible, \
      hud_show_car = c.hudControl.leadVisible, \
      hud_alert = hud_alert, \
      snd_beep = snd_beep, \
      snd_chime = snd_chime)

    self.frame += 1
