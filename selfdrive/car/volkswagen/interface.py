from cereal import car
from selfdrive.config import Conversions as CV
from selfdrive.controls.lib.drive_helpers import create_event, EventTypes as ET
from selfdrive.car.volkswagen.values import CAR, BUTTON_STATES
from common.params import put_nonblocking
from selfdrive.car import STD_CARGO_KG, scale_rot_inertia, scale_tire_stiffness, gen_empty_fingerprint
from selfdrive.car.interfaces import CarInterfaceBase

GEAR = car.CarState.GearShifter

class CarInterface(CarInterfaceBase):
  def __init__(self, CP, CarController, CarState):
    super().__init__(CP, CarController, CarState)

    self.displayMetricUnitsPrev = None
    self.buttonStatesPrev = BUTTON_STATES.copy()

  @staticmethod
  def compute_gb(accel, speed):
    return float(accel) / 4.0

  @staticmethod
  def get_params(candidate, fingerprint=gen_empty_fingerprint(), has_relay=False, car_fw=[]):
    ret = CarInterfaceBase.get_std_params(candidate, fingerprint, has_relay)

    # VW port is a community feature, since we don't own one to test
    ret.communityFeature = True

    if candidate == CAR.GOLF:
      # Set common MQB parameters that will apply globally
      ret.carName = "volkswagen"
      ret.radarOffCan = True
      ret.safetyModel = car.CarParams.SafetyModel.volkswagen

      # Additional common MQB parameters that may be overridden per-vehicle
      ret.steerRateCost = 0.5
      ret.steerActuatorDelay = 0.05 # Hopefully all MQB racks are similar here
      ret.steerLimitTimer = 0.4

      # As a starting point for speed-adjusted lateral tuning, use the example
      # map speed breakpoints from a VW Tiguan (SSP 399 page 9). It's unclear
      # whether the driver assist map breakpoints have any direct bearing on
      # HCA assist torque, but if they're good breakpoints for the driver,
      # they're probably good breakpoints for HCA as well. OP won't be driving
      # 250kph/155mph but it provides interpolation scaling above 100kmh/62mph.
      ret.lateralTuning.pid.kpBP = [0., 15 * CV.KPH_TO_MS, 50 * CV.KPH_TO_MS]
      ret.lateralTuning.pid.kiBP = [0., 15 * CV.KPH_TO_MS, 50 * CV.KPH_TO_MS]

      # FIXME: Per-vehicle parameters need to be reintegrated.
      # For the time being, per-vehicle stuff is being archived since we
      # can't auto-detect very well yet. Now that tuning is figured out,
      # averaged params should work reasonably on a range of cars. Owners
      # can tweak here, as needed, until we have car type auto-detection.

      ret.mass = 1700 + STD_CARGO_KG
      ret.wheelbase = 2.75
      ret.centerToFront = ret.wheelbase * 0.45
      ret.steerRatio = 15.6
      ret.lateralTuning.pid.kf = 0.00006
      ret.lateralTuning.pid.kpV = [0.15, 0.25, 0.60]
      ret.lateralTuning.pid.kiV = [0.05, 0.05, 0.05]
      tire_stiffness_factor = 0.6

    ret.enableCamera = True # Stock camera detection doesn't apply to VW
    ret.transmissionType = car.CarParams.TransmissionType.automatic

    # TODO: get actual value, for now starting with reasonable value for
    # civic and scaling by mass and wheelbase
    ret.rotationalInertia = scale_rot_inertia(ret.mass, ret.wheelbase)

    # TODO: start from empirically derived lateral slip stiffness for the civic and scale by
    # mass and CG position, so all cars will have approximately similar dyn behaviors
    ret.tireStiffnessFront, ret.tireStiffnessRear = scale_tire_stiffness(ret.mass, ret.wheelbase, ret.centerToFront,
                                                                         tire_stiffness_factor=tire_stiffness_factor)

    return ret

  # returns a car.CarState
  def update(self, c, can_strings):
    canMonoTimes = []
    buttonEvents = []

    # Process the most recent CAN message traffic, and check for validity
    # The camera CAN has no signals we use at this time, but we process it
    # anyway so we can test connectivity with can_valid
    self.cp.update_strings(can_strings)
    self.cp_cam.update_strings(can_strings)

    ret = self.CS.update(self.cp)
    ret.canValid = self.cp.can_valid and self.cp_cam.can_valid
    ret.steeringRateLimited = self.CC.steer_rate_limited if self.CC is not None else False

    # Update the EON metric configuration to match the car at first startup,
    # or if there's been a change.
    if self.CS.displayMetricUnits != self.displayMetricUnitsPrev:
      put_nonblocking("IsMetric", "1" if self.CS.displayMetricUnits else "0")

    # Check for and process state-change events (button press or release) from
    # the turn stalk switch or ACC steering wheel/control stalk buttons.
    for button in self.CS.buttonStates:
      if self.CS.buttonStates[button] != self.buttonStatesPrev[button]:
        be = car.CarState.ButtonEvent.new_message()
        be.type = button
        be.pressed = self.CS.buttonStates[button]
        buttonEvents.append(be)

    events = self.create_common_events(ret, extra_gears=[GEAR.eco, GEAR.sport])

    # Vehicle health and operation safety checks
    if self.CS.parkingBrakeSet:
      events.append(create_event('parkBrake', [ET.NO_ENTRY, ET.USER_DISABLE]))
    if self.CS.steeringFault:
      events.append(create_event('steerTempUnavailable', [ET.NO_ENTRY, ET.WARNING]))

    # Engagement and longitudinal control using stock ACC. Make sure OP is
    # disengaged if stock ACC is disengaged.
    if not ret.cruiseState.enabled:
      events.append(create_event('pcmDisable', [ET.USER_DISABLE]))
    # Attempt OP engagement only on rising edge of stock ACC engagement.
    elif not self.cruise_enabled_prev:
      events.append(create_event('pcmEnable', [ET.ENABLE]))

    ret.events = events
    ret.buttonEvents = buttonEvents
    ret.canMonoTimes = canMonoTimes

    # update previous car states
    self.gas_pressed_prev = ret.gasPressed
    self.brake_pressed_prev = ret.brakePressed
    self.cruise_enabled_prev = ret.cruiseState.enabled
    self.displayMetricUnitsPrev = self.CS.displayMetricUnits
    self.buttonStatesPrev = self.CS.buttonStates.copy()

    self.CS.out = ret.as_reader()
    return self.CS.out

  def apply(self, c):
    can_sends = self.CC.update(c.enabled, self.CS, self.frame, c.actuators,
                   c.hudControl.visualAlert,
                   c.hudControl.audibleAlert,
                   c.hudControl.leftLaneVisible,
                   c.hudControl.rightLaneVisible)
    self.frame += 1
    return can_sends
