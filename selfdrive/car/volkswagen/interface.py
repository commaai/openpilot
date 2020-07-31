from cereal import car
from selfdrive.swaglog import cloudlog
from selfdrive.car.volkswagen.values import CAR, BUTTON_STATES, NWL, TRANS, GEAR
from common.params import put_nonblocking
from selfdrive.car import STD_CARGO_KG, scale_rot_inertia, scale_tire_stiffness, gen_empty_fingerprint
from selfdrive.car.interfaces import CarInterfaceBase

EventName = car.CarEvent.EventName

class CarInterface(CarInterfaceBase):
  def __init__(self, CP, CarController, CarState):
    super().__init__(CP, CarController, CarState)

    self.displayMetricUnitsPrev = None
    self.buttonStatesPrev = BUTTON_STATES.copy()

    # Set up an alias to PT/CAM parser for ACC depending on its detected network location
    self.cp_acc = self.cp if CP.networkLocation == NWL.fwdCamera else self.cp_cam

  @staticmethod
  def compute_gb(accel, speed):
    return float(accel) / 4.0

  @staticmethod
  def get_params(candidate, fingerprint=gen_empty_fingerprint(), has_relay=False, car_fw=None):
    ret = CarInterfaceBase.get_std_params(candidate, fingerprint, has_relay)

    ret.enableCamera = True  # Stock camera detection doesn't apply to VW
    ret.carName = "volkswagen"
    ret.radarOffCan = True

    if candidate == CAR.GENERICMQB:
      # Set common MQB parameters that will apply globally
      ret.safetyModel = car.CarParams.SafetyModel.volkswagen

      # Additional common MQB parameters that may be overridden per-vehicle
      ret.steerRateCost = 1.0
      ret.steerActuatorDelay = 0.1  # Hopefully all MQB racks are similar here
      ret.steerLimitTimer = 0.4

      ret.lateralTuning.pid.kpBP = [0.]
      ret.lateralTuning.pid.kiBP = [0.]

      # FIXME: Per-vehicle parameters need to be reintegrated.
      # Until that time, defaulting to VW Golf Mk7 as a baseline.
      ret.mass = 1500 + STD_CARGO_KG
      ret.wheelbase = 2.64
      ret.centerToFront = ret.wheelbase * 0.45
      ret.steerRatio = 15.6
      ret.lateralTuning.pid.kf = 0.00006
      ret.lateralTuning.pid.kpV = [0.6]
      ret.lateralTuning.pid.kiV = [0.2]
      tire_stiffness_factor = 1.0

    # Determine installed network location by checking for radar-camera
    # sensor fusion messages on bus 1.
    if 0x238 in fingerprint[1]:
      ret.networkLocation = NWL.fwdCamera
    else:
      ret.networkLocation = NWL.gateway

    # Determine transmission type by CAN message(s) present on the bus
    if 0xAD in fingerprint[0]:
      # Getribe_11 message detected: traditional automatic or DSG gearbox
      ret.transmissionType = TRANS.automatic
    elif 0x187 in fingerprint[0]:
      # EV_Gearshift message detected: e-Golf or similar direct-drive electric
      ret.transmissionType = TRANS.direct
    else:
      # No trans message at all, must be a true stick-shift manual
      ret.transmissionType = TRANS.manual

    cloudlog.warning("Detected network location: %s", ret.networkLocation)
    cloudlog.warning("Detected transmission type: %s", ret.transmissionType)

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
    buttonEvents = []

    # Process the most recent CAN message traffic, and check for validity
    # The camera CAN has no signals we use at this time, but we process it
    # anyway so we can test connectivity with can_valid
    self.cp.update_strings(can_strings)
    self.cp_cam.update_strings(can_strings)

    ret = self.CS.update(self.cp, self.cp_cam, self.cp_acc, self.CP.transmissionType)
    ret.canValid = self.cp.can_valid  # FIXME: Restore cp_cam valid check after proper LKAS camera detect
    ret.steeringRateLimited = self.CC.steer_rate_limited if self.CC is not None else False

    # Update the device metric configuration to match the car at first startup,
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
      events.add(EventName.parkBrake)
    if self.CS.steeringFault:
      events.add(EventName.steerTempUnavailable)

    ret.events = events.to_msg()
    ret.buttonEvents = buttonEvents

    # update previous car states
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
