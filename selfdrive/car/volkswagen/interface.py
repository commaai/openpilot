from cereal import car
from selfdrive.swaglog import cloudlog
from selfdrive.car.volkswagen.values import ATTRIBUTES, BUTTON_STATES, TransmissionType, GearShifter
from selfdrive.car import STD_CARGO_KG, scale_rot_inertia, scale_tire_stiffness, gen_empty_fingerprint
from selfdrive.car.interfaces import CarInterfaceBase

EventName = car.CarEvent.EventName


class CarInterface(CarInterfaceBase):
  def __init__(self, CP, CarController, CarState):
    super().__init__(CP, CarController, CarState)

    self.displayMetricUnitsPrev = None
    self.buttonStatesPrev = BUTTON_STATES.copy()

  @staticmethod
  def compute_gb(accel, speed):
    return float(accel) / 4.0

  @staticmethod
  def get_params(candidate, fingerprint=gen_empty_fingerprint(), car_fw=None):
    ret = CarInterfaceBase.get_std_params(candidate, fingerprint)

    ret.communityFeature = True  # VW port is a community feature
    ret.enableCamera = True
    ret.radarOffCan = True
    ret.steerLimitTimer = 0.4

    if True:  # pylint: disable=using-constant-test
      # Set common MQB parameters that will apply globally
      ret.carName = "volkswagen"
      ret.safetyModel = car.CarParams.SafetyModel.volkswagen

      if 0xAD in fingerprint[0]:  # Getriebe_11
        ret.transmissionType = TransmissionType.automatic
      elif 0x187 in fingerprint[0]:  # EV_Gearshift
        ret.transmissionType = TransmissionType.direct
      else:  # Manual trans vehicles don't have a trans message
        ret.transmissionType = TransmissionType.manual
      cloudlog.info("Detected transmission type: %s", ret.transmissionType)

      ret.enableBsm = 0x30F in fingerprint[0]  # SWA_01

    # Required per-CAR attributes
    ret.mass = ATTRIBUTES[candidate]["mass"] + STD_CARGO_KG
    ret.wheelbase = ATTRIBUTES[candidate]["wheelbase"]
    ret.centerToFront = ret.wheelbase * 0.45
    # Optional per-CAR attributes, with defaults
    ret.steerActuatorDelay = ATTRIBUTES[candidate].setdefault("steer_actuator_delay", 0.05)  # Seems good for most MQB
    ret.steerRatio = ATTRIBUTES[candidate].setdefault("steer_ratio", 15.6)  # Updated by params learner

    # Tuning values, currently using the same tune for all MQB
    # If we need to tune individual models, we'll need a dict lookup by EPS parameterization, not just CAR
    ret.steerRateCost = 1.0
    tire_stiffness_factor = 1.0  # Updated by params learner
    [ ret.lateralTuning.pid.kpBP, ret.lateralTuning.pid.kiBP,
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV,
      ret.lateralTuning.pid.kf ] = ([0.], [0.], [0.6], [0.2], 0.00006)

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

    ret = self.CS.update(self.cp, self.cp_cam, self.CP.transmissionType)
    ret.canValid = self.cp.can_valid and self.cp_cam.can_valid
    ret.steeringRateLimited = self.CC.steer_rate_limited if self.CC is not None else False

    # TODO: add a field for this to carState, car interface code shouldn't write params
    # Update the device metric configuration to match the car at first startup,
    # or if there's been a change.
    #if self.CS.displayMetricUnits != self.displayMetricUnitsPrev:
    #  put_nonblocking("IsMetric", "1" if self.CS.displayMetricUnits else "0")

    # Check for and process state-change events (button press or release) from
    # the turn stalk switch or ACC steering wheel/control stalk buttons.
    for button in self.CS.buttonStates:
      if self.CS.buttonStates[button] != self.buttonStatesPrev[button]:
        be = car.CarState.ButtonEvent.new_message()
        be.type = button
        be.pressed = self.CS.buttonStates[button]
        buttonEvents.append(be)

    events = self.create_common_events(ret, extra_gears=[GearShifter.eco, GearShifter.sport, GearShifter.manumatic])

    # Vehicle health and operation safety checks
    if self.CS.parkingBrakeSet:
      events.add(EventName.parkBrake)

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
                   c.hudControl.leftLaneVisible,
                   c.hudControl.rightLaneVisible,
                   c.hudControl.leftLaneDepart,
                   c.hudControl.rightLaneDepart)
    self.frame += 1
    return can_sends
