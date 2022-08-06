from cereal import car
from selfdrive.car.faw.values import CAR
from selfdrive.car import STD_CARGO_KG, scale_rot_inertia, scale_tire_stiffness, gen_empty_fingerprint, get_safety_config
from selfdrive.car.interfaces import CarInterfaceBase

EventName = car.CarEvent.EventName


class CarInterface(CarInterfaceBase):
  #def __init__(self, CP, CarController, CarState):
  #  super().__init__(CP, CarController, CarState)

  @staticmethod
  def get_params(candidate, fingerprint=gen_empty_fingerprint(), car_fw=None, disable_radar=False):
    ret = CarInterfaceBase.get_std_params(candidate, fingerprint)
    ret.carName = "faw"
    ret.radarOffCan = True

    # Set global FAW parameters
    ret.safetyConfigs = [get_safety_config(car.CarParams.SafetyModel.faw)]
    # TODO: identify BSM signals
    # ret.enableBsm = 0x30F in fingerprint[0]  # SWA_01

    # Global lateral tuning defaults, can be overridden per-vehicle

    ret.steerActuatorDelay = 0.1
    ret.steerLimitTimer = 0.4
    ret.steerRatio = 15.6  # Let the params learner figure this out
    tire_stiffness_factor = 1.0  # Let the params learner figure this out
    ret.lateralTuning.pid.kpBP = [0.]
    ret.lateralTuning.pid.kiBP = [0.]
    ret.lateralTuning.pid.kf = 0.00006
    ret.lateralTuning.pid.kpV = [0.6]
    ret.lateralTuning.pid.kiV = [0.2]

    # Per-chassis tuning values, override tuning defaults here if desired

    if candidate == CAR.HONGQI_HS5_G1:
      ret.mass = 1780 + STD_CARGO_KG
      ret.wheelbase = 2.87

    else:
      raise ValueError(f"unsupported car {candidate}")

    ret.rotationalInertia = scale_rot_inertia(ret.mass, ret.wheelbase)
    ret.centerToFront = ret.wheelbase * 0.45
    ret.tireStiffnessFront, ret.tireStiffnessRear = scale_tire_stiffness(ret.mass, ret.wheelbase, ret.centerToFront,
                                                                         tire_stiffness_factor=tire_stiffness_factor)
    return ret

  # returns a car.CarState
  def update(self, c, can_strings):
    # buttonEvents = []

    # Process the most recent CAN message traffic, and check for validity
    # The camera CAN has no signals we use at this time, but we process it
    # anyway so we can test connectivity with can_valid
    self.cp.update_strings(can_strings)
    self.cp_cam.update_strings(can_strings)

    ret = self.CS.update(self.cp, self.cp_cam)
    # FIXME: temp bypass
    # ret.canValid = self.cp.can_valid and self.cp_cam.can_valid
    ret.canValid = True
    ret.steeringRateLimited = self.CC.steer_rate_limited if self.CC is not None else False

    # Check for and process state-change events (button press or release) from
    # the turn stalk switch or ACC steering wheel/control stalk buttons.
    # TODO
    # for button in self.CS.buttonStates:
    #  if self.CS.buttonStates[button] != self.buttonStatesPrev[button]:
    #    be = car.CarState.ButtonEvent.new_message()
    #    be.type = button
    #    be.pressed = self.CS.buttonStates[button]
    #    buttonEvents.append(be)

    events = self.create_common_events(ret)

    # Low speed steer alert hysteresis logic
    # TODO verify min steer speed
    # if self.CP.minSteerSpeed > 0. and ret.vEgo < (self.CP.minSteerSpeed + 1.):
    #  self.low_speed_alert = True
    #elif ret.vEgo > (self.CP.minSteerSpeed + 2.):
    #  self.low_speed_alert = False
    #if self.low_speed_alert:
    #  events.add(EventName.belowSteerSpeed)

    ret.events = events.to_msg()
    # ret.buttonEvents = buttonEvents

    # update previous car states
    # self.buttonStatesPrev = self.CS.buttonStates.copy()

    self.CS.out = ret.as_reader()
    return self.CS.out

  def apply(self, c):
    ret = self.CC.update(c, self.CS, self.frame, c.actuators)
    self.frame += 1
    return ret
