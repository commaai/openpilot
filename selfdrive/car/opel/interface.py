# Opel Corsa F (PSA CMP Platform) - Car Interface
# Physical parameters, lateral tuning, and main interface logic
from cereal import car
from selfdrive.car.opel.values import CAR, BUTTON_STATES, CANBUS, NetworkLocation, TransmissionType, GearShifter
from selfdrive.car import STD_CARGO_KG, scale_rot_inertia, scale_tire_stiffness, gen_empty_fingerprint, get_safety_config
from selfdrive.car.interfaces import CarInterfaceBase

EventName = car.CarEvent.EventName


class CarInterface(CarInterfaceBase):
  def __init__(self, CP, CarController, CarState):
    super().__init__(CP, CarController, CarState)

    self.displayMetricUnitsPrev = None
    self.buttonStatesPrev = BUTTON_STATES.copy()

    if CP.networkLocation == NetworkLocation.fwdCamera:
      self.ext_bus = CANBUS.pt
      self.cp_ext = self.cp
    else:
      self.ext_bus = CANBUS.cam
      self.cp_ext = self.cp_cam

  @staticmethod
  def get_params(candidate, fingerprint=gen_empty_fingerprint(), car_fw=None):
    ret = CarInterfaceBase.get_std_params(candidate, fingerprint)
    ret.carName = "opel"
    ret.radarOffCan = True

    # Safety model - using allOutput for research and simulation
    # In a production environment, a specific PSA safety model should be used
    ret.safetyConfigs = [get_safety_config(car.CarParams.SafetyModel.allOutput)]

    # Determine transmission type from CAN fingerprint
    if 0x131 in fingerprint[0]:  # Transmission gear message
      ret.transmissionType = TransmissionType.automatic
    else:
      ret.transmissionType = TransmissionType.manual

    # Determine network location
    ret.networkLocation = NetworkLocation.gateway

    # Global lateral tuning defaults
    ret.steerActuatorDelay = 0.1
    ret.steerRateCost = 1.0
    ret.steerLimitTimer = 0.4
    ret.steerRatio = 14.7
    tire_stiffness_factor = 1.0

    # Lateral tuning - Torque-based control is generally better for CMP platform
    ret.lateralTuning.init('torque')
    ret.lateralTuning.torque.useSteeringAngle = True
    ret.lateralTuning.torque.kp = 1.0
    ret.lateralTuning.torque.kf = 1.0
    ret.lateralTuning.torque.ki = 0.1
    ret.lateralTuning.torque.friction = 0.01

    # Per-vehicle physical parameters
    if candidate == CAR.CORSA_F:
      ret.mass = 1200 + STD_CARGO_KG        # Kerb weight ~1200 kg
      ret.wheelbase = 2.538                   # 2538 mm wheelbase
      ret.steerRatio = 14.7
      ret.centerToFront = ret.wheelbase * 0.44  # Slightly more forward weight
    elif candidate == CAR.PEUGEOT_208:
      ret.mass = 1150 + STD_CARGO_KG        # Kerb weight ~1150 kg
      ret.wheelbase = 2.540                   # 2540 mm wheelbase
      ret.steerRatio = 14.7
      ret.centerToFront = ret.wheelbase * 0.45
    elif candidate == CAR.PEUGEOT_2008:
      ret.mass = 1200 + STD_CARGO_KG        # Kerb weight ~1200 kg
      ret.wheelbase = 2.605                   # 2605 mm wheelbase
      ret.steerRatio = 14.5
      ret.centerToFront = ret.wheelbase * 0.45
    else:
      raise ValueError(f"unsupported car {candidate}")

    ret.rotationalInertia = scale_rot_inertia(ret.mass, ret.wheelbase)
    ret.centerToFront = ret.wheelbase * 0.45
    ret.tireStiffnessFront, ret.tireStiffnessRear = scale_tire_stiffness(
      ret.mass, ret.wheelbase, ret.centerToFront,
      tire_stiffness_factor=tire_stiffness_factor
    )
    return ret

  # returns a car.CarState
  def update(self, c, can_strings):
    buttonEvents = []

    # Process the most recent CAN message traffic
    self.cp.update_strings(can_strings)
    self.cp_cam.update_strings(can_strings)

    ret = self.CS.update(self.cp, self.cp_cam, self.cp_ext, self.CP.transmissionType)
    ret.canValid = self.cp.can_valid and self.cp_cam.can_valid
    ret.steeringRateLimited = self.CC.steer_rate_limited if self.CC is not None else False

    # Check for button state changes
    for button in self.CS.buttonStates:
      if self.CS.buttonStates[button] != self.buttonStatesPrev[button]:
        be = car.CarState.ButtonEvent.new_message()
        be.type = button
        be.pressed = self.CS.buttonStates[button]
        buttonEvents.append(be)

    events = self.create_common_events(ret, extra_gears=[GearShifter.eco, GearShifter.sport, GearShifter.manumatic])

    # Low speed steer alert hysteresis logic
    if self.CP.minSteerSpeed > 0. and ret.vEgo < (self.CP.minSteerSpeed + 1.):
      self.low_speed_alert = True
    elif ret.vEgo > (self.CP.minSteerSpeed + 2.):
      self.low_speed_alert = False
    if self.low_speed_alert:
      events.add(EventName.belowSteerSpeed)

    ret.events = events.to_msg()
    ret.buttonEvents = buttonEvents

    # Update previous car states
    self.displayMetricUnitsPrev = self.CS.displayMetricUnits
    self.buttonStatesPrev = self.CS.buttonStates.copy()

    self.CS.out = ret.as_reader()
    return self.CS.out

  def apply(self, c):
    hud_control = c.hudControl
    ret = self.CC.update(c, self.CS, self.frame, self.ext_bus, c.actuators,
                         hud_control.visualAlert,
                         hud_control.leftLaneVisible,
                         hud_control.rightLaneVisible,
                         hud_control.leftLaneDepart,
                         hud_control.rightLaneDepart)
    self.frame += 1
    return ret
