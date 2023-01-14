from cereal import car
from selfdrive.car.hongqi.values import CAR, GearShifter
from selfdrive.car import STD_CARGO_KG, scale_rot_inertia, scale_tire_stiffness, get_safety_config
from selfdrive.car.interfaces import CarInterfaceBase

EventName = car.CarEvent.EventName


class CarInterface(CarInterfaceBase):
  # def __init__(self, CP, CarController, CarState):
  #   super().__init__(CP, CarController, CarState)

  @staticmethod
  def _get_params(ret, candidate, fingerprint, car_fw, experimental_long):
    ret.carName = "hongqi"
    ret.radarOffCan = True

    # Set global Hongqi parameters
    ret.safetyConfigs = [get_safety_config(car.CarParams.SafetyModel.hongqi)]
    # TODO: identify BSM signals
    # ret.enableBsm = 0x30F in fingerprint[0]  # SWA_01

    # Global lateral tuning defaults, can be overridden per-vehicle

    ret.steerActuatorDelay = 0.1
    ret.steerLimitTimer = 0.4
    ret.steerRatio = 15.6  # Let the params learner figure this out
    tire_stiffness_factor = 1.0  # Let the params learner figure this out
    CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)

    # Per-chassis tuning values, override tuning defaults here if desired

    if candidate == CAR.HS5_G1:
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
  def _update(self, c):
    ret = self.CS.update(self.cp, self.cp_cam)

    events = self.create_common_events(ret, extra_gears=[GearShifter.eco, GearShifter.sport, GearShifter.manumatic])
    ret.events = events.to_msg()

    return ret

  def apply(self, c):
    ret = self.CC.update(c, self.CS, self.frame, c.actuators)
    self.frame += 1
    return ret
