#!/usr/bin/env python3
from cereal import car
from selfdrive.car.chrysler.values import CAR
from selfdrive.car import STD_CARGO_KG, scale_rot_inertia, scale_tire_stiffness, gen_empty_fingerprint, get_safety_config
from selfdrive.car.interfaces import CarInterfaceBase
from selfdrive.controls.lib.latcontrol_torque import set_torque_tune


class CarInterface(CarInterfaceBase):
  @staticmethod
  def get_params(candidate, fingerprint=gen_empty_fingerprint(), car_fw=None, disable_radar=False):
    ret = CarInterfaceBase.get_std_params(candidate, fingerprint)
    ret.carName = "chrysler"
    ret.safetyConfigs = [get_safety_config(car.CarParams.SafetyModel.chrysler)]

    # Speed conversion:              20, 45 mph
    ret.wheelbase = 3.089  # in meters for Pacifica Hybrid 2017
    ret.steerRatio = 16.2  # Pacifica 16.2 AWD 15.7 https://s3.amazonaws.com/chryslermedia.iconicweb.com/mediasite/specs/2021_CH_Pacifica_Specificationsq2sglr6p0lk07r8c4g3v16c97t.pdf
    ret.mass = 2242. + STD_CARGO_KG  # kg curb weight Pacifica Hybrid 2017
    ret.lateralTuning.pid.kpBP, ret.lateralTuning.pid.kiBP = [[9., 20.], [9., 20.]]
    ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.15, 0.30], [0.03, 0.05]]
    ret.lateralTuning.pid.kf = 0.00006   # full torque for 10 deg at 80mph means 0.00007818594
    ret.steerActuatorDelay = 0.1
    ret.steerRateCost = 0.7
    ret.steerLimitTimer = 0.4
    ret.minSteerSpeed = 3.8  # m/s

    if candidate in (CAR.JEEP_CHEROKEE, CAR.JEEP_CHEROKEE_2019):
      ret.wheelbase = 2.91  # in meters
      ret.steerRatio = 16.7 #  2020 17.9:1 (V-6 4x2); 16.5:1 on SRT and Trackhawk; 16.7:1 (all other vehicles)  https://s3.amazonaws.com/chryslermedia.iconicweb.com/mediasite/specs/2020_JP_Grand_Cherokee_SPmar9cqpguibpb9l0c26hemi38d.pdf
      ret.steerActuatorDelay = 0.2  # in seconds

    ret.centerToFront = ret.wheelbase * 0.44

    if candidate in (CAR.RAM_1500):
      ret.wheelbase = 3.88  # 2021 Ram 1500
      ret.steerRatio = 16.3  # Overall Ratio from https://s3.amazonaws.com/chryslermedia.iconicweb.com/mediasite/specs/2019_Ram_1500_SP160igecpp6jn85geq3o0r4cs90.pdf
      ret.mass = 2493. + STD_CARGO_KG  # kg curb weight 2021 Ram 1500
      MAX_LAT_ACCEL = 2.6
      FRICTION = .05
      ret.steerActuatorDelay = 0.1
      ret.steerRateCost = 1.0
      ret.centerToFront = ret.wheelbase * 0.4 # just a guess
      ret.minSteerSpeed = 14.5
      set_torque_tune(ret.lateralTuning, MAX_LAT_ACCEL, FRICTION)

    # TODO: start from empirically derived lateral slip stiffness for the civic and scale by
    # mass and CG position, so all cars will have approximately similar dyn behaviors
    ret.tireStiffnessFront, ret.tireStiffnessRear = scale_tire_stiffness(ret.mass, ret.wheelbase, ret.centerToFront)

    if candidate in (CAR.RAM_2500):
      front_stiffness = 0.36 # want to change these so that front / rear stiffness ratio is learned at ~1.0
      rear_stiffness = 0.36
      ret.wheelbase = 3.785  # in meters
      ret.steerRatio = 15.61  # just a guess
      ret.mass = 3405. + STD_CARGO_KG  # kg curb weight 2021 Ram 2500
      MAX_LAT_ACCEL = 1.0
      FRICTION = .05
      ret.steerActuatorDelay = 0.1
      ret.steerRateCost = 1.0  # may need tuning
      ret.centerToFront = ret.wheelbase * 0.38 # calculated from 100% - (front axle weight/total weight)
      ret.minSteerSpeed = 16.0
      set_torque_tune(ret.lateralTuning, MAX_LAT_ACCEL, FRICTION)


      ret.tireStiffnessFront, ret.tireStiffnessRear = ret.tireStiffnessFront*front_stiffness, ret.tireStiffnessRear*rear_stiffness


    if candidate in (CAR.PACIFICA_2019_HYBRID, CAR.PACIFICA_2020, CAR.JEEP_CHEROKEE_2019):
      # TODO allow 2019 cars to steer down to 13 m/s if already engaged.
      ret.minSteerSpeed = 17.5  # m/s 17 on the way up, 13 on the way down once engaged.

    # starting with reasonable value for civic and scaling by mass and wheelbase
    ret.rotationalInertia = scale_rot_inertia(ret.mass, ret.wheelbase)

    ret.enableBsm = 720 in fingerprint[0]

    return ret

  # returns a car.CarState
  def _update(self, c):
    ret = self.CS.update(self.cp, self.cp_cam)

    ret.steeringRateLimited = self.CC.steer_rate_limited if self.CC is not None else False

    # events
    events = self.create_common_events(ret, extra_gears=[car.CarState.GearShifter.low])

    # Low speed steer alert hysteresis logic
    if self.CP.minSteerSpeed > 0. and ret.vEgo < (self.CP.minSteerSpeed -0.5):
      self.low_speed_alert = True
    elif ret.vEgo > (self.CP.minSteerSpeed):
      self.low_speed_alert = False
    if self.low_speed_alert:
      events.add(car.CarEvent.EventName.belowSteerSpeed)

    ret.events = events.to_msg()

    return ret

  # pass in a car.CarControl
  # to be called @ 100hz
  def apply(self, c):
    return self.CC.update(c, self.CS)
