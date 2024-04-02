from cereal import car
from panda import Panda
from openpilot.selfdrive.car import get_safety_config
from openpilot.selfdrive.car.interfaces import CarInterfaceBase
from openpilot.selfdrive.car.volkswagen.values import CAR, CANBUS, NetworkLocation, TransmissionType, GearShifter, VolkswagenFlags, VolkswagenCarSpecs

ButtonType = car.CarState.ButtonEvent.Type
EventName = car.CarEvent.EventName


class CarInterface(CarInterfaceBase):
  def __init__(self, CP, CarController, CarState):
    super().__init__(CP, CarController, CarState)

    if CP.networkLocation == NetworkLocation.fwdCamera:
      self.ext_bus = CANBUS.pt
      self.cp_ext = self.cp
    else:
      self.ext_bus = CANBUS.cam
      self.cp_ext = self.cp_cam

  @staticmethod
  def _get_params(ret, candidate: CAR, fingerprint, car_fw, experimental_long, docs):
    ret.carName = "volkswagen"
    ret.radarUnavailable = True

    SPECS_MAP = {
      ('AN',): VolkswagenCarSpecs(mass=1733, wheelbase=2.84),
      ('CA',): VolkswagenCarSpecs(mass=2011, wheelbase=2.98),
      ('2K',): VolkswagenCarSpecs(mass=1613, wheelbase=2.6, minSteerSpeed=21 * CV.KPH_TO_MS),
      ('SY', 'SZ',): VolkswagenCarSpecs(mass=2100, wheelbase=3.64, minSteerSpeed=50 * CV.KPH_TO_MS),
      ('5G', 'AU', 'BA', 'BE', 'BU'): VolkswagenCarSpecs(mass=1366, wheelbase=2.62),
      ('BU',): VolkswagenCarSpecs(mass=1328, wheelbase=2.71),
      ('3G',): VolkswagenCarSpecs(mass=1528, wheelbase=2.82),
      ('A3',): VolkswagenCarSpecs(mass=1503, wheelbase=2.80, minSteerSpeed=50*CV.KPH_TO_MS, minEnableSpeed=20*CV.KPH_TO_MS),
      ('AW',): VolkswagenCarSpecs(mass=1230, wheelbase=2.55),
      ('7N',): VolkswagenCarSpecs(mass=1639, wheelbase=2.92, minSteerSpeed=50*CV.KPH_TO_MS),
      ('B2',):  VolkswagenCarSpecs(mass=1498, wheelbase=2.69),
      ('C1',): VolkswagenCarSpecs(mass=1150, wheelbase=2.60),
      ('AD', 'BW',): VolkswagenCarSpecs(mass=1715, wheelbase=2.74),
      ('1T',): VolkswagenCarSpecs(mass=1516, wheelbase=2.79),
      ('7H', '7L',): VolkswagenCarSpecs(mass=1926, wheelbase=3.00, minSteerSpeed=14.0),
      ('A1',): VolkswagenCarSpecs(mass=1413, wheelbase=2.63),
      ('GA',): VolkswagenCarSpecs(mass=1205, wheelbase=2.61),
      ('8U', 'F3', 'FS',): VolkswagenCarSpecs(mass=1623, wheelbase=2.68),
      ('5F',):VolkswagenCarSpecs(mass=1900, wheelbase=2.64),  # TODO: UH OH
      ('5F',): VolkswagenCarSpecs(mass=1227, wheelbase=2.64),
      ('PJ',): VolkswagenCarSpecs(mass=1266, wheelbase=2.56),
      ('NW',): VolkswagenCarSpecs(mass=1265, wheelbase=2.66),
      ('NU',): VolkswagenCarSpecs(mass=1278, wheelbase=2.66),
      ('NS',): VolkswagenCarSpecs(mass=1569, wheelbase=2.79),
      ('NE',): VolkswagenCarSpecs(mass=1388, wheelbase=2.68),
      ('NW',): VolkswagenCarSpecs(mass=1192, wheelbase=2.65),
    }

    chassis_code = ret.carVin[6:8]
    specs = next((SPECS_MAP[codes] for codes in SPECS_MAP if chassis_code in codes), None)
    if specs is not None:
      ret.flags &= VolkswagenFlags.SPECS_SET.value
      ret.mass = specs.mass
      ret.wheelbase = specs.wheelbase

    if ret.flags & VolkswagenFlags.PQ:
      # Set global PQ35/PQ46/NMS parameters
      ret.safetyConfigs = [get_safety_config(car.CarParams.SafetyModel.volkswagenPq)]
      ret.enableBsm = 0x3BA in fingerprint[0]  # SWA_1

      if 0x440 in fingerprint[0] or docs:  # Getriebe_1
        ret.transmissionType = TransmissionType.automatic
      else:
        ret.transmissionType = TransmissionType.manual

      if any(msg in fingerprint[1] for msg in (0x1A0, 0xC2)):  # Bremse_1, Lenkwinkel_1
        ret.networkLocation = NetworkLocation.gateway
      else:
        ret.networkLocation = NetworkLocation.fwdCamera

      # The PQ port is in dashcam-only mode due to a fixed six-minute maximum timer on HCA steering. An unsupported
      # EPS flash update to work around this timer, and enable steering down to zero, is available from:
      #   https://github.com/pd0wm/pq-flasher
      # It is documented in a four-part blog series:
      #   https://blog.willemmelching.nl/carhacking/2022/01/02/vw-part1/
      # Panda ALLOW_DEBUG firmware required.
      ret.dashcamOnly = True

    else:
      # Set global MQB parameters
      ret.safetyConfigs = [get_safety_config(car.CarParams.SafetyModel.volkswagen)]
      ret.enableBsm = 0x30F in fingerprint[0]  # SWA_01

      if 0xAD in fingerprint[0] or docs:  # Getriebe_11
        ret.transmissionType = TransmissionType.automatic
      elif 0x187 in fingerprint[0]:  # EV_Gearshift
        ret.transmissionType = TransmissionType.direct
      else:
        ret.transmissionType = TransmissionType.manual

      if any(msg in fingerprint[1] for msg in (0x40, 0x86, 0xB2, 0xFD)):  # Airbag_01, LWI_01, ESP_19, ESP_21
        ret.networkLocation = NetworkLocation.gateway
      else:
        ret.networkLocation = NetworkLocation.fwdCamera

      if 0x126 in fingerprint[2]:  # HCA_01
        ret.flags |= VolkswagenFlags.STOCK_HCA_PRESENT.value

    # Global lateral tuning defaults, can be overridden per-vehicle

    ret.steerLimitTimer = 0.4
    if ret.flags & VolkswagenFlags.PQ:
      ret.steerActuatorDelay = 0.2
      CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)
    else:
      ret.steerActuatorDelay = 0.1
      ret.lateralTuning.pid.kpBP = [0.]
      ret.lateralTuning.pid.kiBP = [0.]
      ret.lateralTuning.pid.kf = 0.00006
      ret.lateralTuning.pid.kpV = [0.6]
      ret.lateralTuning.pid.kiV = [0.2]

    # Global longitudinal tuning defaults, can be overridden per-vehicle

    ret.experimentalLongitudinalAvailable = ret.networkLocation == NetworkLocation.gateway or docs
    if experimental_long:
      # Proof-of-concept, prep for E2E only. No radar points available. Panda ALLOW_DEBUG firmware required.
      ret.openpilotLongitudinalControl = True
      ret.safetyConfigs[0].safetyParam |= Panda.FLAG_VOLKSWAGEN_LONG_CONTROL
      if ret.transmissionType == TransmissionType.manual:
        ret.minEnableSpeed = 4.5

    ret.pcmCruise = not ret.openpilotLongitudinalControl
    ret.stoppingControl = True
    ret.stopAccel = -0.55
    ret.vEgoStarting = 0.1
    ret.vEgoStopping = 0.5
    ret.longitudinalTuning.kpV = [0.1]
    ret.longitudinalTuning.kiV = [0.0]
    ret.autoResumeSng = ret.minEnableSpeed == -1

    return ret

  # returns a car.CarState
  def _update(self, c):
    ret = self.CS.update(self.cp, self.cp_cam, self.cp_ext, self.CP.transmissionType)

    events = self.create_common_events(ret, extra_gears=[GearShifter.eco, GearShifter.sport, GearShifter.manumatic],
                                       pcm_enable=not self.CS.CP.openpilotLongitudinalControl,
                                       enable_buttons=(ButtonType.setCruise, ButtonType.resumeCruise))

    # Lock out if we weren't able to set model-specific specs
    if not (self.CP.flags & VolkswagenFlags.SPECS_SET):
      events.add(EventName.startupNoControl)

    # Low speed steer alert hysteresis logic
    if self.CP.minSteerSpeed > 0. and ret.vEgo < (self.CP.minSteerSpeed + 1.):
      self.low_speed_alert = True
    elif ret.vEgo > (self.CP.minSteerSpeed + 2.):
      self.low_speed_alert = False
    if self.low_speed_alert:
      events.add(EventName.belowSteerSpeed)

    if self.CS.CP.openpilotLongitudinalControl:
      if ret.vEgo < self.CP.minEnableSpeed + 0.5:
        events.add(EventName.belowEngageSpeed)
      if c.enabled and ret.vEgo < self.CP.minEnableSpeed:
        events.add(EventName.speedTooLow)

    if self.CC.eps_timer_soft_disable_alert:
      events.add(EventName.steerTimeLimit)

    ret.events = events.to_msg()

    return ret

