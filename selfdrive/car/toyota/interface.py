from cereal import car
from openpilot.common.conversions import Conversions as CV
from panda import Panda
from panda.python import uds
from openpilot.selfdrive.car.toyota.values import Ecu, CAR, DBC, ToyotaFlags, CarControllerParams, TSS2_CAR, RADAR_ACC_CAR, NO_DSU_CAR, \
                                        MIN_ACC_SPEED, EPS_SCALE, EV_HYBRID_CAR, UNSUPPORTED_DSU_CAR, NO_STOP_TIMER_CAR, ANGLE_CONTROL_CAR
from openpilot.selfdrive.car import get_safety_config
from openpilot.selfdrive.car.disable_ecu import disable_ecu
from openpilot.selfdrive.car.interfaces import CarInterfaceBase

EventName = car.CarEvent.EventName
SteerControlType = car.CarParams.SteerControlType


class CarInterface(CarInterfaceBase):
  @staticmethod
  def get_pid_accel_limits(CP, current_speed, cruise_speed):
    return CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX

  @staticmethod
  def _get_params(ret, candidate, fingerprint, car_fw, experimental_long, docs):
    ret.carName = "toyota"
    ret.safetyConfigs = [get_safety_config(car.CarParams.SafetyModel.toyota)]
    ret.safetyConfigs[0].safetyParam = EPS_SCALE[candidate]

    # BRAKE_MODULE is on a different address for these cars
    if DBC[candidate]["pt"] == "toyota_new_mc_pt_generated":
      ret.safetyConfigs[0].safetyParam |= Panda.FLAG_TOYOTA_ALT_BRAKE

    # Allow angle control cars with whitelisted EPSs to use torque control (made in Japan)
    # So far only hybrid RAV4 2023 has been seen with this FW version
    angle_car_torque_fw = any(fw.ecu == "eps" and fw.fwVersion == b'8965B42371\x00\x00\x00\x00\x00\x00' for fw in car_fw)
    if candidate not in ANGLE_CONTROL_CAR or (angle_car_torque_fw and candidate == CAR.RAV4H_TSS2_2023):
      CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)

      ret.steerActuatorDelay = 0.12  # Default delay, Prius has larger delay
      ret.steerLimitTimer = 0.4
    else:
      ret.dashcamOnly = True
      ret.steerControlType = SteerControlType.angle
      ret.safetyConfigs[0].safetyParam |= Panda.FLAG_TOYOTA_LTA

      # LTA control can be more delayed and winds up more often
      ret.steerActuatorDelay = 0.25
      ret.steerLimitTimer = 0.8

    ret.stoppingControl = False  # Toyota starts braking more when it thinks you want to stop

    stop_and_go = candidate in TSS2_CAR

    if candidate == CAR.PRIUS:
      stop_and_go = True
      ret.wheelbase = 2.70
      ret.steerRatio = 15.74   # unknown end-to-end spec
      ret.tireStiffnessFactor = 0.6371   # hand-tune
      ret.mass = 3045. * CV.LB_TO_KG
      # Only give steer angle deadzone to for bad angle sensor prius
      for fw in car_fw:
        if fw.ecu == "eps" and not fw.fwVersion == b'8965B47060\x00\x00\x00\x00\x00\x00':
          ret.steerActuatorDelay = 0.25
          CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning, steering_angle_deadzone_deg=0.2)

    elif candidate == CAR.PRIUS_V:
      stop_and_go = True
      ret.wheelbase = 2.78
      ret.steerRatio = 17.4
      ret.tireStiffnessFactor = 0.5533
      ret.mass = 3340. * CV.LB_TO_KG

    elif candidate in (CAR.RAV4, CAR.RAV4H):
      stop_and_go = True if (candidate in CAR.RAV4H) else False
      ret.wheelbase = 2.65
      ret.steerRatio = 16.88   # 14.5 is spec end-to-end
      ret.tireStiffnessFactor = 0.5533
      ret.mass = 3650. * CV.LB_TO_KG  # mean between normal and hybrid

    elif candidate == CAR.COROLLA:
      ret.wheelbase = 2.70
      ret.steerRatio = 18.27
      ret.tireStiffnessFactor = 0.444  # not optimized yet
      ret.mass = 2860. * CV.LB_TO_KG  # mean between normal and hybrid

    elif candidate in (CAR.LEXUS_RX, CAR.LEXUS_RXH, CAR.LEXUS_RX_TSS2, CAR.LEXUS_RXH_TSS2):
      stop_and_go = True
      ret.wheelbase = 2.79
      ret.steerRatio = 16.  # 14.8 is spec end-to-end
      ret.wheelSpeedFactor = 1.035
      ret.tireStiffnessFactor = 0.5533
      ret.mass = 4481. * CV.LB_TO_KG  # mean between min and max

    elif candidate in (CAR.CHR, CAR.CHRH, CAR.CHR_TSS2, CAR.CHRH_TSS2):
      stop_and_go = True
      ret.wheelbase = 2.63906
      ret.steerRatio = 13.6
      ret.tireStiffnessFactor = 0.7933
      ret.mass = 3300. * CV.LB_TO_KG

    elif candidate in (CAR.CAMRY, CAR.CAMRYH, CAR.CAMRY_TSS2, CAR.CAMRYH_TSS2):
      stop_and_go = True
      ret.wheelbase = 2.82448
      ret.steerRatio = 13.7
      ret.tireStiffnessFactor = 0.7933
      ret.mass = 3400. * CV.LB_TO_KG  # mean between normal and hybrid

    elif candidate in (CAR.HIGHLANDER, CAR.HIGHLANDERH, CAR.HIGHLANDER_TSS2, CAR.HIGHLANDERH_TSS2):
      stop_and_go = True
      ret.wheelbase = 2.8194  # average of 109.8 and 112.2 in
      ret.steerRatio = 16.0
      ret.tireStiffnessFactor = 0.8
      ret.mass = 4516. * CV.LB_TO_KG  # mean between normal and hybrid

    elif candidate in (CAR.AVALON, CAR.AVALON_2019, CAR.AVALONH_2019, CAR.AVALON_TSS2, CAR.AVALONH_TSS2):
      # starting from 2019, all Avalon variants have stop and go
      # https://engage.toyota.com/static/images/toyota_safety_sense/TSS_Applicability_Chart.pdf
      stop_and_go = candidate != CAR.AVALON
      ret.wheelbase = 2.82
      ret.steerRatio = 14.8  # Found at https://pressroom.toyota.com/releases/2016+avalon+product+specs.download
      ret.tireStiffnessFactor = 0.7983
      ret.mass = 3505. * CV.LB_TO_KG  # mean between normal and hybrid

    elif candidate in (CAR.RAV4_TSS2, CAR.RAV4_TSS2_2022, CAR.RAV4H_TSS2, CAR.RAV4H_TSS2_2022,
                       CAR.RAV4_TSS2_2023, CAR.RAV4H_TSS2_2023):
      ret.wheelbase = 2.68986
      ret.steerRatio = 14.3
      ret.tireStiffnessFactor = 0.7933
      ret.mass = 3585. * CV.LB_TO_KG  # Average between ICE and Hybrid

      # Only specific EPS FW accept torque on 2023 RAV4, so they likely are all the same
      # TODO: revisit this disparity if there is a divide for 2023
      if candidate not in (CAR.RAV4_TSS2_2023, CAR.RAV4H_TSS2_2023):
        ret.lateralTuning.init('pid')
        ret.lateralTuning.pid.kiBP = [0.0]
        ret.lateralTuning.pid.kpBP = [0.0]
        ret.lateralTuning.pid.kpV = [0.6]
        ret.lateralTuning.pid.kiV = [0.1]
        ret.lateralTuning.pid.kf = 0.00007818594

        # 2019+ RAV4 TSS2 uses two different steering racks and specific tuning seems to be necessary.
        # See https://github.com/commaai/openpilot/pull/21429#issuecomment-873652891
        for fw in car_fw:
          if fw.ecu == "eps" and (fw.fwVersion.startswith(b'\x02') or fw.fwVersion in [b'8965B42181\x00\x00\x00\x00\x00\x00']):
            ret.lateralTuning.pid.kpV = [0.15]
            ret.lateralTuning.pid.kiV = [0.05]
            ret.lateralTuning.pid.kf = 0.00004
            break

    elif candidate in (CAR.COROLLA_TSS2, CAR.COROLLAH_TSS2):
      ret.wheelbase = 2.67  # Average between 2.70 for sedan and 2.64 for hatchback
      ret.steerRatio = 13.9
      ret.tireStiffnessFactor = 0.444  # not optimized yet
      ret.mass = 3060. * CV.LB_TO_KG

    elif candidate in (CAR.LEXUS_ES, CAR.LEXUS_ESH, CAR.LEXUS_ES_TSS2, CAR.LEXUS_ESH_TSS2):
      if candidate not in (CAR.LEXUS_ES,):  # TODO: LEXUS_ES may have sng
        stop_and_go = True
      ret.wheelbase = 2.8702
      ret.steerRatio = 16.0  # not optimized
      ret.tireStiffnessFactor = 0.444  # not optimized yet
      ret.mass = 3677. * CV.LB_TO_KG  # mean between min and max

    elif candidate == CAR.SIENNA:
      stop_and_go = True
      ret.wheelbase = 3.03
      ret.steerRatio = 15.5
      ret.tireStiffnessFactor = 0.444
      ret.mass = 4590. * CV.LB_TO_KG

    elif candidate in (CAR.LEXUS_IS, CAR.LEXUS_IS_TSS2, CAR.LEXUS_RC):
      ret.wheelbase = 2.79908
      ret.steerRatio = 13.3
      ret.tireStiffnessFactor = 0.444
      ret.mass = 3736.8 * CV.LB_TO_KG

    elif candidate == CAR.LEXUS_CTH:
      stop_and_go = True
      ret.wheelbase = 2.60
      ret.steerRatio = 18.6
      ret.tireStiffnessFactor = 0.517
      ret.mass = 3108 * CV.LB_TO_KG  # mean between min and max

    elif candidate in (CAR.LEXUS_NX, CAR.LEXUS_NXH, CAR.LEXUS_NX_TSS2, CAR.LEXUS_NXH_TSS2):
      stop_and_go = True
      ret.wheelbase = 2.66
      ret.steerRatio = 14.7
      ret.tireStiffnessFactor = 0.444  # not optimized yet
      ret.mass = 4070 * CV.LB_TO_KG

    elif candidate == CAR.PRIUS_TSS2:
      ret.wheelbase = 2.70002  # from toyota online sepc.
      ret.steerRatio = 13.4   # True steerRatio from older prius
      ret.tireStiffnessFactor = 0.6371   # hand-tune
      ret.mass = 3115. * CV.LB_TO_KG

    elif candidate == CAR.MIRAI:
      stop_and_go = True
      ret.wheelbase = 2.91
      ret.steerRatio = 14.8
      ret.tireStiffnessFactor = 0.8
      ret.mass = 4300. * CV.LB_TO_KG

    elif candidate in (CAR.ALPHARD_TSS2, CAR.ALPHARDH_TSS2):
      ret.wheelbase = 3.00
      ret.steerRatio = 14.2
      ret.tireStiffnessFactor = 0.444
      ret.mass = 4305. * CV.LB_TO_KG

    ret.centerToFront = ret.wheelbase * 0.44

    # TODO: Some TSS-P platforms have BSM, but are flipped based on region or driving direction.
    # Detect flipped signals and enable for C-HR and others
    ret.enableBsm = 0x3F6 in fingerprint[0] and candidate in TSS2_CAR

    # Detect smartDSU, which intercepts ACC_CMD from the DSU (or radar) allowing openpilot to send it
    # 0x2AA is sent by a similar device which intercepts the radar instead of DSU on NO_DSU_CARs
    if 0x2FF in fingerprint[0] or (0x2AA in fingerprint[0] and candidate in NO_DSU_CAR):
      ret.flags |= ToyotaFlags.SMART_DSU.value

    # No radar dbc for cars without DSU which are not TSS 2.0
    # TODO: make an adas dbc file for dsu-less models
    ret.radarUnavailable = DBC[candidate]['radar'] is None or candidate in (NO_DSU_CAR - TSS2_CAR)

    # In TSS2 cars, the camera does long control
    found_ecus = [fw.ecu for fw in car_fw]
    ret.enableDsu = len(found_ecus) > 0 and Ecu.dsu not in found_ecus and candidate not in (NO_DSU_CAR | UNSUPPORTED_DSU_CAR) \
                                        and not (ret.flags & ToyotaFlags.SMART_DSU)
    ret.enableGasInterceptor = 0x201 in fingerprint[0]

    # if the smartDSU is detected, openpilot can send ACC_CONTROL and the smartDSU will block it from the DSU or radar.
    # since we don't yet parse radar on TSS2/TSS-P radar-based ACC cars, gate longitudinal behind experimental toggle
    use_sdsu = bool(ret.flags & ToyotaFlags.SMART_DSU)
    if candidate in (RADAR_ACC_CAR | NO_DSU_CAR):
      ret.experimentalLongitudinalAvailable = use_sdsu

      if not use_sdsu:
        # Disabling radar is only supported on TSS2 radar-ACC cars
        if experimental_long and candidate in RADAR_ACC_CAR and False:  # TODO: disabling radar isn't supported yet
          ret.flags |= ToyotaFlags.DISABLE_RADAR.value
      else:
        use_sdsu = use_sdsu and experimental_long

    # openpilot longitudinal enabled by default:
    #  - non-(TSS2 radar ACC cars) w/ smartDSU installed
    #  - cars w/ DSU disconnected
    #  - TSS2 cars with camera sending ACC_CONTROL where we can block it
    # openpilot longitudinal behind experimental long toggle:
    #  - TSS2 radar ACC cars w/ smartDSU installed
    #  - TSS2 radar ACC cars w/o smartDSU installed (disables radar)
    #  - TSS-P DSU-less cars w/ CAN filter installed (no radar parser yet)
    ret.openpilotLongitudinalControl = use_sdsu or ret.enableDsu or candidate in (TSS2_CAR - RADAR_ACC_CAR) or bool(ret.flags & ToyotaFlags.DISABLE_RADAR.value)
    ret.autoResumeSng = ret.openpilotLongitudinalControl and candidate in NO_STOP_TIMER_CAR

    if not ret.openpilotLongitudinalControl:
      ret.safetyConfigs[0].safetyParam |= Panda.FLAG_TOYOTA_STOCK_LONGITUDINAL

    # we can't use the fingerprint to detect this reliably, since
    # the EV gas pedal signal can take a couple seconds to appear
    if candidate in EV_HYBRID_CAR:
      ret.flags |= ToyotaFlags.HYBRID.value

    # min speed to enable ACC. if car can do stop and go, then set enabling speed
    # to a negative value, so it won't matter.
    ret.minEnableSpeed = -1. if (stop_and_go or ret.enableGasInterceptor) else MIN_ACC_SPEED

    tune = ret.longitudinalTuning
    tune.deadzoneBP = [0., 9.]
    tune.deadzoneV = [.0, .15]
    if candidate in TSS2_CAR or ret.enableGasInterceptor:
      tune.kpBP = [0., 5., 20.]
      tune.kpV = [1.3, 1.0, 0.7]
      tune.kiBP = [0., 5., 12., 20., 27.]
      tune.kiV = [.35, .23, .20, .17, .1]
      if candidate in TSS2_CAR:
        ret.vEgoStopping = 0.25
        ret.vEgoStarting = 0.25
        ret.stoppingDecelRate = 0.3  # reach stopping target smoothly
    else:
      tune.kpBP = [0., 5., 35.]
      tune.kiBP = [0., 35.]
      tune.kpV = [3.6, 2.4, 1.5]
      tune.kiV = [0.54, 0.36]

    return ret

  @staticmethod
  def init(CP, logcan, sendcan):
    # disable radar if alpha longitudinal toggled on radar-ACC car without CAN filter/smartDSU
    if CP.flags & ToyotaFlags.DISABLE_RADAR.value:
      communication_control = bytes([uds.SERVICE_TYPE.COMMUNICATION_CONTROL, uds.CONTROL_TYPE.ENABLE_RX_DISABLE_TX, uds.MESSAGE_TYPE.NORMAL])
      disable_ecu(logcan, sendcan, bus=0, addr=0x750, sub_addr=0xf, com_cont_req=communication_control)

  # returns a car.CarState
  def _update(self, c):
    ret = self.CS.update(self.cp, self.cp_cam)

    # events
    events = self.create_common_events(ret)

    # Lane Tracing Assist control is unavailable (EPS_STATUS->LTA_STATE=0) until
    # the more accurate angle sensor signal is initialized
    if self.CP.steerControlType == SteerControlType.angle and not self.CS.accurate_steer_angle_seen:
      events.add(EventName.vehicleSensorsInvalid)

    if self.CP.openpilotLongitudinalControl:
      if ret.cruiseState.standstill and not ret.brakePressed and not self.CP.enableGasInterceptor:
        events.add(EventName.resumeRequired)
      if self.CS.low_speed_lockout:
        events.add(EventName.lowSpeedLockout)
      if ret.vEgo < self.CP.minEnableSpeed:
        events.add(EventName.belowEngageSpeed)
        if c.actuators.accel > 0.3:
          # some margin on the actuator to not false trigger cancellation while stopping
          events.add(EventName.speedTooLow)
        if ret.vEgo < 0.001:
          # while in standstill, send a user alert
          events.add(EventName.manualRestart)

    ret.events = events.to_msg()

    return ret

  # pass in a car.CarControl
  # to be called @ 100hz
  def apply(self, c, now_nanos):
    return self.CC.update(c, self.CS, now_nanos)
