import os
import time
from cereal import car
from common.kalman.simple_kalman import KF1D
from common.realtime import DT_CTRL
from selfdrive.car import gen_empty_fingerprint
from selfdrive.config import Conversions as CV
from selfdrive.controls.lib.drive_helpers import EventTypes as ET, create_event
from selfdrive.controls.lib.vehicle_model import VehicleModel

GearShifter = car.CarState.GearShifter

# generic car and radar interfaces

class CarInterfaceBase():
  def __init__(self, CP, CarController, CarState):
    self.CP = CP
    self.VM = VehicleModel(CP)

    self.frame = 0
    self.low_speed_alert = False

    self.CS = CarState(CP)
    self.cp = self.CS.get_can_parser(CP)
    self.cp_cam = self.CS.get_cam_can_parser(CP)

    self.CC = None
    if CarController is not None:
      self.CC = CarController(self.cp.dbc_name, CP, self.VM)

  @staticmethod
  def calc_accel_override(a_ego, a_target, v_ego, v_target):
    return 1.

  @staticmethod
  def compute_gb(accel, speed):
    raise NotImplementedError

  @staticmethod
  def get_params(candidate, fingerprint=gen_empty_fingerprint(), has_relay=False, car_fw=[]):
    raise NotImplementedError

  # returns a set of default params to avoid repetition in car specific params
  @staticmethod
  def get_std_params(candidate, fingerprint, has_relay):
    ret = car.CarParams.new_message()
    ret.carFingerprint = candidate
    ret.isPandaBlack = has_relay

    # standard ALC params
    ret.steerControlType = car.CarParams.SteerControlType.torque
    ret.steerMaxBP = [0.]
    ret.steerMaxV = [1.]
    ret.minSteerSpeed = 0.

    # stock ACC by default
    ret.enableCruise = True
    ret.minEnableSpeed = -1.  # enable is done by stock ACC, so ignore this
    ret.steerRatioRear = 0.  # no rear steering, at least on the listed cars aboveA
    ret.gasMaxBP = [0.]
    ret.gasMaxV = [.5]  # half max brake
    ret.brakeMaxBP = [0.]
    ret.brakeMaxV = [1.]
    ret.openpilotLongitudinalControl = False
    ret.startAccel = 0.0
    ret.stoppingControl = False
    ret.longitudinalTuning.deadzoneBP = [0.]
    ret.longitudinalTuning.deadzoneV = [0.]
    ret.longitudinalTuning.kpBP = [0.]
    ret.longitudinalTuning.kpV = [1.]
    ret.longitudinalTuning.kiBP = [0.]
    ret.longitudinalTuning.kiV = [1.]
    return ret

  # returns a car.CarState, pass in car.CarControl
  def update(self, c, can_strings):
    raise NotImplementedError

  # return sendcan, pass in a car.CarControl
  def apply(self, c):
    raise NotImplementedError

  def create_common_events(self, cs_out, extra_gears=[], gas_resume_speed=-1, pcm_enable=True):
    events = []

    if cs_out.doorOpen:
      events.append(create_event('doorOpen', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
    if cs_out.seatbeltUnlatched:
      events.append(create_event('seatbeltNotLatched', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
    if cs_out.gearShifter != GearShifter.drive and cs_out.gearShifter not in extra_gears:
      events.append(create_event('wrongGear', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
    if cs_out.gearShifter == GearShifter.reverse:
      events.append(create_event('reverseGear', [ET.NO_ENTRY, ET.IMMEDIATE_DISABLE]))
    if not cs_out.cruiseState.available:
      events.append(create_event('wrongCarMode', [ET.NO_ENTRY, ET.USER_DISABLE]))
    if cs_out.espDisabled:
      events.append(create_event('espDisabled', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
    if cs_out.gasPressed:
      events.append(create_event('pedalPressed', [ET.PRE_ENABLE]))
    if cs_out.stockAeb:
      events.append(create_event('stockAeb', []))
    if cs_out.vEgo > 92 * CV.MPH_TO_MS:
      events.append(create_event('speedTooHigh', [ET.NO_ENTRY, ET.SOFT_DISABLE]))

    if cs_out.steerError:
      events.append(create_event('steerUnavailable', [ET.NO_ENTRY, ET.IMMEDIATE_DISABLE, ET.PERMANENT]))
    elif cs_out.steerWarning:
      events.append(create_event('steerTempUnavailable', [ET.NO_ENTRY, ET.WARNING]))

    # Disable on rising edge of gas or brake. Also disable on brake when speed > 0.
    # Optionally allow to press gas at zero speed to resume.
    # e.g. Chrysler does not spam the resume button yet, so resuming with gas is handy. FIXME!
    if (cs_out.gasPressed and (not self.CS.out.gasPressed) and cs_out.vEgo > gas_resume_speed) or \
       (cs_out.brakePressed and (not self.CS.out.brakePressed or not cs_out.standstill)):
      events.append(create_event('pedalPressed', [ET.NO_ENTRY, ET.USER_DISABLE]))

    # we engage when pcm is active (rising edge)
    if pcm_enable:
      if cs_out.cruiseState.enabled and not self.CS.out.cruiseState.enabled:
        events.append(create_event('pcmEnable', [ET.ENABLE]))
      elif not cs_out.cruiseState.enabled:
        events.append(create_event('pcmDisable', [ET.USER_DISABLE]))

    return events

class RadarInterfaceBase():
  def __init__(self, CP):
    self.pts = {}
    self.delay = 0
    self.radar_ts = CP.radarTimeStep

  def update(self, can_strings):
    ret = car.RadarData.new_message()

    if 'NO_RADAR_SLEEP' not in os.environ:
      time.sleep(self.radar_ts)  # radard runs on RI updates

    return ret

class CarStateBase:
  def __init__(self, CP):
    self.CP = CP
    self.car_fingerprint = CP.carFingerprint
    self.cruise_buttons = 0
    self.out = car.CarState.new_message()

    # Q = np.matrix([[10.0, 0.0], [0.0, 100.0]])
    # R = 1e3
    self.v_ego_kf = KF1D(x0=[[0.0], [0.0]],
                         A=[[1.0, DT_CTRL], [0.0, 1.0]],
                         C=[1.0, 0.0],
                         K=[[0.12287673], [0.29666309]])

  def update_speed_kf(self, v_ego_raw):
    if abs(v_ego_raw - self.v_ego_kf.x[0][0]) > 2.0:  # Prevent large accelerations when car starts at non zero speed
      self.v_ego_kf.x = [[v_ego_raw], [0.0]]

    v_ego_x = self.v_ego_kf.update(v_ego_raw)
    return float(v_ego_x[0]), float(v_ego_x[1])

  @staticmethod
  def parse_gear_shifter(gear):
    return {'P': GearShifter.park, 'R': GearShifter.reverse, 'N': GearShifter.neutral,
            'E': GearShifter.eco, 'T': GearShifter.manumatic, 'D': GearShifter.drive,
            'S': GearShifter.sport, 'L': GearShifter.low, 'B': GearShifter.brake}.get(gear, GearShifter.unknown)

  @staticmethod
  def get_cam_can_parser(CP):
    return None
