import os
import time
from cereal import car
from common.kalman.simple_kalman import KF1D
from common.realtime import DT_CTRL
from selfdrive.car import gen_empty_fingerprint
from selfdrive.controls.lib.drive_helpers import EventTypes as ET, create_event

GearShifter = car.CarState.GearShifter

# generic car and radar interfaces

class CarInterfaceBase():
  def __init__(self, CP, CarController):
    pass

  @staticmethod
  def calc_accel_override(a_ego, a_target, v_ego, v_target):
    return 1.

  @staticmethod
  def compute_gb(accel, speed):
    raise NotImplementedError

  @staticmethod
  def get_params(candidate, fingerprint=gen_empty_fingerprint(), has_relay=False, car_fw=[]):
    raise NotImplementedError

  # returns a car.CarState, pass in car.CarControl
  def update(self, c, can_strings):
    raise NotImplementedError

  # return sendcan, pass in a car.CarControl
  def apply(self, c):
    raise NotImplementedError

  def create_common_events(self, c, cs_out):
    events = []

    if self.CP.openpilotLongitudinalControl:
      if not cs_out.gearShifter == GearShifter.drive:
        events.append(create_event('wrongGear', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
      if self.CS.esp_disabled:
        events.append(create_event('espDisabled', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
      if not cs_out.cruiseState.available:
        events.append(create_event('wrongCarMode', [ET.NO_ENTRY, ET.USER_DISABLE]))
      if cs_out.gearShifter == GearShifter.reverse:
        events.append(create_event('reverseGear', [ET.NO_ENTRY, ET.IMMEDIATE_DISABLE]))

      # TODO: is this one toyota specific?
      if cs_out.vEgo < self.CP.minEnableSpeed:
        events.append(create_event('speedTooLow', [ET.NO_ENTRY]))
        if c.actuators.gas > 0.1:
          # some margin on the actuator to not false trigger cancellation while stopping
          events.append(create_event('speedTooLow', [ET.IMMEDIATE_DISABLE]))
        if cs_out.vEgo < 0.001:
          # while in standstill, send a user alert
          events.append(create_event('manualRestart', [ET.WARNING]))

    # TODO: add these to carstate struct?
    try:
      if self.CS.steer_error:
        events.append(create_event('steerUnavailable', [ET.NO_ENTRY, ET.IMMEDIATE_DISABLE, ET.PERMANENT]))

      if self.CS.brake_error:
        events.append(create_event('brakeUnavailable', [ET.NO_ENTRY, ET.IMMEDIATE_DISABLE, ET.PERMANENT]))
    except:
      pass

    if cs_out.doorOpen:
      events.append(create_event('doorOpen', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
    if cs_out.seatbeltUnlatched:
      events.append(create_event('seatbeltNotLatched', [ET.NO_ENTRY, ET.SOFT_DISABLE]))

    # TODO: some are temp unavailable and some are just unavailable
    #if self.CS.steer_error:
    #  events.append(create_event('steerTempUnavailable', [ET.NO_ENTRY, ET.WARNING]))

    # TODO: add these to carstate struct?
    try:
      if self.CS.park_brake:
        events.append(create_event('parkBrake', [ET.NO_ENTRY, ET.USER_DISABLE]))
    except:
      pass

    # disable on pedals rising edge or when brake is pressed and speed isn't zero
    if (cs_out.gasPressed and not self.gas_pressed_prev) or \
       (cs_out.brakePressed and (not self.brake_pressed_prev or cs_out.vEgo > 0.001)):
      events.append(create_event('pedalPressed', [ET.NO_ENTRY, ET.USER_DISABLE]))

    if cs_out.gasPressed:
      events.append(create_event('pedalPressed', [ET.PRE_ENABLE]))

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
