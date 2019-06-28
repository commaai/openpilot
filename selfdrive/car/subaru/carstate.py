import copy
from common.kalman.simple_kalman import KF1D
from selfdrive.config import Conversions as CV
from selfdrive.can.parser import CANParser
from selfdrive.car.subaru.values import DBC, STEER_THRESHOLD

def get_powertrain_can_parser(CP):
  # this function generates lists for signal, messages and initial values
  signals = [
    # sig_name, sig_address, default
    ("Steer_Torque_Sensor", "Steering_Torque", 0),
    ("Steering_Angle", "Steering_Torque", 0),
    ("Cruise_On", "CruiseControl", 0),
    ("Cruise_Activated", "CruiseControl", 0),
    ("Brake_Pedal", "Brake_Pedal", 0),
    ("Throttle_Pedal", "Throttle", 0),
    ("LEFT_BLINKER", "Dashlights", 0),
    ("RIGHT_BLINKER", "Dashlights", 0),
    ("SEATBELT_FL", "Dashlights", 0),
    ("FL", "Wheel_Speeds", 0),
    ("FR", "Wheel_Speeds", 0),
    ("RL", "Wheel_Speeds", 0),
    ("RR", "Wheel_Speeds", 0),
    ("DOOR_OPEN_FR", "BodyInfo", 1),
    ("DOOR_OPEN_FL", "BodyInfo", 1),
    ("DOOR_OPEN_RR", "BodyInfo", 1),
    ("DOOR_OPEN_RL", "BodyInfo", 1),
    ("Units", "Dash_State", 1),
  ]

  checks = [
    # sig_address, frequency
    ("Dashlights", 10),
    ("CruiseControl", 20),
    ("Wheel_Speeds", 50),
    ("Steering_Torque", 50),
    ("BodyInfo", 10),
  ]

  return CANParser(DBC[CP.carFingerprint]['pt'], signals, checks, 0, timeout=100)


def get_camera_can_parser(CP):
  signals = [
    ("Cruise_Set_Speed", "ES_DashStatus", 0),

    ("Counter", "ES_Distance", 0),
    ("Signal1", "ES_Distance", 0),
    ("Signal2", "ES_Distance", 0),
    ("Main", "ES_Distance", 0),
    ("Signal3", "ES_Distance", 0),

    ("Checksum", "ES_LKAS_State", 0),
    ("Counter", "ES_LKAS_State", 0),
    ("Keep_Hands_On_Wheel", "ES_LKAS_State", 0),
    ("Empty_Box", "ES_LKAS_State", 0),
    ("Signal1", "ES_LKAS_State", 0),
    ("LKAS_ACTIVE", "ES_LKAS_State", 0),
    ("Signal2", "ES_LKAS_State", 0),
    ("Backward_Speed_Limit_Menu", "ES_LKAS_State", 0),
    ("LKAS_ENABLE_3", "ES_LKAS_State", 0),
    ("Signal3", "ES_LKAS_State", 0),
    ("LKAS_ENABLE_2", "ES_LKAS_State", 0),
    ("Signal4", "ES_LKAS_State", 0),
    ("LKAS_Left_Line_Visible", "ES_LKAS_State", 0),
    ("Signal6", "ES_LKAS_State", 0),
    ("LKAS_Right_Line_Visible", "ES_LKAS_State", 0),
    ("Signal7", "ES_LKAS_State", 0),
    ("FCW_Cont_Beep", "ES_LKAS_State", 0),
    ("FCW_Repeated_Beep", "ES_LKAS_State", 0),
    ("Throttle_Management_Activated", "ES_LKAS_State", 0),
    ("Traffic_light_Ahead", "ES_LKAS_State", 0),
    ("Right_Depart", "ES_LKAS_State", 0),
    ("Signal5", "ES_LKAS_State", 0),

  ]

  checks = [
    ("ES_DashStatus", 10),
  ]

  return CANParser(DBC[CP.carFingerprint]['pt'], signals, checks, 2, timeout=100)


class CarState(object):
  def __init__(self, CP):
    # initialize can parser
    self.CP = CP

    self.car_fingerprint = CP.carFingerprint
    self.left_blinker_on = False
    self.prev_left_blinker_on = False
    self.right_blinker_on = False
    self.prev_right_blinker_on = False
    self.steer_torque_driver = 0
    self.steer_not_allowed = False
    self.main_on = False

    # vEgo kalman filter
    dt = 0.01
    self.v_ego_kf = KF1D(x0=[[0.], [0.]],
                         A=[[1., dt], [0., 1.]],
                         C=[1., 0.],
                         K=[[0.12287673], [0.29666309]])
    self.v_ego = 0.

  def update(self, cp, cp_cam):

    self.pedal_gas = cp.vl["Throttle"]['Throttle_Pedal']
    self.brake_pressure = cp.vl["Brake_Pedal"]['Brake_Pedal']
    self.user_gas_pressed = self.pedal_gas > 0
    self.brake_pressed = self.brake_pressure > 0
    self.brake_lights = bool(self.brake_pressed)

    self.v_wheel_fl = cp.vl["Wheel_Speeds"]['FL'] * CV.KPH_TO_MS
    self.v_wheel_fr = cp.vl["Wheel_Speeds"]['FR'] * CV.KPH_TO_MS
    self.v_wheel_rl = cp.vl["Wheel_Speeds"]['RL'] * CV.KPH_TO_MS
    self.v_wheel_rr = cp.vl["Wheel_Speeds"]['RR'] * CV.KPH_TO_MS

    self.v_cruise_pcm = cp_cam.vl["ES_DashStatus"]['Cruise_Set_Speed']
    # 1 = imperial, 6 = metric
    if cp.vl["Dash_State"]['Units'] == 1:
      self.v_cruise_pcm *= CV.MPH_TO_KPH

    v_wheel = (self.v_wheel_fl + self.v_wheel_fr + self.v_wheel_rl + self.v_wheel_rr) / 4.
    # Kalman filter, even though Hyundai raw wheel speed is heaviliy filtered by default
    if abs(v_wheel - self.v_ego) > 2.0:  # Prevent large accelerations when car starts at non zero speed
      self.v_ego_kf.x = [[v_wheel], [0.0]]

    self.v_ego_raw = v_wheel
    v_ego_x = self.v_ego_kf.update(v_wheel)

    self.v_ego = float(v_ego_x[0])
    self.a_ego = float(v_ego_x[1])
    self.standstill = self.v_ego_raw < 0.01

    self.prev_left_blinker_on = self.left_blinker_on
    self.prev_right_blinker_on = self.right_blinker_on
    self.left_blinker_on = cp.vl["Dashlights"]['LEFT_BLINKER'] == 1
    self.right_blinker_on = cp.vl["Dashlights"]['RIGHT_BLINKER'] == 1
    self.seatbelt_unlatched = cp.vl["Dashlights"]['SEATBELT_FL'] == 1
    self.steer_torque_driver = cp.vl["Steering_Torque"]['Steer_Torque_Sensor']
    self.acc_active = cp.vl["CruiseControl"]['Cruise_Activated']
    self.main_on = cp.vl["CruiseControl"]['Cruise_On']
    self.steer_override = abs(self.steer_torque_driver) > STEER_THRESHOLD[self.car_fingerprint]
    self.angle_steers = cp.vl["Steering_Torque"]['Steering_Angle']
    self.door_open = any([cp.vl["BodyInfo"]['DOOR_OPEN_RR'],
      cp.vl["BodyInfo"]['DOOR_OPEN_RL'],
      cp.vl["BodyInfo"]['DOOR_OPEN_FR'],
      cp.vl["BodyInfo"]['DOOR_OPEN_FL']])

    self.es_distance_msg = copy.copy(cp_cam.vl["ES_Distance"])
    self.es_lkas_msg = copy.copy(cp_cam.vl["ES_LKAS_State"])
