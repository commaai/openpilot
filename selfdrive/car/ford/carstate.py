from opendbc.can.parser import CANParser
from common.numpy_fast import mean
from selfdrive.config import Conversions as CV
from selfdrive.car.ford.values import DBC
from common.kalman.simple_kalman import KF1D

WHEEL_RADIUS = 0.33

def get_can_parser(CP):

  signals = [
    # sig_name, sig_address, default
    ("WhlRr_W_Meas", "WheelSpeed_CG1", 0.),
    ("WhlRl_W_Meas", "WheelSpeed_CG1", 0.),
    ("WhlFr_W_Meas", "WheelSpeed_CG1", 0.),
    ("WhlFl_W_Meas", "WheelSpeed_CG1", 0.),
    ("SteWhlRelInit_An_Sns", "Steering_Wheel_Data_CG1", 0.),
    ("Cruise_State", "Cruise_Status", 0.),
    ("Set_Speed", "Cruise_Status", 0.),
    ("LaActAvail_D_Actl", "Lane_Keep_Assist_Status", 0),
    ("LaHandsOff_B_Actl", "Lane_Keep_Assist_Status", 0),
    ("LaActDeny_B_Actl", "Lane_Keep_Assist_Status", 0),
    ("ApedPosScal_Pc_Actl", "EngineData_14", 0.),
    ("Dist_Incr", "Steering_Buttons", 0.),
    ("Brake_Drv_Appl", "Cruise_Status", 0.),
    ("Brake_Lights", "BCM_to_HS_Body", 0.),
  ]

  checks = [
  ]

  return CANParser(DBC[CP.carFingerprint]['pt'], signals, checks, 0)


class CarState():
  def __init__(self, CP):

    self.CP = CP
    self.left_blinker_on = 0
    self.right_blinker_on = 0

    # initialize can parser
    self.car_fingerprint = CP.carFingerprint

    # vEgo kalman filter
    dt = 0.01
    # Q = np.matrix([[10.0, 0.0], [0.0, 100.0]])
    # R = 1e3
    self.v_ego_kf = KF1D(x0=[[0.0], [0.0]],
                         A=[[1.0, dt], [0.0, 1.0]],
                         C=[1.0, 0.0],
                         K=[[0.12287673], [0.29666309]])
    self.v_ego = 0.0

  def update(self, cp):
    # update prevs, update must run once per loop
    self.prev_left_blinker_on = self.left_blinker_on
    self.prev_right_blinker_on = self.right_blinker_on

    # calc best v_ego estimate, by averaging two opposite corners
    self.v_wheel_fl = cp.vl["WheelSpeed_CG1"]['WhlRr_W_Meas'] * WHEEL_RADIUS
    self.v_wheel_fr = cp.vl["WheelSpeed_CG1"]['WhlRl_W_Meas'] * WHEEL_RADIUS
    self.v_wheel_rl = cp.vl["WheelSpeed_CG1"]['WhlFr_W_Meas'] * WHEEL_RADIUS
    self.v_wheel_rr = cp.vl["WheelSpeed_CG1"]['WhlFl_W_Meas'] * WHEEL_RADIUS
    v_wheel = mean([self.v_wheel_fl, self.v_wheel_fr, self.v_wheel_rl, self.v_wheel_rr])

    # Kalman filter
    if abs(v_wheel - self.v_ego) > 2.0:  # Prevent large accelerations when car starts at non zero speed
      self.v_ego_kf.x = [[v_wheel], [0.0]]

    self.v_ego_raw = v_wheel
    v_ego_x = self.v_ego_kf.update(v_wheel)
    self.v_ego = float(v_ego_x[0])
    self.a_ego = float(v_ego_x[1])
    self.standstill = not v_wheel > 0.001

    self.angle_steers = cp.vl["Steering_Wheel_Data_CG1"]['SteWhlRelInit_An_Sns']
    self.v_cruise_pcm = cp.vl["Cruise_Status"]['Set_Speed'] * CV.MPH_TO_MS
    self.pcm_acc_status = cp.vl["Cruise_Status"]['Cruise_State']
    self.main_on = cp.vl["Cruise_Status"]['Cruise_State'] != 0
    self.lkas_state = cp.vl["Lane_Keep_Assist_Status"]['LaActAvail_D_Actl']
    # TODO: we also need raw driver torque, needed for Assisted Lane Change
    self.steer_override = not cp.vl["Lane_Keep_Assist_Status"]['LaHandsOff_B_Actl']
    self.steer_error = cp.vl["Lane_Keep_Assist_Status"]['LaActDeny_B_Actl']
    self.user_gas = cp.vl["EngineData_14"]['ApedPosScal_Pc_Actl']
    self.brake_pressed = bool(cp.vl["Cruise_Status"]["Brake_Drv_Appl"])
    self.brake_lights = bool(cp.vl["BCM_to_HS_Body"]["Brake_Lights"])
    self.generic_toggle = bool(cp.vl["Steering_Buttons"]["Dist_Incr"])
