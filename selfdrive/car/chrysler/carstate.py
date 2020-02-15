from cereal import car
from opendbc.can.parser import CANParser
from selfdrive.car.chrysler.values import DBC, STEER_THRESHOLD
from common.kalman.simple_kalman import KF1D

GearShifter = car.CarState.GearShifter

def parse_gear_shifter(can_gear):
  return {0x1: GearShifter.park, 0x2: GearShifter.reverse, 0x3: GearShifter.neutral,
          0x4: GearShifter.drive, 0x5: GearShifter.low}.get(can_gear, GearShifter.unknown)


def get_can_parser(CP):

  signals = [
    # sig_name, sig_address, default
    ("PRNDL", "GEAR", 0),
    ("DOOR_OPEN_FL", "DOORS", 0),
    ("DOOR_OPEN_FR", "DOORS", 0),
    ("DOOR_OPEN_RL", "DOORS", 0),
    ("DOOR_OPEN_RR", "DOORS", 0),
    ("BRAKE_PRESSED_2", "BRAKE_2", 0),
    ("ACCEL_134", "ACCEL_GAS_134", 0),
    ("SPEED_LEFT", "SPEED_1", 0),
    ("SPEED_RIGHT", "SPEED_1", 0),
    ("WHEEL_SPEED_FL", "WHEEL_SPEEDS", 0),
    ("WHEEL_SPEED_RR", "WHEEL_SPEEDS", 0),
    ("WHEEL_SPEED_RL", "WHEEL_SPEEDS", 0),
    ("WHEEL_SPEED_FR", "WHEEL_SPEEDS", 0),
    ("STEER_ANGLE", "STEERING", 0),
    ("STEERING_RATE", "STEERING", 0),
    ("TURN_SIGNALS", "STEERING_LEVERS", 0),
    ("ACC_STATUS_2", "ACC_2", 0),
    ("HIGH_BEAM_FLASH", "STEERING_LEVERS", 0),
    ("ACC_SPEED_CONFIG_KPH", "DASHBOARD", 0),
    ("TORQUE_DRIVER", "EPS_STATUS", 0),
    ("TORQUE_MOTOR", "EPS_STATUS", 0),
    ("LKAS_STATE", "EPS_STATUS", 1),
    ("COUNTER", "EPS_STATUS", -1),
    ("TRACTION_OFF", "TRACTION_BUTTON", 0),
    ("SEATBELT_DRIVER_UNLATCHED", "SEATBELT_STATUS", 0),
    ("COUNTER", "WHEEL_BUTTONS", -1),  # incrementing counter for 23b
  ]

  # It's considered invalid if it is not received for 10x the expected period (1/f).
  checks = [
    # sig_address, frequency
    ("BRAKE_2", 50),
    ("EPS_STATUS", 100),
    ("SPEED_1", 100),
    ("WHEEL_SPEEDS", 50),
    ("STEERING", 100),
    ("ACC_2", 50),
  ]

  return CANParser(DBC[CP.carFingerprint]['pt'], signals, checks, 0)

def get_camera_parser(CP):
  signals = [
    # sig_name, sig_address, default
    # TODO read in all the other values
    ("COUNTER", "LKAS_COMMAND", -1),
    ("CAR_MODEL", "LKAS_HUD", -1),
    ("LKAS_STATUS_OK", "LKAS_HEARTBIT", -1)
  ]
  checks = []

  return CANParser(DBC[CP.carFingerprint]['pt'], signals, checks, 2)


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


  def update(self, cp, cp_cam):

    # update prevs, update must run once per loop
    self.prev_left_blinker_on = self.left_blinker_on
    self.prev_right_blinker_on = self.right_blinker_on

    self.frame_23b = int(cp.vl["WHEEL_BUTTONS"]['COUNTER'])
    self.frame = int(cp.vl["EPS_STATUS"]['COUNTER'])

    self.door_all_closed = not any([cp.vl["DOORS"]['DOOR_OPEN_FL'],
                                    cp.vl["DOORS"]['DOOR_OPEN_FR'],
                                    cp.vl["DOORS"]['DOOR_OPEN_RL'],
                                    cp.vl["DOORS"]['DOOR_OPEN_RR']])
    self.seatbelt = (cp.vl["SEATBELT_STATUS"]['SEATBELT_DRIVER_UNLATCHED'] == 0)

    self.brake_pressed = cp.vl["BRAKE_2"]['BRAKE_PRESSED_2'] == 5 # human-only
    self.pedal_gas = cp.vl["ACCEL_GAS_134"]['ACCEL_134']
    self.car_gas = self.pedal_gas
    self.esp_disabled = (cp.vl["TRACTION_BUTTON"]['TRACTION_OFF'] == 1)

    self.v_wheel_fl = cp.vl['WHEEL_SPEEDS']['WHEEL_SPEED_FL']
    self.v_wheel_rr = cp.vl['WHEEL_SPEEDS']['WHEEL_SPEED_RR']
    self.v_wheel_rl = cp.vl['WHEEL_SPEEDS']['WHEEL_SPEED_RL']
    self.v_wheel_fr = cp.vl['WHEEL_SPEEDS']['WHEEL_SPEED_FR']
    v_wheel = (cp.vl['SPEED_1']['SPEED_LEFT'] + cp.vl['SPEED_1']['SPEED_RIGHT']) / 2.

    # Kalman filter
    if abs(v_wheel - self.v_ego) > 2.0:  # Prevent large accelerations when car starts at non zero speed
      self.v_ego_kf.x = [[v_wheel], [0.0]]

    self.v_ego_raw = v_wheel
    v_ego_x = self.v_ego_kf.update(v_wheel)
    self.v_ego = float(v_ego_x[0])
    self.a_ego = float(v_ego_x[1])
    self.standstill = not v_wheel > 0.001

    self.angle_steers = cp.vl["STEERING"]['STEER_ANGLE']
    self.angle_steers_rate = cp.vl["STEERING"]['STEERING_RATE']
    self.gear_shifter = parse_gear_shifter(cp.vl['GEAR']['PRNDL'])
    self.main_on = cp.vl["ACC_2"]['ACC_STATUS_2'] == 7  # ACC is green.
    self.left_blinker_on = cp.vl["STEERING_LEVERS"]['TURN_SIGNALS'] == 1
    self.right_blinker_on = cp.vl["STEERING_LEVERS"]['TURN_SIGNALS'] == 2

    self.steer_torque_driver = cp.vl["EPS_STATUS"]["TORQUE_DRIVER"]
    self.steer_torque_motor = cp.vl["EPS_STATUS"]["TORQUE_MOTOR"]
    self.steer_override = abs(self.steer_torque_driver) > STEER_THRESHOLD
    steer_state = cp.vl["EPS_STATUS"]["LKAS_STATE"]
    self.steer_error = steer_state == 4 or (steer_state == 0 and self.v_ego > self.CP.minSteerSpeed)

    self.user_brake = 0
    self.brake_lights = self.brake_pressed
    self.v_cruise_pcm = cp.vl["DASHBOARD"]['ACC_SPEED_CONFIG_KPH']
    self.pcm_acc_status = self.main_on

    self.generic_toggle = bool(cp.vl["STEERING_LEVERS"]['HIGH_BEAM_FLASH'])

    self.lkas_counter = cp_cam.vl["LKAS_COMMAND"]['COUNTER']
    self.lkas_car_model = cp_cam.vl["LKAS_HUD"]['CAR_MODEL']
    self.lkas_status_ok = cp_cam.vl["LKAS_HEARTBIT"]['LKAS_STATUS_OK']
