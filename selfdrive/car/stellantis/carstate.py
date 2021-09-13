from cereal import car
from opendbc.can.parser import CANParser
from opendbc.can.can_define import CANDefine
from selfdrive.config import Conversions as CV
from selfdrive.car.interfaces import CarStateBase
from selfdrive.car.ram.values import DBC, STEER_THRESHOLD

GearShifter = car.CarState.GearShifter


class CarState(CarStateBase):

  def update(self, cp, cp_cam):

    ret = car.CarState.new_message()

    self.steer_command_bit = cp_cam.vl["FORWARD_CAMERA_LKAS"]['LKAS_CONTROL_BIT']

    #    ret.autoHighBeamBit = cp_cam.vl["FORWARD_CAMERA_HUD"]['AUTO_HIGH_BEAM_BIT']  # HUD needs work, going with stock, but we can read this bit for fun

    ret.doorOpen = any([cp.vl["DOORS"]['DOOR_OPEN_LF'],
                        cp.vl["DOORS"]['DOOR_OPEN_RF'],
                        cp.vl["DOORS"]['DOOR_OPEN_LR'],
                        cp.vl["DOORS"]['DOOR_OPEN_RR']])
    ret.seatbeltUnlatched = cp.vl["OCCUPANT_RESTRAINT_MODULE"]['DRIVER_SEATBELT_STATUS'] == 1

    ret.brakePressed = cp.vl["ABS_2"][
                         'DRIVER_BRAKE'] > 5  # human-only... TODO: find values when ACC uses brakes - might have been lucky
    ret.brake = 0
    #    ret.brakeLights = ret.brakePressed
    ret.gas = cp.vl["TPS_1"]['THROTTLE_POSITION']
    ret.gasPressed = ret.gas > 45  # up from 5

    ret.espDisabled = (cp.vl["CENTER_STACK"]['TRAC_OFF'] == 1)

    MAX_SPEED = 300  # hack to prevent outrageous numbers
    ret.wheelSpeeds.fl = MAX_SPEED if cp.vl['WHEEL_SPEEDS']['WHEEL_SPEED_LF'] >= MAX_SPEED else cp.vl['WHEEL_SPEEDS'][
      'WHEEL_SPEED_LF']
    ret.wheelSpeeds.rr = MAX_SPEED if cp.vl['WHEEL_SPEEDS']['WHEEL_SPEED_RR'] >= MAX_SPEED else cp.vl['WHEEL_SPEEDS'][
      'WHEEL_SPEED_RR']
    ret.wheelSpeeds.rl = MAX_SPEED if cp.vl['WHEEL_SPEEDS']['WHEEL_SPEED_LR'] >= MAX_SPEED else cp.vl['WHEEL_SPEEDS'][
      'WHEEL_SPEED_LR']
    ret.wheelSpeeds.fr = MAX_SPEED if cp.vl['WHEEL_SPEEDS']['WHEEL_SPEED_RF'] >= MAX_SPEED else cp.vl['WHEEL_SPEEDS'][
      'WHEEL_SPEED_RF']
    ret.vEgoRaw = (cp.vl['WHEEL_SPEEDS']['WHEEL_SPEED_LF'] + cp.vl['WHEEL_SPEEDS']['WHEEL_SPEED_RR']) / 2.
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
    ret.standstill = not ret.vEgoRaw > 0.001

    ret.leftBlinker = bool(cp.vl["STEERING_LEVERS"]['BLINKER_LEFT'])
    ret.rightBlinker = bool(cp.vl["STEERING_LEVERS"]['BLINKER_RIGHT'])
    ret.steeringAngleDeg = cp.vl["EPS_1"]['STEER_ANGLE']
    ret.steeringRateDeg = cp.vl["EPS_2"]['STEER_RATE_DRIVER']

    #    ret.cruiseState.enabled = cp_cam.vl["FORWARD_CAMERA_ACC"]['ACC_STATUS'] == 3  # ACC is green.
    #    ret.cruiseState.available = cp_cam.vl["FORWARD_CAMERA_ACC"]['ACC_STATUS'] == 1 # ACC is white... right???
    ret.cruiseState.available = True  # for dev
    ret.cruiseState.enabled = cp_cam.vl["FORWARD_CAMERA_ACC"]['ACC_STATUS'] == 3  # for dev
    #    ret.cruiseState.autoHighBeamBit = cp_cam.vl["FORWARD_CAMERA_HUD"]['AUTO_HIGH_BEAM_BIT'] # this might not be a cruisestate thing

    ret.cruiseState.speed = cp_cam.vl["FORWARD_CAMERA_CLUSTER"]['ACC_SET_SPEED'] * CV.KPH_TO_MS
    # CRUISE_STATE is a three bit msg, 0 is off, 1 and 2 are Non-ACC mode, 3 and 4 are ACC mode, find if there are other states too
    # ret.cruiseState.nonAdaptive = cp.vl["DASHBOARD"]['CRUISE_STATE'] in [1, 2]  do we need this at all?

    ret.steeringTorque = cp.vl["EPS_2"]["TORQUE_DRIVER"]
    # ret.steeringTorqueEps = cp.vl["EPS_2"]['TORQUE_MOTOR_IN']
    ret.steeringTorqueEps = cp.vl["EPS_1"]["TORQUE_MOTOR"]
    ret.steeringPressed = abs(ret.steeringTorque) > STEER_THRESHOLD
    steer_state = cp.vl["EPS_2"]["EPS_STATUS"]
    ret.steerError = steer_state == 0 or (steer_state == 0 and ret.vEgo > self.CP.minSteerSpeed)

    ret.genericToggle = bool(cp.vl["STEERING_LEVERS"]['HIGH_BEAM_FLASH'])

    gear = cp.vl["SHIFTER_ASSM"]['SHIFTER_POSITION']
    if gear in (4, 8):
      ret.gearShifter = GearShifter.drive
    elif gear == 3:
      ret.gearShifter = GearShifter.neutral
    elif gear == 1:
      ret.gearShifter = GearShifter.park
    elif gear == 2:
      ret.gearShifter = GearShifter.reverse
    else:
      ret.gearShifter = GearShifter.unknown

    return ret

  @staticmethod
  def get_can_parser(CP):
    signals = [
      # sig_name, sig_address, default
      ("SHIFTER_POSITION", "SHIFTER_ASSM", 0),
      ("DOOR_OPEN_LF", "DOORS", 0),
      ("DOOR_OPEN_RF", "DOORS", 0),
      ("DOOR_OPEN_LR", "DOORS", 0),
      ("DOOR_OPEN_RR", "DOORS", 0),
      ("BRAKE_PEDAL", "ABS_1", 0),
      ("DRIVER_BRAKE", "ABS_2", 0),
      ("THROTTLE_POSITION", "TPS_1", 0),
      ("WHEEL_SPEED_RR", "WHEEL_SPEEDS", 0),
      ("WHEEL_SPEED_LR", "WHEEL_SPEEDS", 0),
      ("WHEEL_SPEED_RF", "WHEEL_SPEEDS", 0),
      ("WHEEL_SPEED_LF", "WHEEL_SPEEDS", 0),
      ("STEER_ANGLE", "EPS_1", 0),
      ("STEER_RATE_DRIVER", "EPS_2", 0),
      ("BLINKER_LEFT", "STEERING_LEVERS", 0),
      ("BLINKER_RIGHT", "STEERING_LEVERS", 0),
      ("HIGH_BEAM_FLASH", "STEERING_LEVERS", 0),
      ("TORQUE_DRIVER", "EPS_2", 0),
      ("EPS_STATUS", "EPS_2", 0),
      ("TORQUE_MOTOR", "EPS_1", 0),
      #      ("TORQUE_MOTOR_IN", "EPS_2", 0),
      #      ("COUNTER", "EPS_2", -1),
      ("TRAC_OFF", "CENTER_STACK", 0),
      ("DRIVER_SEATBELT_STATUS", "OCCUPANT_RESTRAINT_MODULE", 0),
    ]

    checks = [
      # sig_address, frequency
      ("ABS_1", 50),
      ("ABS_2", 50),
      ("EPS_1", 100),
      ("EPS_2", 100),
      ("WHEEL_SPEEDS", 50),
      ("SHIFTER_ASSM", 50),
      ("TPS_1", 50),
      ("STEERING_LEVERS", 10),
      ("OCCUPANT_RESTRAINT_MODULE", 1),
      ("DOORS", 1),
      ("CENTER_STACK", 20),
      ("WHEEL_BUTTONS_CRUISE_CONTROL", 50),
    ]

    return CANParser(DBC[CP.carFingerprint]['pt'], signals, checks, 0)

  @staticmethod
  def get_cam_can_parser(CP):
    signals = [
      # sig_name, sig_address, default
      #      ("COUNTER", "FORWARD_CAMERA_LKAS", -1),
      ("ACC_STATUS", "FORWARD_CAMERA_ACC", 0),
      #      ("COUNTER", "FORWARD_CAMERA_ACC", -1),
      ("ACC_SET_SPEED", "FORWARD_CAMERA_CLUSTER", -1),
      #      ("LKAS_HUD", "FOWARD_CAMERA_HUD", -1),
      ("LKAS_CONTROL_BIT", "FORWARD_CAMERA_LKAS", 0),
      #      ("AUTO_HIGH_BEAM_BIT", "FORWARD_CAMERA_HUD", -1), # why didn't i check this before, is it broken?
    ]
    checks = [
      ("FORWARD_CAMERA_LKAS", 50),
      #      ("FORWARD_CAMERA_HUD", 15), # HUD needs work and auto high beams - add this later
      ("FORWARD_CAMERA_ACC", 50),
      ("FORWARD_CAMERA_CLUSTER", 50),
    ]

    return CANParser(DBC[CP.carFingerprint]['pt'], signals, checks, 2)
