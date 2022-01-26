from cereal import car
from selfdrive.car import apply_std_steer_torque_limits
from selfdrive.car.faw import volkswagencan
from selfdrive.car.faw.values import DBC_FILES, CANBUS, MQB_LDW_MESSAGES, BUTTON_STATES, CarControllerParams as P
from opendbc.can.packer import CANPacker


class CarController():
  def __init__(self, dbc_name, CP, VM):    
    self.packer_pt = CANPacker(DBC_FILES.mqb)

    self.hcaSameTorqueCount = 0
    self.hcaEnabledFrameCount = 0
    self.graButtonStatesToSend = None
    self.graMsgSentCount = 0
    self.graMsgStartFramePrev = 0
    self.graMsgBusCounterPrev = 0

    self.steer_rate_limited = False

  def update(self, enabled, CS, frame, ext_bus, actuators, visual_alert, left_lane_visible, right_lane_visible, left_lane_depart, right_lane_depart):
    """ Controls thread """
    can_sends = []
    new_actuators = actuators.copy()

    return new_actuators, can_sends