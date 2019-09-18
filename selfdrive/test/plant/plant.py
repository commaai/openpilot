#!/usr/bin/env python
import os
import struct
import time
from collections import namedtuple
import numpy as np

from opendbc import DBC_PATH

from common.realtime import Ratekeeper
from selfdrive.config import Conversions as CV
import selfdrive.messaging as messaging
from selfdrive.services import service_list
from selfdrive.car import crc8_pedal
from selfdrive.car.honda.hondacan import fix
from selfdrive.car.honda.values import CAR
from selfdrive.car.honda.carstate import get_can_signals
from selfdrive.boardd.boardd import can_capnp_to_can_list, can_list_to_can_capnp

from selfdrive.can.plant_can_parser import CANParser
from selfdrive.car.honda.interface import CarInterface

from common.dbc import dbc
honda = dbc(os.path.join(DBC_PATH, "honda_civic_touring_2016_can_generated.dbc"))

# Trick: set 0x201 (interceptor) in fingerprints for gas is controlled like if there was an interceptor
CP = CarInterface.get_params(CAR.CIVIC, {0x201})


def car_plant(pos, speed, grade, gas, brake):
  # vehicle parameters
  mass = 1700
  aero_cd = 0.3
  force_peak = mass*3.
  force_brake_peak = -mass*10.     #1g
  power_peak = 100000   # 100kW
  speed_base = power_peak/force_peak
  rolling_res = 0.01
  g = 9.81
  frontal_area = 2.2
  air_density = 1.225
  gas_to_peak_linear_slope = 3.33
  brake_to_peak_linear_slope = 0.3
  creep_accel_v = [1., 0.]
  creep_accel_bp = [0., 1.5]

  #*** longitudinal model ***
  # find speed where peak torque meets peak power
  force_brake = brake * force_brake_peak * brake_to_peak_linear_slope
  if speed < speed_base: # torque control
    force_gas = gas * force_peak * gas_to_peak_linear_slope
  else: # power control
    force_gas = gas * power_peak / speed * gas_to_peak_linear_slope

  force_grade = - grade * mass  # positive grade means uphill

  creep_accel = np.interp(speed, creep_accel_bp, creep_accel_v)
  force_creep = creep_accel * mass

  force_resistance = -(rolling_res * mass * g + 0.5 * speed**2 * aero_cd * air_density * frontal_area)
  force = force_gas + force_brake + force_resistance + force_grade + force_creep
  acceleration = force / mass

  # TODO: lateral model
  return speed, acceleration

def get_car_can_parser():
  dbc_f = 'honda_civic_touring_2016_can_generated.dbc'
  signals = [
    ("STEER_TORQUE", 0xe4, 0),
    ("STEER_TORQUE_REQUEST", 0xe4, 0),
    ("COMPUTER_BRAKE", 0x1fa, 0),
    ("COMPUTER_BRAKE_REQUEST", 0x1fa, 0),
    ("GAS_COMMAND", 0x200, 0),
  ]
  checks = [
    (0xe4, 100),
    (0x1fa, 50),
    (0x200, 50),
  ]
  return CANParser(dbc_f, signals, checks)

def to_3_byte(x):
  return struct.pack("!H", int(x)).encode("hex")[1:]

def to_3s_byte(x):
  return struct.pack("!h", int(x)).encode("hex")[1:]

class Plant(object):
  messaging_initialized = False

  def __init__(self, lead_relevancy=False, rate=100, speed=0.0, distance_lead=2.0):
    self.rate = rate

    if not Plant.messaging_initialized:
      Plant.logcan = messaging.pub_sock(service_list['can'].port)
      Plant.sendcan = messaging.sub_sock(service_list['sendcan'].port)
      Plant.model = messaging.pub_sock(service_list['model'].port)
      Plant.live_params = messaging.pub_sock(service_list['liveParameters'].port)
      Plant.health = messaging.pub_sock(service_list['health'].port)
      Plant.thermal = messaging.pub_sock(service_list['thermal'].port)
      Plant.driverMonitoring = messaging.pub_sock(service_list['driverMonitoring'].port)
      Plant.cal = messaging.pub_sock(service_list['liveCalibration'].port)
      Plant.controls_state = messaging.sub_sock(service_list['controlsState'].port)
      Plant.plan = messaging.sub_sock(service_list['plan'].port)
      Plant.messaging_initialized = True

    self.angle_steer = 0.
    self.gear_choice = 0
    self.speed, self.speed_prev = 0., 0.

    self.esp_disabled = 0
    self.main_on = 1
    self.user_gas = 0
    self.computer_brake,self.user_brake = 0,0
    self.brake_pressed = 0
    self.angle_steer_rate = 0
    self.distance, self.distance_prev = 0., 0.
    self.speed, self.speed_prev = speed, speed
    self.steer_error, self.brake_error, self.steer_not_allowed = 0, 0, 0
    self.gear_shifter = 8   # D gear
    self.pedal_gas = 0
    self.cruise_setting = 0

    self.seatbelt, self.door_all_closed = True, True
    self.steer_torque, self.v_cruise, self.acc_status = 0, 0, 0  # v_cruise is reported from can, not the one used for controls

    self.lead_relevancy = lead_relevancy

    # lead car
    self.distance_lead, self.distance_lead_prev = distance_lead , distance_lead

    self.rk = Ratekeeper(rate, print_delay_threshold=100)
    self.ts = 1./rate

    self.cp = get_car_can_parser()
    self.response_seen = False

    time.sleep(1)
    messaging.drain_sock(Plant.sendcan)
    messaging.drain_sock(Plant.controls_state)

  def close(self):
    Plant.logcan.close()
    Plant.model.close()
    Plant.live_params.close()

  def speed_sensor(self, speed):
    if speed<0.3:
      return 0
    else:
      return speed * CV.MS_TO_KPH

  def current_time(self):
    return float(self.rk.frame) / self.rate

  def step(self, v_lead=0.0, cruise_buttons=None, grade=0.0, publish_model = True):
    gen_signals, gen_checks = get_can_signals(CP)
    sgs = [s[0] for s in gen_signals]
    msgs = [s[1] for s in gen_signals]
    cks_msgs = set(check[0] for check in gen_checks)
    cks_msgs.add(0x18F)
    cks_msgs.add(0x30C)

    # ******** get messages sent to the car ********
    can_msgs = []
    for a in messaging.drain_sock(Plant.sendcan, wait_for_one=self.response_seen):
      can_msgs.extend(can_capnp_to_can_list(a.sendcan, [0,2]))

    # After the first response the car is done fingerprinting, so we can run in lockstep with controlsd
    if can_msgs:
      self.response_seen = True

    self.cp.update_can(can_msgs)

    # ******** get controlsState messages for plotting ***
    controls_state_msgs = []
    for a in messaging.drain_sock(Plant.controls_state, wait_for_one=self.response_seen):
      controls_state_msgs.append(a.controlsState)

    fcw = None
    for a in messaging.drain_sock(Plant.plan):
      if a.plan.fcw:
        fcw = True

    if self.cp.vl[0x1fa]['COMPUTER_BRAKE_REQUEST']:
      brake = self.cp.vl[0x1fa]['COMPUTER_BRAKE'] * 0.003906248
    else:
      brake = 0.0

    if self.cp.vl[0x200]['GAS_COMMAND'] > 0:
      gas = self.cp.vl[0x200]['GAS_COMMAND'] / 256.0
    else:
      gas = 0.0

    if self.cp.vl[0xe4]['STEER_TORQUE_REQUEST']:
      steer_torque = self.cp.vl[0xe4]['STEER_TORQUE']*1.0/0xf00
    else:
      steer_torque = 0.0

    distance_lead = self.distance_lead_prev + v_lead * self.ts

    # ******** run the car ********
    speed, acceleration = car_plant(self.distance_prev, self.speed_prev, grade, gas, brake)
    distance = self.distance_prev + speed * self.ts
    speed = self.speed_prev + self.ts * acceleration
    if speed <= 0:
      speed = 0
      acceleration = 0

    # ******** lateral ********
    self.angle_steer -= (steer_torque/10.0) * self.ts

    # *** radar model ***
    if self.lead_relevancy:
      d_rel = np.maximum(0., distance_lead - distance)
      v_rel = v_lead - speed
    else:
      d_rel = 200.
      v_rel = 0.
    lateral_pos_rel = 0.

    # print at 5hz
    if (self.rk.frame % (self.rate//5)) == 0:
      print("%6.2f m  %6.2f m/s  %6.2f m/s2   %.2f ang   gas: %.2f  brake: %.2f  steer: %5.2f     lead_rel: %6.2f m  %6.2f m/s" % (distance, speed, acceleration, self.angle_steer, gas, brake, steer_torque, d_rel, v_rel))

    # ******** publish the car ********
    vls_tuple = namedtuple('vls', [
           'XMISSION_SPEED',
           'WHEEL_SPEED_FL', 'WHEEL_SPEED_FR', 'WHEEL_SPEED_RL', 'WHEEL_SPEED_RR',
           'STEER_ANGLE', 'STEER_ANGLE_RATE', 'STEER_TORQUE_SENSOR', 'STEER_TORQUE_MOTOR',
           'LEFT_BLINKER', 'RIGHT_BLINKER',
           'GEAR',
           'WHEELS_MOVING',
           'BRAKE_ERROR_1', 'BRAKE_ERROR_2',
           'SEATBELT_DRIVER_LAMP', 'SEATBELT_DRIVER_LATCHED',
           'BRAKE_PRESSED', 'BRAKE_SWITCH',
           'CRUISE_BUTTONS',
           'ESP_DISABLED',
           'HUD_LEAD',
           'USER_BRAKE',
           'STEER_STATUS',
           'GEAR_SHIFTER',
           'PEDAL_GAS',
           'CRUISE_SETTING',
           'ACC_STATUS',

           'CRUISE_SPEED_PCM',
           'CRUISE_SPEED_OFFSET',

           'DOOR_OPEN_FL', 'DOOR_OPEN_FR', 'DOOR_OPEN_RL', 'DOOR_OPEN_RR',

           'CAR_GAS',
           'MAIN_ON',
           'EPB_STATE',
           'BRAKE_HOLD_ACTIVE',
           'INTERCEPTOR_GAS',
           'INTERCEPTOR_GAS2',
           'IMPERIAL_UNIT',
           ])
    vls = vls_tuple(
           self.speed_sensor(speed),
           self.speed_sensor(speed), self.speed_sensor(speed), self.speed_sensor(speed), self.speed_sensor(speed),
           self.angle_steer, self.angle_steer_rate, 0, 0,#Steer torque sensor
           0, 0,  # Blinkers
           self.gear_choice,
           speed != 0,
           self.brake_error, self.brake_error,
           not self.seatbelt, self.seatbelt,  # Seatbelt
           self.brake_pressed, 0., #Brake pressed, Brake switch
           cruise_buttons,
           self.esp_disabled,
           0,  # HUD lead
           self.user_brake,
           self.steer_error,
           self.gear_shifter,
           self.pedal_gas,
           self.cruise_setting,
           self.acc_status,

           self.v_cruise,
           0,  # Cruise speed offset

           0, 0, 0, 0,  # Doors

           self.user_gas,
           self.main_on,
           0,  # EPB State
           0,  # Brake hold
           0,  # Interceptor feedback
           0,  # Interceptor 2 feedback
           False
           )

    # TODO: publish each message at proper frequency
    can_msgs = []
    for msg in set(msgs):
      msg_struct = {}
      indxs = [i for i, x in enumerate(msgs) if msg == msgs[i]]
      for i in indxs:
        msg_struct[sgs[i]] = getattr(vls, sgs[i])

      if "COUNTER" in honda.get_signals(msg):
        msg_struct["COUNTER"] = self.rk.frame % 4

      if "COUNTER_PEDAL" in honda.get_signals(msg):
        msg_struct["COUNTER_PEDAL"] = self.rk.frame % 0xf

      msg = honda.lookup_msg_id(msg)
      msg_data = honda.encode(msg, msg_struct)

      if "CHECKSUM" in honda.get_signals(msg):
        msg_data = fix(msg_data, msg)

      if "CHECKSUM_PEDAL" in honda.get_signals(msg):
        msg_struct["CHECKSUM_PEDAL"] = crc8_pedal([ord(i) for i in msg_data][:-1])
        msg_data = honda.encode(msg, msg_struct)

      can_msgs.append([msg, 0, msg_data, 0])

    # add the radar message
    # TODO: use the DBC
    if self.rk.frame % 5 == 0:
      radar_state_msg = '\x79\x00\x00\x00\x00\x00\x00\x00'
      radar_msg = to_3_byte(d_rel*16.0) + \
                  to_3_byte(int(lateral_pos_rel*16.0)&0x3ff) + \
                  to_3s_byte(int(v_rel*32.0)) + \
                  "0f00000"
      can_msgs.append([0x400, 0, radar_state_msg, 1])
      can_msgs.append([0x445, 0, radar_msg.decode("hex"), 1])

    # add camera msg so controlsd thinks it's alive
    msg_struct["COUNTER"] = self.rk.frame % 4
    msg = honda.lookup_msg_id(0xe4)
    msg_data = honda.encode(msg, msg_struct)
    msg_data = fix(msg_data, 0xe4)
    can_msgs.append([0xe4, 0, msg_data, 2])


    # Fake sockets that controlsd subscribes to
    live_parameters = messaging.new_message()
    live_parameters.init('liveParameters')
    live_parameters.liveParameters.valid = True
    live_parameters.liveParameters.sensorValid = True
    live_parameters.liveParameters.posenetValid = True
    live_parameters.liveParameters.steerRatio = CP.steerRatio
    live_parameters.liveParameters.stiffnessFactor = 1.0
    Plant.live_params.send(live_parameters.to_bytes())

    driver_monitoring = messaging.new_message()
    driver_monitoring.init('driverMonitoring')
    driver_monitoring.driverMonitoring.faceOrientation = [0.] * 3
    driver_monitoring.driverMonitoring.facePosition = [0.] * 2
    Plant.driverMonitoring.send(driver_monitoring.to_bytes())

    health = messaging.new_message()
    health.init('health')
    health.health.controlsAllowed = True
    Plant.health.send(health.to_bytes())

    thermal = messaging.new_message()
    thermal.init('thermal')
    thermal.thermal.freeSpace = 1.
    thermal.thermal.batteryPercent = 100
    Plant.thermal.send(thermal.to_bytes())

    # ******** publish a fake model going straight and fake calibration ********
    # note that this is worst case for MPC, since model will delay long mpc by one time step
    if publish_model and self.rk.frame % 5 == 0:
      md = messaging.new_message()
      cal = messaging.new_message()
      md.init('model')
      cal.init('liveCalibration')
      md.model.frameId = 0
      for x in [md.model.path, md.model.leftLane, md.model.rightLane]:
        x.points = [0.0]*50
        x.prob = 1.0
        x.std = 1.0

      if self.lead_relevancy:
        d_rel = np.maximum(0., distance_lead - distance)
        v_rel = v_lead - speed
        prob = 1.0
      else:
        d_rel = 200.
        v_rel = 0.
        prob = 0.0

      md.model.lead.dist = float(d_rel)
      md.model.lead.prob = prob
      md.model.lead.relY = 0.0
      md.model.lead.relYStd = 1.
      md.model.lead.relVel = float(v_rel)
      md.model.lead.relVelStd = 1.
      md.model.lead.relA = 0.0
      md.model.lead.relAStd = 10.
      md.model.lead.std = 1.0

      cal.liveCalibration.calStatus = 1
      cal.liveCalibration.calPerc = 100
      cal.liveCalibration.rpyCalib = [0.] * 3
      # fake values?
      Plant.model.send(md.to_bytes())
      Plant.cal.send(cal.to_bytes())

    Plant.logcan.send(can_list_to_can_capnp(can_msgs))

    # ******** update prevs ********
    self.speed = speed
    self.distance = distance
    self.distance_lead = distance_lead

    self.speed_prev = speed
    self.distance_prev = distance
    self.distance_lead_prev = distance_lead

    if self.response_seen:
      self.rk.monitor_time()
    else:
      self.rk.keep_time()

    return {
      "distance": distance,
      "speed": speed,
      "acceleration": acceleration,
      "distance_lead": distance_lead,
      "brake": brake,
      "gas": gas,
      "steer_torque": steer_torque,
      "fcw": fcw,
      "controls_state_msgs": controls_state_msgs,
    }

# simple engage in standalone mode
def plant_thread(rate=100):
  plant = Plant(rate)
  while 1:
    plant.step()

if __name__ == "__main__":
  plant_thread()
