#!/usr/bin/env python3

import unittest
from cereal import car, log
from selfdrive.car.honda.values import CAR as HONDA
from selfdrive.car.honda.carcontroller import CarController
from selfdrive.car.honda.interface import CarInterface
from common.realtime import sec_since_boot

from selfdrive.boardd.boardd import can_list_to_can_capnp
from selfdrive.config import Conversions as CV
import cereal.messaging as messaging
from cereal.services import service_list
from opendbc.can.parser import CANParser

import zmq
import time
import numpy as np


class TestHondaCarcontroller(unittest.TestCase):
  def test_honda_lkas_hud(self):
    self.longMessage = True

    sendcan = messaging.pub_sock('sendcan')

    car_name = HONDA.CIVIC
    params = CarInterface.get_params(car_name)
    CI = CarInterface(params, CarController)

    # Get parser
    parser_signals = [
      ('SET_ME_X41', 'LKAS_HUD', 0),
      ('SET_ME_X48', 'LKAS_HUD', 0),
      ('STEERING_REQUIRED', 'LKAS_HUD', 0),
      ('SOLID_LANES', 'LKAS_HUD', 0),
      ('LEAD_SPEED', 'RADAR_HUD', 0),
      ('LEAD_STATE', 'RADAR_HUD', 0),
      ('LEAD_DISTANCE', 'RADAR_HUD', 0),
      ('ACC_ALERTS', 'RADAR_HUD', 0),
    ]

    VA = car.CarControl.HUDControl.VisualAlert

    parser = CANParser(CI.cp.dbc_name, parser_signals, [], 0, sendcan=True, tcp_addr="127.0.0.1")
    time.sleep(0.2)  # Slow joiner syndrome


    alerts = {
      VA.none: 0,
      VA.brakePressed: 10,
      VA.wrongGear: 6,
      VA.seatbeltUnbuckled: 5,
      VA.speedTooHigh: 8,
    }

    for steer_required in [True, False]:
      for lanes in [True, False]:
        for alert in alerts.keys():
          control = car.CarControl.new_message()
          hud = car.CarControl.HUDControl.new_message()

          control.enabled = True

          if steer_required:
            hud.visualAlert = VA.steerRequired
          else:
            hud.visualAlert = alert

          hud.lanesVisible = lanes
          control.hudControl = hud

          CI.update(control)

          for _ in range(25):
            can_sends = CI.apply(control)
            sendcan.send(can_list_to_can_capnp(can_sends, msgtype='sendcan'))

          for _ in range(5):
            parser.update(int(sec_since_boot() * 1e9), False)
            time.sleep(0.01)

          self.assertEqual(0x41, parser.vl['LKAS_HUD']['SET_ME_X41'])
          self.assertEqual(0x48, parser.vl['LKAS_HUD']['SET_ME_X48'])
          self.assertEqual(steer_required, parser.vl['LKAS_HUD']['STEERING_REQUIRED'])
          self.assertEqual(lanes, parser.vl['LKAS_HUD']['SOLID_LANES'])

          self.assertEqual(0x1fe, parser.vl['RADAR_HUD']['LEAD_SPEED'])
          self.assertEqual(0x7, parser.vl['RADAR_HUD']['LEAD_STATE'])
          self.assertEqual(0x1e, parser.vl['RADAR_HUD']['LEAD_DISTANCE'])
          self.assertEqual(alerts[alert] if not steer_required else 0, parser.vl['RADAR_HUD']['ACC_ALERTS'])

  def test_honda_ui_cruise_speed(self):
    self.longMessage = True

    sendcan = messaging.pub_sock('sendcan')

    car_name = HONDA.CIVIC
    params = CarInterface.get_params(car_name)
    CI = CarInterface(params, CarController)

    # Get parser
    parser_signals = [
      # 780 - 0x30c
      ('CRUISE_SPEED', 'ACC_HUD', 0),
    ]
    parser = CANParser(CI.cp.dbc_name, parser_signals, [], 0, sendcan=True, tcp_addr="127.0.0.1")
    time.sleep(0.2)  # Slow joiner syndrome

    for cruise_speed in np.linspace(0, 50, 20):
      for visible in [False, True]:
        control = car.CarControl.new_message()
        hud = car.CarControl.HUDControl.new_message()
        hud.setSpeed = float(cruise_speed)
        hud.speedVisible = visible
        control.enabled = True
        control.hudControl = hud

        CI.update(control)

        for _ in range(25):
          can_sends = CI.apply(control)
          sendcan.send(can_list_to_can_capnp(can_sends, msgtype='sendcan'))

        for _ in range(5):
          parser.update(int(sec_since_boot() * 1e9), False)
          time.sleep(0.01)

        expected_cruise_speed = round(cruise_speed * CV.MS_TO_KPH)
        if not visible:
          expected_cruise_speed = 255

        self.assertAlmostEqual(parser.vl['ACC_HUD']['CRUISE_SPEED'], expected_cruise_speed, msg="Car: %s, speed: %.2f" % (car_name, cruise_speed))

  def test_honda_ui_pcm_accel(self):
    self.longMessage = True

    sendcan = messaging.pub_sock('sendcan')

    car_name = HONDA.CIVIC
    params = CarInterface.get_params(car_name)
    CI = CarInterface(params, CarController)

    # Get parser
    parser_signals = [
      # 780 - 0x30c
      ('PCM_GAS', 'ACC_HUD', 0),

    ]
    parser = CANParser(CI.cp.dbc_name, parser_signals, [], 0, sendcan=True, tcp_addr="127.0.0.1")
    time.sleep(0.2)  # Slow joiner syndrome

    for pcm_accel in np.linspace(0, 1, 25):
      cc = car.CarControl.CruiseControl.new_message()
      cc.accelOverride = float(pcm_accel)
      control = car.CarControl.new_message()
      control.enabled = True
      control.cruiseControl = cc

      CI.update(control)

      for _ in range(25):
        can_sends = CI.apply(control)
        sendcan.send(can_list_to_can_capnp(can_sends, msgtype='sendcan'))

      for _ in range(5):
        parser.update(int(sec_since_boot() * 1e9), False)
        time.sleep(0.01)

      self.assertAlmostEqual(parser.vl['ACC_HUD']['PCM_GAS'], int(0xc6 * pcm_accel), msg="Car: %s, accel: %.2f" % (car_name, pcm_accel))

  def test_honda_ui_pcm_speed(self):
    self.longMessage = True

    sendcan = messaging.pub_sock('sendcan')

    car_name = HONDA.CIVIC
    params = CarInterface.get_params(car_name)
    CI = CarInterface(params, CarController)

    # Get parser
    parser_signals = [
      # 780 - 0x30c
      ('PCM_SPEED', 'ACC_HUD', 99),
    ]
    parser = CANParser(CI.cp.dbc_name, parser_signals, [], 0, sendcan=True, tcp_addr="127.0.0.1")
    time.sleep(0.2)  # Slow joiner syndrome

    for pcm_speed in np.linspace(0, 100, 20):
      cc = car.CarControl.CruiseControl.new_message()
      cc.speedOverride = float(pcm_speed * CV.KPH_TO_MS)
      control = car.CarControl.new_message()
      control.enabled = True
      control.cruiseControl = cc

      CI.update(control)

      for _ in range(25):
        can_sends = CI.apply(control)
        sendcan.send(can_list_to_can_capnp(can_sends, msgtype='sendcan'))

      for _ in range(5):
        parser.update(int(sec_since_boot() * 1e9), False)
        time.sleep(0.01)

      self.assertAlmostEqual(parser.vl['ACC_HUD']['PCM_SPEED'], round(pcm_speed, 2), msg="Car: %s, speed: %.2f" % (car_name, pcm_speed))

  def test_honda_ui_hud_lead(self):
    self.longMessage = True

    sendcan = messaging.pub_sock('sendcan')

    for car_name in [HONDA.CIVIC]:
      params = CarInterface.get_params(car_name)
      CI = CarInterface(params, CarController)

      # Get parser
      parser_signals = [
        # 780 - 0x30c
        # 3: acc off, 2: solid car (hud_show_car), 1: dashed car (enabled, not hud show car), 0: no car (not enabled)
        ('HUD_LEAD', 'ACC_HUD', 99),
        ('SET_ME_X03', 'ACC_HUD', 99),
        ('SET_ME_X03_2', 'ACC_HUD', 99),
        ('SET_ME_X01', 'ACC_HUD', 99),
        ('ENABLE_MINI_CAR', 'ACC_HUD', 99),
      ]
      parser = CANParser(CI.cp.dbc_name, parser_signals, [], 0, sendcan=True, tcp_addr="127.0.0.1")
      time.sleep(0.2)  # Slow joiner syndrome

      for enabled in [True, False]:
        for leadVisible in [True, False]:

          control = car.CarControl.new_message()
          hud = car.CarControl.HUDControl.new_message()
          hud.leadVisible = leadVisible
          control.enabled = enabled
          control.hudControl = hud
          CI.update(control)

          for _ in range(25):
            can_sends = CI.apply(control)
            sendcan.send(can_list_to_can_capnp(can_sends, msgtype='sendcan'))

          for _ in range(5):
            parser.update(int(sec_since_boot() * 1e9), False)
            time.sleep(0.01)

          if not enabled:
            hud_lead = 0
          else:
            hud_lead = 2 if leadVisible else 1
          self.assertEqual(int(parser.vl['ACC_HUD']['HUD_LEAD']), hud_lead, msg="Car: %s, lead: %s, enabled %s" % (car_name, leadVisible, enabled))
          self.assertTrue(parser.vl['ACC_HUD']['ENABLE_MINI_CAR'])
          self.assertEqual(0x3, parser.vl['ACC_HUD']['SET_ME_X03'])
          self.assertEqual(0x3, parser.vl['ACC_HUD']['SET_ME_X03_2'])
          self.assertEqual(0x1, parser.vl['ACC_HUD']['SET_ME_X01'])


  def test_honda_steering(self):
    self.longMessage = True
    limits = {
      HONDA.CIVIC: 0x1000,
      HONDA.ODYSSEY: 0x1000,
      HONDA.PILOT: 0x1000,
      HONDA.CRV: 0x3e8,
      HONDA.ACURA_ILX: 0xF00,
      HONDA.ACURA_RDX: 0x3e8,
    }

    sendcan = messaging.pub_sock('sendcan')

    for car_name in limits.keys():
      params = CarInterface.get_params(car_name)
      CI = CarInterface(params, CarController)

      # Get parser
      parser_signals = [
        ('STEER_TORQUE', 'STEERING_CONTROL', 0),
      ]
      parser = CANParser(CI.cp.dbc_name, parser_signals, [], 0, sendcan=True, tcp_addr="127.0.0.1")
      time.sleep(0.2)  # Slow joiner syndrome

      for steer in np.linspace(-1., 1., 25):
        control = car.CarControl.new_message()
        actuators = car.CarControl.Actuators.new_message()
        actuators.steer = float(steer)
        control.enabled = True
        control.actuators = actuators
        CI.update(control)

        CI.CS.steer_not_allowed = False

        for _ in range(25):
          can_sends = CI.apply(control)
          sendcan.send(can_list_to_can_capnp(can_sends, msgtype='sendcan'))

        for _ in range(5):
          parser.update(int(sec_since_boot() * 1e9), False)
          time.sleep(0.01)

        torque = parser.vl['STEERING_CONTROL']['STEER_TORQUE']
        self.assertAlmostEqual(int(limits[car_name] * -actuators.steer), torque, msg="Car: %s, steer %.2f" % (car_name, steer))

    sendcan.close()

  def test_honda_gas(self):
    self.longMessage = True

    sendcan = messaging.pub_sock('sendcan')

    car_name = HONDA.ACURA_ILX

    params = CarInterface.get_params(car_name, {0: {0x201: 6}, 1: {}, 2: {}})  # Add interceptor to fingerprint
    CI = CarInterface(params, CarController)

    # Get parser
    parser_signals = [
      ('GAS_COMMAND', 'GAS_COMMAND', -1),
      ('GAS_COMMAND2', 'GAS_COMMAND', -1),
      ('ENABLE', 'GAS_COMMAND', -1),
    ]
    parser = CANParser(CI.cp.dbc_name, parser_signals, [], 0, sendcan=True, tcp_addr="127.0.0.1")
    time.sleep(0.2)  # Slow joiner syndrome

    for gas in np.linspace(0., 0.95, 25):
      control = car.CarControl.new_message()
      actuators = car.CarControl.Actuators.new_message()
      actuators.gas = float(gas)
      control.enabled = True
      control.actuators = actuators
      CI.update(control)

      CI.CS.steer_not_allowed = False

      for _ in range(25):
        can_sends = CI.apply(control)
        sendcan.send(can_list_to_can_capnp(can_sends, msgtype='sendcan'))

      for _ in range(5):
        parser.update(int(sec_since_boot() * 1e9), False)
        time.sleep(0.01)

      gas_command = parser.vl['GAS_COMMAND']['GAS_COMMAND'] / 255.0
      gas_command2 = parser.vl['GAS_COMMAND']['GAS_COMMAND2'] / 255.0
      enabled = gas > 0.001
      self.assertEqual(enabled, parser.vl['GAS_COMMAND']['ENABLE'], msg="Car: %s, gas %.2f" % (car_name, gas))
      if enabled:
        self.assertAlmostEqual(gas, gas_command, places=2, msg="Car: %s, gas %.2f" % (car_name, gas))
        self.assertAlmostEqual(gas, gas_command2, places=2, msg="Car: %s, gas %.2f" % (car_name, gas))

    sendcan.close()

  def test_honda_brake(self):
    self.longMessage = True

    sendcan = messaging.pub_sock('sendcan')

    car_name = HONDA.CIVIC

    params = CarInterface.get_params(car_name)
    CI = CarInterface(params, CarController)

    # Get parser
    parser_signals = [
      ('COMPUTER_BRAKE', 'BRAKE_COMMAND', 0),
      ('BRAKE_PUMP_REQUEST', 'BRAKE_COMMAND', 0),  # pump_on
      ('CRUISE_OVERRIDE', 'BRAKE_COMMAND', 0),  # pcm_override
      ('CRUISE_FAULT_CMD', 'BRAKE_COMMAND', 0),  # pcm_fault_cmd
      ('CRUISE_CANCEL_CMD', 'BRAKE_COMMAND', 0),  # pcm_cancel_cmd
      ('COMPUTER_BRAKE_REQUEST', 'BRAKE_COMMAND', 0),  # brake_rq
      ('SET_ME_0X80', 'BRAKE_COMMAND', 0),
      ('BRAKE_LIGHTS', 'BRAKE_COMMAND', 0),  # brakelights
      ('FCW', 'BRAKE_COMMAND', 0),
    ]
    parser = CANParser(CI.cp.dbc_name, parser_signals, [], 0, sendcan=True, tcp_addr="127.0.0.1")
    time.sleep(0.2)  # Slow joiner syndrome

    VA = car.CarControl.HUDControl.VisualAlert

    for override in [True, False]:
      for cancel in [True, False]:
        for fcw in [True, False]:
          steps = 25 if not override and not cancel else 2
          for brake in np.linspace(0., 0.95, steps):
            control = car.CarControl.new_message()

            hud = car.CarControl.HUDControl.new_message()
            if fcw:
              hud.visualAlert = VA.fcw

            cruise = car.CarControl.CruiseControl.new_message()
            cruise.cancel = cancel
            cruise.override = override

            actuators = car.CarControl.Actuators.new_message()
            actuators.brake = float(brake)

            control.enabled = True
            control.actuators = actuators
            control.hudControl = hud
            control.cruiseControl = cruise

            CI.update(control)

            CI.CS.steer_not_allowed = False

            for _ in range(20):
              can_sends = CI.apply(control)
              sendcan.send(can_list_to_can_capnp(can_sends, msgtype='sendcan'))

            for _ in range(5):
              parser.update(int(sec_since_boot() * 1e9), False)
              time.sleep(0.01)

            brake_command = parser.vl['BRAKE_COMMAND']['COMPUTER_BRAKE']
            min_expected_brake = int(1024 / 4 * (actuators.brake - 0.02))
            max_expected_brake = int(1024 / 4 * (actuators.brake + 0.02))
            braking = actuators.brake > 0

            braking_ok = min_expected_brake <= brake_command <= max_expected_brake
            if steps == 2:
              braking_ok = True

            self.assertTrue(braking_ok, msg="Car: %s, brake %.2f" % (car_name, brake))
            self.assertEqual(0x80, parser.vl['BRAKE_COMMAND']['SET_ME_0X80'])
            self.assertEqual(braking, parser.vl['BRAKE_COMMAND']['BRAKE_PUMP_REQUEST'])
            self.assertEqual(braking, parser.vl['BRAKE_COMMAND']['COMPUTER_BRAKE_REQUEST'])
            self.assertEqual(braking, parser.vl['BRAKE_COMMAND']['BRAKE_LIGHTS'])
            self.assertFalse(parser.vl['BRAKE_COMMAND']['CRUISE_FAULT_CMD'])
            self.assertEqual(override, parser.vl['BRAKE_COMMAND']['CRUISE_OVERRIDE'])
            self.assertEqual(cancel, parser.vl['BRAKE_COMMAND']['CRUISE_CANCEL_CMD'])
            self.assertEqual(fcw, bool(parser.vl['BRAKE_COMMAND']['FCW']))

if __name__ == '__main__':
  unittest.main()
