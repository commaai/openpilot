#!/usr/bin/env python3

import unittest
from cereal import car, log
from selfdrive.car.toyota.values import CAR as TOYOTA
from selfdrive.car.toyota.carcontroller import CarController
from selfdrive.car.toyota.interface import CarInterface
from common.realtime import sec_since_boot

from selfdrive.boardd.boardd import can_list_to_can_capnp
from selfdrive.config import Conversions as CV
import cereal.messaging as messaging
from cereal.services import service_list
from opendbc.can.parser import CANParser
import zmq
import time
import numpy as np


class TestToyotaCarcontroller(unittest.TestCase):
  def test_fcw(self):
    # TODO: This message has a 0xc1 setme which is not yet checked or in the dbc file
    self.longMessage = True
    car_name = TOYOTA.RAV4

    sendcan = messaging.pub_sock('sendcan')

    params = CarInterface.get_params(car_name)
    CI = CarInterface(params, CarController)

    # Get parser
    parser_signals = [
      ('FCW', 'ACC_HUD', 0),
      ('SET_ME_X20', 'ACC_HUD', 0),
      ('SET_ME_X10', 'ACC_HUD', 0),
      ('SET_ME_X80', 'ACC_HUD', 0),
    ]
    parser = CANParser(CI.cp.dbc_name, parser_signals, [], 0, sendcan=True, tcp_addr="127.0.0.1")
    time.sleep(0.2)  # Slow joiner syndrome

    VA = car.CarControl.HUDControl.VisualAlert
    for fcw in [True, False]:
      control = car.CarControl.new_message()
      control.enabled = True

      hud = car.CarControl.HUDControl.new_message()
      if fcw:
        hud.visualAlert = VA.fcw
        control.hudControl = hud

      CI.update(control)

      for _ in range(200):
        can_sends = CI.apply(control)
        sendcan.send(can_list_to_can_capnp(can_sends, msgtype='sendcan'))

      for _ in range(5):
        parser.update(int(sec_since_boot() * 1e9), False)
        time.sleep(0.01)

      self.assertEqual(fcw, parser.vl['ACC_HUD']['FCW'])
      self.assertEqual(0x20, parser.vl['ACC_HUD']['SET_ME_X20'])
      self.assertEqual(0x10, parser.vl['ACC_HUD']['SET_ME_X10'])
      self.assertEqual(0x80, parser.vl['ACC_HUD']['SET_ME_X80'])

  def test_ui(self):
    self.longMessage = True
    car_name = TOYOTA.RAV4

    sendcan = messaging.pub_sock('sendcan')

    params = CarInterface.get_params(car_name)
    CI = CarInterface(params, CarController)

    # Get parser
    parser_signals = [
      ('BARRIERS', 'LKAS_HUD', -1),
      ('RIGHT_LINE', 'LKAS_HUD', 0),
      ('LEFT_LINE', 'LKAS_HUD', 0),
      ('SET_ME_X01', 'LKAS_HUD', 0),
      ('SET_ME_X01_2', 'LKAS_HUD', 0),
      ('LDA_ALERT', 'LKAS_HUD', -1),
      ('SET_ME_X0C', 'LKAS_HUD', 0),
      ('SET_ME_X2C', 'LKAS_HUD', 0),
      ('SET_ME_X38', 'LKAS_HUD', 0),
      ('SET_ME_X02', 'LKAS_HUD', 0),
    ]
    parser = CANParser(CI.cp.dbc_name, parser_signals, [], 0, sendcan=True, tcp_addr="127.0.0.1")
    time.sleep(0.2)  # Slow joiner syndrome

    VA = car.CarControl.HUDControl.VisualAlert

    for left_lane in [True, False]:
      for right_lane in [True, False]:
        for steer in [True, False]:
          control = car.CarControl.new_message()
          control.enabled = True

          hud = car.CarControl.HUDControl.new_message()
          if steer:
            hud.visualAlert = VA.steerRequired

          hud.leftLaneVisible = left_lane
          hud.rightLaneVisible = right_lane

          control.hudControl = hud
          CI.update(control)

            for _ in range(200):  # UI is only sent at 1Hz
              can_sends = CI.apply(control)
              sendcan.send(can_list_to_can_capnp(can_sends, msgtype='sendcan'))

            for _ in range(5):
              parser.update(int(sec_since_boot() * 1e9), False)
              time.sleep(0.01)

            self.assertEqual(0x0c, parser.vl['LKAS_HUD']['SET_ME_X0C'])
            self.assertEqual(0x2c, parser.vl['LKAS_HUD']['SET_ME_X2C'])
            self.assertEqual(0x38, parser.vl['LKAS_HUD']['SET_ME_X38'])
            self.assertEqual(0x02, parser.vl['LKAS_HUD']['SET_ME_X02'])
            self.assertEqual(0, parser.vl['LKAS_HUD']['BARRIERS'])
            self.assertEqual(1 if right_lane else 2, parser.vl['LKAS_HUD']['RIGHT_LINE'])
            self.assertEqual(1 if left_lane else 2, parser.vl['LKAS_HUD']['LEFT_LINE'])
            self.assertEqual(1, parser.vl['LKAS_HUD']['SET_ME_X01'])
            self.assertEqual(1, parser.vl['LKAS_HUD']['SET_ME_X01_2'])
            self.assertEqual(steer, parser.vl['LKAS_HUD']['LDA_ALERT'])

  def test_standstill_and_cancel(self):
    self.longMessage = True
    car_name = TOYOTA.RAV4

    sendcan = messaging.pub_sock('sendcan')

    params = CarInterface.get_params(car_name)
    CI = CarInterface(params, CarController)

    # Get parser
    parser_signals = [
      ('RELEASE_STANDSTILL', 'ACC_CONTROL', 0),
      ('CANCEL_REQ', 'ACC_CONTROL', 0),
      ('SET_ME_X3', 'ACC_CONTROL', 0),
      ('SET_ME_1', 'ACC_CONTROL', 0),
    ]
    parser = CANParser(CI.cp.dbc_name, parser_signals, [], 0, sendcan=True, tcp_addr="127.0.0.1")
    time.sleep(0.2)  # Slow joiner syndrome

    control = car.CarControl.new_message()
    control.enabled = True

    CI.update(control)

    CI.CS.pcm_acc_status = 8  # Active
    CI.CS.standstill = True
    can_sends = CI.apply(control)

    sendcan.send(can_list_to_can_capnp(can_sends, msgtype='sendcan'))

    for _ in range(5):
      parser.update(int(sec_since_boot() * 1e9), False)
      time.sleep(0.01)

    self.assertEqual(0x3, parser.vl['ACC_CONTROL']['SET_ME_X3'])
    self.assertEqual(1, parser.vl['ACC_CONTROL']['SET_ME_1'])
    self.assertFalse(parser.vl['ACC_CONTROL']['RELEASE_STANDSTILL'])
    self.assertFalse(parser.vl['ACC_CONTROL']['CANCEL_REQ'])

    CI.CS.pcm_acc_status = 7  # Standstill

    for _ in range(10):
      can_sends = CI.apply(control)
      sendcan.send(can_list_to_can_capnp(can_sends, msgtype='sendcan'))

    for _ in range(5):
      parser.update(int(sec_since_boot() * 1e9), False)
      time.sleep(0.01)

    self.assertTrue(parser.vl['ACC_CONTROL']['RELEASE_STANDSTILL'])

    cruise = car.CarControl.CruiseControl.new_message()
    cruise.cancel = True
    control.cruiseControl = cruise

    for _ in range(10):
      can_sends = CI.apply(control)

    sendcan.send(can_list_to_can_capnp(can_sends, msgtype='sendcan'))

    for _ in range(5):
      parser.update(int(sec_since_boot() * 1e9), False)
      time.sleep(0.01)

    self.assertTrue(parser.vl['ACC_CONTROL']['CANCEL_REQ'])

  @unittest.skip("IPAS logic changed, fix test")
  def test_steering_ipas(self):
    self.longMessage = True
    car_name = TOYOTA.RAV4

    sendcan = messaging.pub_sock('sendcan')

    params = CarInterface.get_params(car_name)
    params.enableApgs = True
    CI = CarInterface(params, CarController)
    CI.CC.angle_control = True

    # Get parser
    parser_signals = [
      ('SET_ME_X10', 'STEERING_IPAS', 0),
      ('SET_ME_X40', 'STEERING_IPAS', 0),
      ('ANGLE', 'STEERING_IPAS', 0),
      ('STATE', 'STEERING_IPAS', 0),
      ('DIRECTION_CMD', 'STEERING_IPAS', 0),
    ]
    parser = CANParser(CI.cp.dbc_name, parser_signals, [], 0, sendcan=True, tcp_addr="127.0.0.1")
    time.sleep(0.2)  # Slow joiner syndrome

    for enabled in [True, False]:
      for steer in np.linspace(-510., 510., 25):
          control = car.CarControl.new_message()
          actuators = car.CarControl.Actuators.new_message()
          actuators.steerAngle = float(steer)
          control.enabled = enabled
          control.actuators = actuators
          CI.update(control)

          CI.CS.steer_not_allowed = False

          for _ in range(1000 if steer < -505 else 25):
            can_sends = CI.apply(control)
            sendcan.send(can_list_to_can_capnp(can_sends, msgtype='sendcan'))
            parser.update(int(sec_since_boot() * 1e9), False)

          self.assertEqual(0x10, parser.vl['STEERING_IPAS']['SET_ME_X10'])
          self.assertEqual(0x40, parser.vl['STEERING_IPAS']['SET_ME_X40'])

          expected_state = 3 if enabled else 1
          self.assertEqual(expected_state, parser.vl['STEERING_IPAS']['STATE'])

          if steer < 0:
            direction = 3
          elif steer > 0:
            direction = 1
          else:
            direction = 2

          if not enabled:
            direction = 2
          self.assertEqual(direction, parser.vl['STEERING_IPAS']['DIRECTION_CMD'])

          expected_steer = int(round(steer / 1.5)) * 1.5 if enabled else 0
          self.assertAlmostEqual(expected_steer, parser.vl['STEERING_IPAS']['ANGLE'])

    sendcan.close()

  def test_steering(self):
    self.longMessage = True
    car_name = TOYOTA.RAV4

    sendcan = messaging.pub_sock('sendcan')

    params = CarInterface.get_params(car_name)
    CI = CarInterface(params, CarController)

    limit = 1500

    # Get parser
    parser_signals = [
      ('STEER_REQUEST', 'STEERING_LKA', 0),
      ('SET_ME_1', 'STEERING_LKA', 0),
      ('STEER_TORQUE_CMD', 'STEERING_LKA', -1),
      ('LKA_STATE', 'STEERING_LKA', -1),
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
      CI.CS.steer_torque_motor = limit * steer

      # More control applies for the first one because of rate limits
      for _ in range(1000 if steer < -0.99 else 25):
        can_sends = CI.apply(control)
        sendcan.send(can_list_to_can_capnp(can_sends, msgtype='sendcan'))

        parser.update(int(sec_since_boot() * 1e9), False)

      self.assertEqual(1, parser.vl['STEERING_LKA']['SET_ME_1'])
      self.assertEqual(True, parser.vl['STEERING_LKA']['STEER_REQUEST'])
      self.assertAlmostEqual(round(steer * limit), parser.vl['STEERING_LKA']['STEER_TORQUE_CMD'])
      self.assertEqual(0, parser.vl['STEERING_LKA']['LKA_STATE'])

    sendcan.close()

  def test_accel(self):
    self.longMessage = True
    car_name = TOYOTA.RAV4

    sendcan = messaging.pub_sock('sendcan')

    params = CarInterface.get_params(car_name)
    CI = CarInterface(params, CarController)

    # Get parser
    parser_signals = [
      ('ACCEL_CMD', 'ACC_CONTROL', 0),
    ]
    parser = CANParser(CI.cp.dbc_name, parser_signals, [], 0, sendcan=True, tcp_addr="127.0.0.1")
    time.sleep(0.2)  # Slow joiner syndrome

    for accel in np.linspace(-3., 1.5, 25):
      control = car.CarControl.new_message()
      actuators = car.CarControl.Actuators.new_message()

      gas = accel / 3. if accel > 0. else 0.
      brake = -accel / 3. if accel < 0. else 0.

      actuators.gas = float(gas)
      actuators.brake = float(brake)
      control.enabled = True
      control.actuators = actuators
      CI.update(control)

      # More control applies for the first one because of rate limits
      for _ in range(25):
        can_sends = CI.apply(control)
        sendcan.send(can_list_to_can_capnp(can_sends, msgtype='sendcan'))

      for _ in range(5):
        parser.update(int(sec_since_boot() * 1e9), False)
        time.sleep(0.01)

      min_accel = accel - 0.061
      max_accel = accel + 0.061
      sent_accel = parser.vl['ACC_CONTROL']['ACCEL_CMD']
      accel_ok = min_accel <= sent_accel <= max_accel
      self.assertTrue(accel_ok, msg="%.2f <= %.2f <= %.2f" % (min_accel, sent_accel, max_accel))
    sendcan.close()


if __name__ == '__main__':
  unittest.main()
