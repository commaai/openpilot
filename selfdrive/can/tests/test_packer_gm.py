import unittest
import random

from selfdrive.can.tests.packer_old import CANPacker as CANPackerOld
from selfdrive.can.packer import CANPacker
import selfdrive.car.gm.gmcan as gmcan
from selfdrive.car.gm.interface import CanBus as GMCanBus


class TestPackerMethods(unittest.TestCase):
  def setUp(self):
    self.gm_cp_old = CANPackerOld("gm_global_a_powertrain")
    self.gm_cp = CANPacker("gm_global_a_powertrain")

    self.ct6_cp_old = CANPackerOld("cadillac_ct6_chassis")
    self.ct6_cp = CANPacker("cadillac_ct6_chassis")

  def test_correctness(self):
    # Test all cars' commands, randomize the params.
    for _ in range(1000):
      bus = random.randint(0, 65536)
      apply_steer = (random.randint(0, 2) % 2 == 0)
      idx = random.randint(0, 65536)
      lkas_active = (random.randint(0, 2) % 2 == 0)
      m_old = gmcan.create_steering_control(self.gm_cp_old, bus, apply_steer, idx, lkas_active)
      m = gmcan.create_steering_control(self.gm_cp, bus, apply_steer, idx, lkas_active)
      self.assertEqual(m_old, m)

      canbus = GMCanBus()
      apply_steer = (random.randint(0, 2) % 2 == 0)
      v_ego = random.randint(0, 65536)
      idx = random.randint(0, 65536)
      enabled = (random.randint(0, 2) % 2 == 0)
      m_old = gmcan.create_steering_control_ct6(self.ct6_cp_old, canbus, apply_steer, v_ego, idx, enabled)
      m = gmcan.create_steering_control_ct6(self.ct6_cp, canbus, apply_steer, v_ego, idx, enabled)
      self.assertEqual(m_old, m)

      bus = random.randint(0, 65536)
      throttle = random.randint(0, 65536)
      idx = random.randint(0, 65536)
      acc_engaged = (random.randint(0, 2) % 2 == 0)
      at_full_stop = (random.randint(0, 2) % 2 == 0)
      m_old = gmcan.create_gas_regen_command(self.gm_cp_old, bus, throttle, idx, acc_engaged, at_full_stop)
      m = gmcan.create_gas_regen_command(self.gm_cp, bus, throttle, idx, acc_engaged, at_full_stop)
      self.assertEqual(m_old, m)

      bus = random.randint(0, 65536)
      apply_brake = (random.randint(0, 2) % 2 == 0)
      idx = random.randint(0, 65536)
      near_stop = (random.randint(0, 2) % 2 == 0)
      at_full_stop = (random.randint(0, 2) % 2 == 0)
      m_old = gmcan.create_friction_brake_command(self.ct6_cp_old, bus, apply_brake, idx, near_stop, at_full_stop)
      m = gmcan.create_friction_brake_command(self.ct6_cp, bus, apply_brake, idx, near_stop, at_full_stop)
      self.assertEqual(m_old, m)

      bus = random.randint(0, 65536)
      acc_engaged = (random.randint(0, 2) % 2 == 0)
      target_speed_kph = random.randint(0, 65536)
      lead_car_in_sight = (random.randint(0, 2) % 2 == 0)
      m_old = gmcan.create_acc_dashboard_command(self.gm_cp_old, bus, acc_engaged, target_speed_kph, lead_car_in_sight)
      m = gmcan.create_acc_dashboard_command(self.gm_cp, bus, acc_engaged, target_speed_kph, lead_car_in_sight)
      self.assertEqual(m_old, m)


if __name__ == "__main__":
  unittest.main()
