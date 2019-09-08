import unittest
import random

from selfdrive.can.tests.packer_old import CANPacker as CANPackerOld
from selfdrive.can.packer import CANPacker
from selfdrive.car.toyota.toyotacan import (
  create_ipas_steer_command, create_steer_command, create_accel_command,
  create_fcw_command, create_ui_command
)
from common.realtime import sec_since_boot


class TestPackerMethods(unittest.TestCase):
  def setUp(self):
    self.cp_old = CANPackerOld("toyota_rav4_hybrid_2017_pt_generated")
    self.cp = CANPacker("toyota_rav4_hybrid_2017_pt_generated")

  def test_correctness(self):
    # Test all commands, randomize the params.
    for _ in xrange(1000):
      # Toyota
      steer = random.randint(-1, 1)
      enabled = (random.randint(0, 2) % 2 == 0)
      apgs_enabled = (random.randint(0, 2) % 2 == 0)
      m_old = create_ipas_steer_command(self.cp_old, steer, enabled, apgs_enabled)
      m = create_ipas_steer_command(self.cp, steer, enabled, apgs_enabled)
      self.assertEqual(m_old, m)

      steer = (random.randint(0, 2) % 2 == 0)
      steer_req = (random.randint(0, 2) % 2 == 0)
      raw_cnt = random.randint(1, 65536)
      m_old = create_steer_command(self.cp_old, steer, steer_req, raw_cnt)
      m = create_steer_command(self.cp, steer, steer_req, raw_cnt)
      self.assertEqual(m_old, m)

      accel = (random.randint(0, 2) % 2 == 0)
      pcm_cancel = (random.randint(0, 2) % 2 == 0)
      standstill_req = (random.randint(0, 2) % 2 == 0)
      lead = (random.randint(0, 2) % 2 == 0)
      m_old = create_accel_command(self.cp_old, accel, pcm_cancel, standstill_req, lead)
      m = create_accel_command(self.cp, accel, pcm_cancel, standstill_req, lead)
      self.assertEqual(m_old, m)

      fcw = random.randint(1, 65536)
      m_old = create_fcw_command(self.cp_old, fcw)
      m = create_fcw_command(self.cp, fcw)
      self.assertEqual(m_old, m)

      steer = (random.randint(0, 2) % 2 == 0)
      left_line = (random.randint(0, 2) % 2 == 0)
      right_line = (random.randint(0, 2) % 2 == 0)
      left_lane_depart = (random.randint(0, 2) % 2 == 0)
      right_lane_depart = (random.randint(0, 2) % 2 == 0)
      m_old = create_ui_command(self.cp_old, steer, left_line, right_line, left_lane_depart, right_lane_depart)
      m = create_ui_command(self.cp, steer, left_line, right_line, left_lane_depart, right_lane_depart)
      self.assertEqual(m_old, m)

  def test_performance(self):
    n1 = sec_since_boot()
    recursions = 100000
    steer = (random.randint(0, 2) % 2 == 0)
    left_line = (random.randint(0, 2) % 2 == 0)
    right_line = (random.randint(0, 2) % 2 == 0)
    left_lane_depart = (random.randint(0, 2) % 2 == 0)
    right_lane_depart = (random.randint(0, 2) % 2 == 0)

    for _ in xrange(recursions):
      create_ui_command(self.cp_old, steer, left_line, right_line, left_lane_depart, right_lane_depart)
    n2 = sec_since_boot()
    elapsed_old = n2 - n1

    # print('Old API, elapsed time: {} secs'.format(elapsed_old))
    n1 = sec_since_boot()
    for _ in xrange(recursions):
      create_ui_command(self.cp, steer, left_line, right_line, left_lane_depart, right_lane_depart)
    n2 = sec_since_boot()
    elapsed_new = n2 - n1
    # print('New API, elapsed time: {} secs'.format(elapsed_new))
    self.assertTrue(elapsed_new < elapsed_old / 2)



if __name__ == "__main__":
  unittest.main()
