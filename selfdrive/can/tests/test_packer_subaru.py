import unittest
import random

from selfdrive.can.tests.packer_old import CANPacker as CANPackerOld
from selfdrive.can.packer import CANPacker
import selfdrive.car.subaru.subarucan as subarucan
from selfdrive.car.subaru.values import CAR as subaru_car


class TestPackerMethods(unittest.TestCase):
  def setUp(self):
    self.subaru_cp_old = CANPackerOld("subaru_global_2017")
    self.subaru_cp = CANPacker("subaru_global_2017")

  def test_correctness(self):
    # Test all cars' commands, randomize the params.
    for _ in range(1000):
      apply_steer = (random.randint(0, 2) % 2 == 0)
      frame = random.randint(1, 65536)
      steer_step = random.randint(1, 65536)
      m_old = subarucan.create_steering_control(self.subaru_cp_old, subaru_car.IMPREZA, apply_steer, frame, steer_step)
      m = subarucan.create_steering_control(self.subaru_cp, subaru_car.IMPREZA, apply_steer, frame, steer_step)
      self.assertEqual(m_old, m)

      m_old = subarucan.create_steering_status(self.subaru_cp_old, subaru_car.IMPREZA, apply_steer, frame, steer_step)
      m = subarucan.create_steering_status(self.subaru_cp, subaru_car.IMPREZA, apply_steer, frame, steer_step)
      self.assertEqual(m_old, m)

      es_distance_msg = {}
      pcm_cancel_cmd = (random.randint(0, 2) % 2 == 0)
      m_old = subarucan.create_es_distance(self.subaru_cp_old, es_distance_msg, pcm_cancel_cmd)
      m = subarucan.create_es_distance(self.subaru_cp, es_distance_msg, pcm_cancel_cmd)
      self.assertEqual(m_old, m)


if __name__ == "__main__":
  unittest.main()
