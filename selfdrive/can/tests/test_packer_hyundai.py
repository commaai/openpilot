import unittest
import random

from selfdrive.can.tests.packer_old import CANPacker as CANPackerOld
from selfdrive.can.packer import CANPacker
import selfdrive.car.hyundai.hyundaican as hyundaican
from selfdrive.car.hyundai.values import CHECKSUM as hyundai_checksum


class TestPackerMethods(unittest.TestCase):
  def setUp(self):
    self.hyundai_cp_old = CANPackerOld("hyundai_kia_generic")
    self.hyundai_cp = CANPacker("hyundai_kia_generic")

  def test_correctness(self):
    # Test all commands, randomize the params.
    for _ in range(1000):
      # Hyundai
      car_fingerprint = hyundai_checksum["crc8"][0]
      apply_steer = (random.randint(0, 2) % 2 == 0)
      steer_req = (random.randint(0, 2) % 2 == 0)
      cnt = random.randint(0, 65536)
      enabled = (random.randint(0, 2) % 2 == 0)
      lkas11 = {
        "CF_Lkas_LdwsSysState": random.randint(0,65536),
        "CF_Lkas_SysWarning": random.randint(0,65536),
        "CF_Lkas_LdwsLHWarning": random.randint(0,65536),
        "CF_Lkas_LdwsRHWarning": random.randint(0,65536),
        "CF_Lkas_HbaLamp": random.randint(0,65536),
        "CF_Lkas_FcwBasReq": random.randint(0,65536),
        "CF_Lkas_ToiFlt": random.randint(0,65536),
        "CF_Lkas_HbaSysState": random.randint(0,65536),
        "CF_Lkas_FcwOpt": random.randint(0,65536),
        "CF_Lkas_HbaOpt": random.randint(0,65536),
        "CF_Lkas_FcwSysState": random.randint(0,65536),
        "CF_Lkas_FcwCollisionWarning": random.randint(0,65536),
        "CF_Lkas_FusionState": random.randint(0,65536),
        "CF_Lkas_FcwOpt_USM": random.randint(0,65536),
        "CF_Lkas_LdwsOpt_USM": random.randint(0,65536)
      }
      hud_alert = random.randint(0, 65536)
      keep_stock = (random.randint(0, 2) % 2 == 0)
      m_old = hyundaican.create_lkas11(self.hyundai_cp_old, car_fingerprint, apply_steer, steer_req, cnt, enabled,
                                       lkas11, hud_alert, keep_stock)
      m = hyundaican.create_lkas11(self.hyundai_cp, car_fingerprint, apply_steer, steer_req, cnt, enabled,
                                  lkas11, hud_alert, keep_stock)
      self.assertEqual(m_old, m)

      clu11 = {
        "CF_Clu_CruiseSwState": random.randint(0,65536),
        "CF_Clu_CruiseSwMain": random.randint(0,65536),
        "CF_Clu_SldMainSW": random.randint(0,65536),
        "CF_Clu_ParityBit1": random.randint(0,65536),
        "CF_Clu_VanzDecimal": random.randint(0,65536),
        "CF_Clu_Vanz": random.randint(0,65536),
        "CF_Clu_SPEED_UNIT": random.randint(0,65536),
        "CF_Clu_DetentOut": random.randint(0,65536),
        "CF_Clu_RheostatLevel": random.randint(0,65536),
        "CF_Clu_CluInfo": random.randint(0,65536),
        "CF_Clu_AmpInfo": random.randint(0,65536),
        "CF_Clu_AliveCnt1": random.randint(0,65536),
      }
      button = random.randint(0, 65536)
      m_old = hyundaican.create_clu11(self.hyundai_cp_old, clu11, button)
      m = hyundaican.create_clu11(self.hyundai_cp, clu11, button)
      self.assertEqual(m_old, m)


if __name__ == "__main__":
  unittest.main()
