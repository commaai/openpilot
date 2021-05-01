#!/usr/bin/env python3
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from common.params import Params
from selfdrive.athena.registration import register

# TODO: mock different hw types

class TestRegistration(unittest.TestCase):

  def setUp(self):
    # clear params and setup key paths
    self.params = Params()
    self.params.clear_all()

    self.persist = tempfile.TemporaryDirectory()
    os.mkdir(os.path.join(self.persist.name, "comma"))
    self.priv_key = Path(os.path.join(self.persist.name, "comma/id_rsa"))
    self.pub_key = Path(os.path.join(self.persist.name, "comma/id_rsa.pub"))
    self.patcher = mock.patch("selfdrive.athena.registration.PERSIST", self.persist.name)

  def tearDown(self):
    self.patcher.stop()
    self.persist.cleanup()

  def test_already_registered(self):
    # if all params are written, return the cached dongle id
    self.params.put("IMEI", "imei")
    self.params.put("HardwareSerial", "serial")
    self.pub_key.touch()

    # TODO: assert api_get not called
    dongle = "DONGLE_ID_123"
    self.params.put("DongleId", dongle)
    self.assertEqual(register(), dongle)

  # PC-specific test
  #def test_unregistered_pc(self):
  #  register()
  #  print(register())

  #@mock.patch("common.api.api_get")
  #def test_handle_40x(self, mock_get):
  #  pass

if __name__ == "__main__":
  unittest.main()
