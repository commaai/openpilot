#!/usr/bin/env python3
import json
import os
import tempfile
import unittest
from Crypto.PublicKey import RSA
from pathlib import Path
from unittest import mock

from common.params import Params
from selfdrive.athena.registration import register, UNREGISTERED_DONGLE_ID
from selfdrive.athena.tests.helpers import MockResponse


class TestRegistration(unittest.TestCase):

  def setUp(self):
    # clear params and setup key paths
    self.params = Params()
    self.params.clear_all()

    self.persist = tempfile.TemporaryDirectory()
    os.mkdir(os.path.join(self.persist.name, "comma"))
    self.priv_key = Path(os.path.join(self.persist.name, "comma/id_rsa"))
    self.pub_key = Path(os.path.join(self.persist.name, "comma/id_rsa.pub"))
    self.persist_patcher = mock.patch("selfdrive.athena.registration.PERSIST", self.persist.name)
    self.persist_patcher.start()

  def tearDown(self):
    self.persist_patcher.stop()
    self.persist.cleanup()

  def _generate_keys(self):
    self.pub_key.touch()
    k = RSA.generate(2048)
    with open(self.priv_key, "wb") as f:
      f.write(k.export_key())
    with open(self.pub_key, "wb") as f:
      f.write(k.publickey().export_key())

  def test_valid_cache(self):
    # if all params are written, return the cached dongle id
    self.params.put("IMEI", "imei")
    self.params.put("HardwareSerial", "serial")
    self._generate_keys()

    with mock.patch("selfdrive.athena.registration.api_get", autospec=True) as m:
      dongle = "DONGLE_ID_123"
      self.params.put("DongleId", dongle)
      self.assertEqual(register(), dongle)
      self.assertFalse(m.called)

  def test_missing_cache(self):
    # keys exist but no dongle id
    self._generate_keys()
    with mock.patch("selfdrive.athena.registration.api_get", autospec=True) as m:
      dongle = "DONGLE_ID_123"
      m.return_value = MockResponse(json.dumps({'dongle_id': dongle}), 200)
      self.assertEqual(register(), dongle)
      self.assertEqual(m.call_count, 1)

      # call again, shouldn't hit the API this time
      self.assertEqual(register(), dongle)
      self.assertEqual(m.call_count, 1)
    self.assertEqual(self.params.get("DongleId", encoding='utf-8'), dongle)

  def test_unregistered(self):
    # no keys, no dongle id
    with mock.patch("selfdrive.athena.registration.api_get", autospec=True) as m:
      m.return_value = MockResponse(None, 402)
      dongle = register()
      self.assertEqual(m.call_count, 1)
      self.assertEqual(dongle, UNREGISTERED_DONGLE_ID)
    self.assertEqual(self.params.get("DongleId", encoding='utf-8'), dongle)


if __name__ == "__main__":
  unittest.main()
