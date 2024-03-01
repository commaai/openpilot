#!/usr/bin/env python3
import json
import unittest
from Crypto.PublicKey import RSA
from pathlib import Path
from unittest import mock

from openpilot.common.params import Params
from openpilot.selfdrive.athena.registration import register, UNREGISTERED_DONGLE_ID
from openpilot.selfdrive.athena.tests.helpers import MockResponse
from openpilot.system.hardware.hw import Paths


class TestRegistration(unittest.TestCase):
  def setUp(self):
    self.params = Params()

    persist_dir = Path(Paths.persist_root()) / "comma"
    persist_dir.mkdir(parents=True, exist_ok=True)

    self.priv_key = persist_dir / "id_rsa"
    self.pub_key = persist_dir / "id_rsa.pub"
    self.persist_dongle = persist_dir / "dongle_id"

  def _generate_keys(self):
    self.pub_key.touch()
    k = RSA.generate(2048)
    with open(self.priv_key, "wb") as f:
      f.write(k.export_key())
    with open(self.pub_key, "wb") as f:
      f.write(k.publickey().export_key())

  def _register(self, expected_dongle):
    assert register() == expected_dongle
    assert self.params.get("DongleId", encoding='utf-8') == expected_dongle
    with open(self.persist_dongle) as f:
      assert f.read().strip() == expected_dongle

  def test_valid_cache_persist(self):
    self._generate_keys()
    dongle = "DONGLE_ID_123"
    with open(self.persist_dongle, 'w') as f:
      f.write(dongle)
    with mock.patch("openpilot.selfdrive.athena.registration.api_get", autospec=True) as m:
      self._register(dongle)
      assert not m.called

  def test_valid_cache_params(self):
    self._generate_keys()
    with mock.patch("openpilot.selfdrive.athena.registration.api_get", autospec=True) as m:
      dongle = "DONGLE_ID_123"
      self.params.put("DongleId", dongle)
      self._register(dongle)
      assert not m.called

  def test_no_keys(self):
    # missing pubkey
    with mock.patch("openpilot.selfdrive.athena.registration.api_get", autospec=True) as m:
      self._register(UNREGISTERED_DONGLE_ID)
      assert m.call_count == 0

  def test_missing_cache(self):
    # keys exist but no dongle id
    self._generate_keys()
    with mock.patch("openpilot.selfdrive.athena.registration.api_get", autospec=True) as m:
      dongle = "DONGLE_ID_123"
      m.return_value = MockResponse(json.dumps({'dongle_id': dongle}), 200)
      self._register(dongle)
      assert m.call_count == 1

      # call again, shouldn't hit the API this time
      self._register(dongle)
      assert m.call_count == 1

  def test_unregistered(self):
    # keys exist, but unregistered
    self._generate_keys()
    with mock.patch("openpilot.selfdrive.athena.registration.api_get", autospec=True) as m:
      m.return_value = MockResponse(None, 402)
      self._register(UNREGISTERED_DONGLE_ID)
      assert m.call_count == 1


if __name__ == "__main__":
  unittest.main()
