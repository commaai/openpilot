import os
import unittest
from unittest.mock import Mock, patch

import requests

from openpilot.selfdrive.ui.lib.prime_state import PrimeState, PrimeType


class TestPairingProvider(unittest.TestCase):
  def setUp(self):
    self.enterContext(patch.dict(os.environ, {"PAIRING_PROVIDER": ""}))
    params = self.enterContext(patch("openpilot.selfdrive.ui.lib.prime_state.Params"))
    params.return_value.get.return_value = "0123456789abcdef"
    self.enterContext(patch.object(PrimeState, "_load_initial_state", return_value=PrimeType.UNKNOWN))
    self.enterContext(patch("openpilot.selfdrive.ui.lib.prime_state.get_token", return_value="device-token"))
    self.clock = self.enterContext(patch("openpilot.selfdrive.ui.lib.prime_state.time.monotonic", return_value=100))
    self.api_get = self.enterContext(patch("openpilot.selfdrive.ui.lib.prime_state.api_get"))
    self.state = PrimeState()
    self.addCleanup(self.state.stop)

  def _fetch(self, user_id, prime_type=PrimeType.NONE):
    self.api_get.side_effect = [
      Mock(status_code=200, json=Mock(return_value={"is_paired": True, "prime_type": prime_type})),
      Mock(status_code=200, json=Mock(return_value={"user_id": user_id})),
    ]
    self.state._fetch_prime_status()

  def test_provider_prefixes(self):
    for user_id, expected in (("github_123", "github"), ("google_123", "google"), ("apple_123", "apple"),
                              ("other_123", None), ("github", "github"), (None, None)):
      with self.subTest(user_id=user_id):
        self.state.set_type(PrimeType.UNPAIRED)
        self._fetch(user_id)
        self.assertEqual(self.state.get_pairing_provider(), expected)
        self.api_get.assert_called_with("v1/devices/0123456789abcdef/owner", timeout=self.state.API_TIMEOUT,
                                         access_token="device-token", session=self.state._session)

  def test_provider_refresh_interval(self):
    self._fetch("google_123")
    self.api_get.reset_mock()
    self.api_get.side_effect = None
    self.api_get.return_value = Mock(status_code=200, json=Mock(return_value={"is_paired": True, "prime_type": PrimeType.LITE}))
    self.clock.return_value = 105
    self.state._fetch_prime_status()
    self.api_get.assert_called_once()
    self.assertEqual(self.state.get_type(), PrimeType.LITE)
    self.assertEqual(self.state.get_pairing_provider(), "google")

    self.clock.return_value = 160
    self._fetch("apple_456")
    self.assertEqual(self.state.get_pairing_provider(), "apple")

  def test_unpair_and_repair(self):
    self._fetch("github_123")
    self.api_get.reset_mock()
    self.api_get.side_effect = None
    self.api_get.return_value = Mock(status_code=200, json=Mock(return_value={"is_paired": False}))
    self.state._fetch_prime_status()
    self.api_get.assert_called_once()
    self.assertIsNone(self.state.get_pairing_provider())

    self.clock.return_value = 105
    self._fetch("apple_123")
    self.assertEqual(self.state.get_pairing_provider(), "apple")

  def test_owner_failure_preserves_prime_status(self):
    for response in (Mock(status_code=500), requests.Timeout()):
      with self.subTest(response=response):
        self.state.set_type(PrimeType.UNPAIRED)
        self.api_get.side_effect = [
          Mock(status_code=200, json=Mock(return_value={"is_paired": True, "prime_type": PrimeType.LITE})),
          response,
        ]
        self.state._fetch_prime_status()
        self.assertEqual(self.state.get_type(), PrimeType.LITE)
        self.assertIsNone(self.state.get_pairing_provider())

  def test_missing_owner_clears_provider(self):
    self._fetch("google_123")
    self.clock.return_value = 160
    self.api_get.side_effect = [
      Mock(status_code=200, json=Mock(return_value={"is_paired": True, "prime_type": PrimeType.NONE})),
      Mock(status_code=404),
    ]
    self.state._fetch_prime_status()
    self.assertIsNone(self.state.get_pairing_provider())


class TestPairingProviderOverride(unittest.TestCase):
  def setUp(self):
    self.params = self.enterContext(patch("openpilot.selfdrive.ui.lib.prime_state.Params"))
    self.params.return_value.get.return_value = "UnregisteredDevice"
    self.api_get = self.enterContext(patch("openpilot.selfdrive.ui.lib.prime_state.api_get"))

  def test_desktop_preview(self):
    for provider in ("github", "google", "apple"):
      for prime_type in ("-1", "0", "1", "2"):
        with self.subTest(provider=provider, prime_type=prime_type), \
             patch.dict(os.environ, {"PAIRING_PROVIDER": provider, "PRIME_TYPE": prime_type}):
          state = PrimeState()
          self.addCleanup(state.stop)
          state._fetch_prime_status()
          self.assertEqual(state.get_type(), PrimeType(int(prime_type)))
          self.assertEqual(state.get_pairing_provider(), provider if prime_type != "-1" else None)
    self.api_get.assert_not_called()
    self.params.return_value.put.assert_not_called()

  def test_invalid_provider(self):
    with patch.dict(os.environ, {"PAIRING_PROVIDER": "invalid", "PRIME_TYPE": "0"}):
      state = PrimeState()
      self.addCleanup(state.stop)
      self.assertIsNone(state.get_pairing_provider())

  def test_override_skips_owner_request(self):
    self.params.return_value.get.return_value = "0123456789abcdef"
    self.api_get.return_value = Mock(status_code=200, json=Mock(return_value={"is_paired": True, "prime_type": 0}))
    with patch.dict(os.environ, {"PAIRING_PROVIDER": "google", "PRIME_TYPE": "0"}), \
         patch("openpilot.selfdrive.ui.lib.prime_state.get_token", return_value="device-token"):
      state = PrimeState()
      self.addCleanup(state.stop)
      state._fetch_prime_status()
      self.assertEqual(state.get_pairing_provider(), "google")
      self.api_get.assert_called_once_with("v1.1/devices/0123456789abcdef", timeout=state.API_TIMEOUT,
                                           access_token="device-token", session=state._session)


class TestPrimeTrial(unittest.TestCase):
  def setUp(self):
    self.enterContext(patch.dict(os.environ, {"PAIRING_PROVIDER": "google", "PRIME_TYPE": "0", "PRIME_TRIAL_CLAIMED": ""}))
    params = self.enterContext(patch("openpilot.selfdrive.ui.lib.prime_state.Params"))
    params.return_value.get.return_value = "0123456789abcdef"
    self.enterContext(patch("openpilot.selfdrive.ui.lib.prime_state.get_token", return_value="device-token"))
    self.api_get = self.enterContext(patch("openpilot.selfdrive.ui.lib.prime_state.api_get"))
    self.state = PrimeState()
    self.addCleanup(self.state.stop)

  def test_trial_eligibility(self):
    for prime_type, claimed, eligible, expected in (
      (PrimeType.NONE, False, True, True),
      (PrimeType.NONE, True, True, False),
      (PrimeType.NONE, False, False, False),
      (PrimeType.NONE, None, True, False),
      (PrimeType.LITE, False, True, False),
      (PrimeType.MAGENTA, False, True, False),
      (PrimeType.UNPAIRED, False, True, False),
    ):
      with self.subTest(prime_type=prime_type, claimed=claimed, eligible=eligible):
        self.api_get.return_value = Mock(status_code=200, json=Mock(return_value={
          "is_paired": prime_type != PrimeType.UNPAIRED, "prime_type": prime_type,
          "trial_claimed": claimed, "eligible_features": {"prime": eligible},
        }))
        self.state._fetch_prime_status()
        self.assertEqual(self.state.can_claim_prime_trial(), expected)

  def test_trial_desktop_override(self):
    for claimed, expected in (("0", True), ("1", False), ("invalid", False)):
      with self.subTest(claimed=claimed), patch.dict(os.environ, {"PRIME_TRIAL_CLAIMED": claimed}):
        state = PrimeState()
        self.addCleanup(state.stop)
        self.assertEqual(state.can_claim_prime_trial(), expected)
        state.set_type(PrimeType.UNPAIRED)
        self.assertFalse(state.can_claim_prime_trial())
