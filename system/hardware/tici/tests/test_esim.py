import pytest

from openpilot.system.hardware import TICI
from openpilot.system.hardware.tici.esim import LPA2, LPAProfileNotFoundError


class TestEsim:
  """
  https://euicc-manual.osmocom.org/docs/rsp/known-test-profile
  """
  TEST_ACTIVATION_CODE = 'LPA:1$rsp.truphone.com$QRF-BETTERROAMING-PMRDGIR2EARDEIT5'
  TEST_ICCID = '8944476500001944011'
  TEST_NICKNAME = 'test_profile'

  @classmethod
  def setup_class(cls):
    if not TICI:
      pytest.skip()

  def setup_method(self):
    self._cleanup()

  def teardown_method(self):
    self._cleanup()

  def test_list_profiles(self):
    lpa = LPA2()
    profiles = lpa.list_profiles()
    assert profiles is not None

  def test_download_profile(self):
    lpa = LPA2()
    lpa.download_profile(self.TEST_ACTIVATION_CODE, self.TEST_NICKNAME)
    assert self._profile_exists(self.TEST_ICCID, self.TEST_NICKNAME)

    self.enable_profile(lpa)
    self.disable_profile(lpa)

  def enable_profile(self, lpa: LPA2):
    lpa.enable_profile(self.TEST_ICCID)
    current = lpa.get_active_profile()
    assert current is not None
    assert current['iccid'] == self.TEST_ICCID

  def disable_profile(self, lpa: LPA2):
    lpa.disable_profile(self.TEST_ICCID)
    current = lpa.get_active_profile()
    assert current is None

  def _cleanup(self):
    lpa = LPA2()
    try:
      lpa.delete_profile(self.TEST_ICCID)
    except LPAProfileNotFoundError:
      pass
    assert not self._profile_exists(self.TEST_ICCID, self.TEST_NICKNAME)
    lpa = LPA2()
    lpa.process_notifications()
    assert len(lpa.list_notifications()) == 0

  def _profile_exists(self, iccid: str, nickname: str) -> bool:
    lpa = LPA2()
    profiles = lpa.list_profiles()
    return any(p['iccid'] == iccid and p['nickname'] == nickname for p in profiles)
