import pytest

from openpilot.system.hardware import TICI
from openpilot.system.hardware.tici.esim import LPA2, LPAProfileNotFoundError

# https://euicc-manual.osmocom.org/docs/rsp/known-test-profile
TEST_ACTIVATION_CODE = 'LPA:1$rsp.truphone.com$QRF-BETTERROAMING-PMRDGIR2EARDEIT5'
TEST_ICCID = '8944476500001944011'

TEST_NICKNAME = 'test_profile'

def cleanup():
  lpa = LPA2()
  try:
    lpa.delete_profile(TEST_ICCID)
  except LPAProfileNotFoundError:
    pass
  assert lpa.get_active_profile() is None
  lpa.process_notifications()
  assert len(lpa.list_notifications()) == 0

class TestEsim:

  @classmethod
  def setup_class(cls):
    if not TICI:
      pytest.skip()
      return
    cleanup()

  def teardown_class(self):
    cleanup()

  def test_list_profiles(self):
    lpa = LPA2()
    profiles = lpa.list_profiles()
    assert profiles is not None

  def test_download_enable_disable_profile(self):
    lpa = LPA2()
    lpa.download_profile(self.TEST_ACTIVATION_CODE, self.TEST_NICKNAME)
    assert self._profile_exists(lpa, self.TEST_ICCID, self.TEST_NICKNAME)

    self._enable_profile(lpa)
    self._disable_profile(lpa)

  def _enable_profile(self, lpa: LPA2):
    lpa.enable_profile(self.TEST_ICCID)
    current = lpa.get_active_profile()
    assert current is not None
    assert current['iccid'] == self.TEST_ICCID

  def _disable_profile(self, lpa: LPA2):
    lpa.disable_profile(self.TEST_ICCID)
    current = lpa.get_active_profile()
    assert current is None

  def _profile_exists(self, lpa: LPA2, iccid: str, nickname: str) -> bool:
    profiles = lpa.list_profiles()
    return any(p['iccid'] == iccid and p['nickname'] == nickname for p in profiles)
