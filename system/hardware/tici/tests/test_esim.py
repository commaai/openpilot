import pytest

from openpilot.system.hardware import TICI
from openpilot.system.hardware.tici.esim import LPA, LPAProfileNotFoundError

# https://euicc-manual.osmocom.org/docs/rsp/known-test-profile
# iccid is always the same for the given activation code
TEST_ACTIVATION_CODE = 'LPA:1$rsp.truphone.com$QRF-BETTERROAMING-PMRDGIR2EARDEIT5'
TEST_ICCID = '8944476500001944011'

TEST_NICKNAME = 'test_profile'

def cleanup():
  lpa = LPA()
  try:
    lpa.delete_profile(TEST_ICCID)
  except LPAProfileNotFoundError:
    pass
  lpa.process_notifications()

class TestEsim:

  @classmethod
  def setup_class(cls):
    if not TICI:
      pytest.skip()
    cleanup()

  @classmethod
  def teardown_class(cls):
    cleanup()

  def test_provision_enable_disable(self):
    lpa = LPA()
    current_active = lpa.get_active_profile()

    lpa.download_profile(TEST_ACTIVATION_CODE, TEST_NICKNAME)
    assert any(p.iccid == TEST_ICCID and p.nickname == TEST_NICKNAME for p in lpa.list_profiles())

    lpa.enable_profile(TEST_ICCID)
    new_active = lpa.get_active_profile()
    assert new_active is not None
    assert new_active.iccid == TEST_ICCID
    assert new_active.nickname == TEST_NICKNAME

    lpa.disable_profile(TEST_ICCID)
    new_active = lpa.get_active_profile()
    assert new_active is None

    if current_active:
      lpa.enable_profile(current_active.iccid)
