import pytest

from openpilot.system.hardware import TICI
from openpilot.system.hardware.tici.esim import LPA2, LPAProfileNotFoundError

# https://euicc-manual.osmocom.org/docs/rsp/known-test-profile
TEST_ACTIVATION_CODE = 'LPA:1$rsp.truphone.com$QRF-BETTERROAMING-PMRDGIR2EARDEIT5'
TEST_ICCID = '8944476500001944011'

TEST_NICKNAME = 'test_profile'

def cleanup(active_profile_iccid: str):
  lpa = LPA2()
  try:
    lpa.delete_profile(TEST_ICCID)
  except LPAProfileNotFoundError:
    pass
  lpa.process_notifications()
  assert len(lpa.list_notifications()) == 0

class TestEsim:

  @classmethod
  def setup_class(cls):
    if not TICI:
      pytest.skip()

    # Save the active profile before running tests
    cls.lpa = LPA2()
    cls.original_profile = cls.lpa.get_active_profile()
    if cls.original_profile:
      cls.original_iccid = cls.original_profile['iccid']
    else:
      cls.original_iccid = None

    # Download test profile once for all tests
    cleanup(TEST_ICCID)
    cls.lpa.download_profile(TEST_ACTIVATION_CODE, TEST_NICKNAME)

    # Verify profile was downloaded
    profiles = cls.lpa.list_profiles()
    cls.test_profile = next((p for p in profiles if p['iccid'] == TEST_ICCID), None)
    assert cls.test_profile is not None
    assert cls.test_profile['nickname'] == TEST_NICKNAME

  @classmethod
  def teardown_class(cls):
    # Restore the original profile if it existed
    if cls.original_iccid:
      try:
        cls.lpa.enable_profile(cls.original_iccid)
      except Exception as e:
        print(f"Failed to restore original profile: {e}")
    cleanup(TEST_ICCID)

  def setup_method(self):
    # Clean up any test profile before each test
    cleanup(TEST_ICCID)

  def test_list_profiles(self):
    profiles = self.lpa.list_profiles()
    assert isinstance(profiles, list)
    for profile in profiles:
      assert isinstance(profile, dict)
      assert 'iccid' in profile
      assert 'isdp_aid' in profile
      assert 'nickname' in profile
      assert 'enabled' in profile
      assert 'provider' in profile

  def test_get_active_profile(self):
    active_profile = self.lpa.get_active_profile()
    if active_profile:
      assert isinstance(active_profile, dict)
      assert active_profile['enabled']
      assert 'iccid' in active_profile
      assert 'isdp_aid' in active_profile
      assert 'nickname' in active_profile
      assert 'provider' in active_profile

  def test_list_notifications(self):
    notifications = self.lpa.list_notifications()
    assert isinstance(notifications, list)
    for notification in notifications:
      assert isinstance(notification, dict)
      assert 'sequence_number' in notification
      assert 'profile_management_operation' in notification
      assert 'notification_address' in notification
      assert 'iccid' in notification

  def test_enable_profile(self):
    # Enable the profile
    self.lpa.enable_profile(TEST_ICCID)

    # Verify profile is enabled
    active_profile = self.lpa.get_active_profile()
    assert active_profile is not None
    assert active_profile['iccid'] == TEST_ICCID
    assert active_profile['enabled']

  def test_disable_profile(self):
    # First enable the profile
    self.lpa.enable_profile(TEST_ICCID)

    # Disable the profile
    self.lpa.disable_profile(TEST_ICCID)

    # Verify profile is disabled
    active_profile = self.lpa.get_active_profile()
    assert active_profile is None or active_profile['iccid'] != TEST_ICCID

  def test_delete_profile(self):
    # Delete the profile
    self.lpa.delete_profile(TEST_ICCID)

    # Verify profile is deleted
    profiles = self.lpa.list_profiles()
    assert not any(p['iccid'] == TEST_ICCID for p in profiles)

  def test_nickname_profile(self):
    # Change nickname
    new_nickname = "new_test_nickname"
    self.lpa.nickname_profile(TEST_ICCID, new_nickname)

    # Verify nickname was changed
    profiles = self.lpa.list_profiles()
    test_profile = next((p for p in profiles if p['iccid'] == TEST_ICCID), None)
    assert test_profile is not None
    assert test_profile['nickname'] == new_nickname

  def test_profile_not_found_error(self):
    with pytest.raises(LPAProfileNotFoundError):
      self.lpa.enable_profile("nonexistent_iccid")

    with pytest.raises(LPAProfileNotFoundError):
      self.lpa.disable_profile("nonexistent_iccid")

    with pytest.raises(LPAProfileNotFoundError):
      self.lpa.delete_profile("nonexistent_iccid")

    with pytest.raises(LPAProfileNotFoundError):
      self.lpa.nickname_profile("nonexistent_iccid", "test")
