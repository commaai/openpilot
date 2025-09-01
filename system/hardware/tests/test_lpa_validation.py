import pytest

from openpilot.system.hardware.base import LPABase, LPAProfileNotFoundError, Profile


class TestLPABase(LPABase):

  def list_profiles(self) -> list[Profile]:
    return []

  def get_active_profile(self) -> Profile | None:
    return None

  def delete_profile(self, iccid: str) -> None:
    pass

  def download_profile(self, qr: str, nickname: str | None = None) -> None:
    pass

  def nickname_profile(self, iccid: str, nickname: str) -> None:
    pass

  def switch_profile(self, iccid: str) -> None:
    pass


class TestLPAValidation:

  def setup_method(self):
    self.lpa = TestLPABase()

  def test_validate_iccid(self):
    self.lpa._validate_iccid('8988303000000614227')

    with pytest.raises(AssertionError, match='invalid ICCID format'):
      self.lpa._validate_iccid('')

    with pytest.raises(AssertionError, match='invalid ICCID format'):
      self.lpa._validate_iccid('1234567890123456789') # Doesn't start with 89

  def test_validate_lpa_activation_code(self):
    self.lpa._validate_lpa_activation_code('LPA:1$rsp.truphone.com$QRF-BETTERROAMING-PMRDGIR2EARDEIT5')

    with pytest.raises(AssertionError, match='invalid LPA activation code format'):
      self.lpa._validate_lpa_activation_code('')

    with pytest.raises(AssertionError, match='invalid LPA activation code format'):
      self.lpa._validate_lpa_activation_code('LPA:1$domain.com') # Missing third part

  def test_validate_nickname(self):
    self.lpa._validate_nickname('test_profile')

    with pytest.raises(AssertionError, match='nickname must be between 1 and 16 characters'):
      self.lpa._validate_nickname('')

    with pytest.raises(AssertionError, match='nickname must contain only alphanumeric characters'):
      self.lpa._validate_nickname('test.profile') # Contains invalid character

  def test_validate_successful(self):
    self.lpa._validate_successful([{'payload': {'message': 'success'}}])

    with pytest.raises(AssertionError, match='expected at least one message'):
      self.lpa._validate_successful([])

    with pytest.raises(AssertionError, match='expected success notification'):
      self.lpa._validate_successful([{'payload': {'message': 'error'}}])

  def test_validate_profile_exists(self, mocker):
    existing_profiles = [Profile(iccid='8988303000000614227', nickname='test1', enabled=True, provider='Test Provider')]

    mocker.patch.object(self.lpa, 'list_profiles', return_value=existing_profiles)
    self.lpa._validate_profile_exists('8988303000000614227')

    mocker.patch.object(self.lpa, 'list_profiles', return_value=[])
    with pytest.raises(LPAProfileNotFoundError, match='profile 8988303000000614227 does not exist'):
      self.lpa._validate_profile_exists('8988303000000614227')

    mocker.patch.object(self.lpa, 'list_profiles', return_value=existing_profiles)
    with pytest.raises(LPAProfileNotFoundError, match='profile 8988303000000614229 does not exist'):
      self.lpa._validate_profile_exists('8988303000000614229')
