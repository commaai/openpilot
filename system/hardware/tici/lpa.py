from openpilot.system.hardware.base import LPABase, Profile


class TiciLPA(LPABase):
  def __init__(self):
    pass

  def list_profiles(self) -> list[Profile]:
    return []

  def get_active_profile(self) -> Profile | None:
    return None

  def delete_profile(self, iccid: str) -> None:
    return None

  def download_profile(self, qr: str, nickname: str | None = None) -> None:
    return None

  def nickname_profile(self, iccid: str, nickname: str) -> None:
    return None

  def switch_profile(self, iccid: str) -> None:
    return None
