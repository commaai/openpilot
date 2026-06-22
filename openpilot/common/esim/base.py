from abc import ABC, abstractmethod
from dataclasses import dataclass


class LPAError(RuntimeError):
  pass


class LPAProfileNotFoundError(LPAError):
  pass


@dataclass
class Profile:
  iccid: str
  nickname: str
  enabled: bool
  provider: str

  @property
  def is_comma(self) -> bool:
    return self.provider == 'Webbing' and self.iccid.startswith('8985235')


class LPABase(ABC):
  @abstractmethod
  def list_profiles(self) -> list[Profile]:
    pass

  @abstractmethod
  def get_active_profile(self) -> Profile | None:
    pass

  @abstractmethod
  def delete_profile(self, iccid: str) -> None:
    pass

  @abstractmethod
  def download_profile(self, qr: str, nickname: str | None = None) -> None:
    pass

  @abstractmethod
  def nickname_profile(self, iccid: str, nickname: str) -> None:
    pass

  @abstractmethod
  def switch_profile(self, iccid: str) -> None:
    pass

  def process_notifications(self) -> None:
    pass

  @abstractmethod
  def is_euicc(self) -> bool:
    pass
