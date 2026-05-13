import functools
import re

from openpilot.tools.lib.auth_config import get_token
from openpilot.tools.lib.api import CommaApi
from openpilot.tools.lib.helpers import RE


@functools.total_ordering
class Bootlog:
  def __init__(self, url: str):
    self._url = url

    r = re.search(RE.BOOTLOG_NAME, url)
    if not r:
      raise Exception(f"Unable to parse: {url}")

    self._id = r.group('log_id')
    self._dongle_id = r.group('dongle_id')

  @property
  def url(self) -> str:
    return self._url

  @property
  def dongle_id(self) -> str:
    return self._dongle_id

  @property
  def id(self) -> str:
    return self._id

  def __str__(self):
    return f"{self._dongle_id}/{self._id}"

  def __eq__(self, b) -> bool:
    if not isinstance(b, Bootlog):
      return False
    return self.id == b.id

  def __lt__(self, b) -> bool:
    if not isinstance(b, Bootlog):
      return False
    return self.id < b.id

def get_bootlog_from_id(bootlog_id: str) -> Bootlog | None:
  # TODO: implement an API endpoint for this
  bl = Bootlog(bootlog_id)
  for b in get_bootlogs(bl.dongle_id):
    if b == bl:
      return b
  return None

def get_bootlogs(dongle_id: str) -> list[Bootlog]:
  api = CommaApi(get_token())
  r = api.get(f'v1/devices/{dongle_id}/bootlogs')
  return [Bootlog(b) for b in r]
