import datetime
import functools
import re

from tools.lib.auth_config import get_token
from tools.lib.api import CommaApi
from tools.lib.helpers import RE, timestamp_to_datetime


@functools.total_ordering
class Bootlog:
  def __init__(self, url: str):
    self._url = url

    r = re.search(RE.BOOTLOG_NAME, url)
    if not r:
      raise Exception(f"Unable to parse: {url}")

    self._dongle_id = r.group('dongle_id')
    self._timestamp = r.group('timestamp')

  @property
  def url(self) -> str:
    return self._url

  @property
  def dongle_id(self) -> str:
    return self._dongle_id

  @property
  def timestamp(self) -> str:
    return self._timestamp

  @property
  def datetime(self) -> datetime.datetime:
    return timestamp_to_datetime(self._timestamp)

  def __eq__(self, b) -> bool:
    return self.datetime == b.datetime

  def __lt__(self, b) -> bool:
    return self.datetime < b.datetime


def get_bootlogs(dongle_id: str):
  api = CommaApi(get_token())
  r = api.get(f'v1/devices/{dongle_id}/bootlogs')
  return [Bootlog(b) for b in r]
