import time
from functools import lru_cache
from openpilot.common.api import Api
from openpilot.common.time_helpers import system_time_valid

TOKEN_EXPIRY_HOURS = 2


@lru_cache(maxsize=1)
def _get_token(dongle_id: str, t: int):
  if not system_time_valid():
    raise RuntimeError("System time is not valid, cannot generate token")

  return Api(dongle_id).get_token(expiry_hours=TOKEN_EXPIRY_HOURS)


def get_token(dongle_id: str):
  return _get_token(dongle_id, int(time.monotonic() / (TOKEN_EXPIRY_HOURS / 2 * 60 * 60)))
