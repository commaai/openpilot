import time
from functools import lru_cache
from openpilot.common.api import Api

TOKEN_EXPIRY_HOURS = 2


@lru_cache(maxsize=1)
def _get_token(dongle_id: str, t: int):
  return Api(dongle_id).get_token(expiry_hours=TOKEN_EXPIRY_HOURS)


def get_token(dongle_id: str):
  return _get_token(dongle_id, int(time.monotonic() / (TOKEN_EXPIRY_HOURS / 2 * 60 * 60)))
