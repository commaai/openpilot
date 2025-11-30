from enum import IntEnum
import os
import json

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.ui.lib.api_helpers import RequestRepeater


class PrimeType(IntEnum):
  UNKNOWN = -2
  UNPAIRED = -1
  NONE = 0
  MAGENTA = 1
  LITE = 2
  BLUE = 3
  MAGENTA_NEW = 4
  PURPLE = 5


class PrimeState:
  def __init__(self):
    self._params = Params()
    self.prime_type: PrimeType = self._load_initial_state()

    dongle_id = self._params.get("DongleId")
    self._request_repeater = RequestRepeater(dongle_id, f"v1.1/devices/{dongle_id}", 5, "ApiCache_Device")
    self._request_repeater.add_request_done_callback(self._handle_reply)
    self._request_repeater.load_cache()  # sets prime_type from API response cache

  def _load_initial_state(self) -> PrimeType:
    prime_type_str = os.getenv("PRIME_TYPE") or self._params.get("PrimeType")
    try:
      if prime_type_str is not None:
        return PrimeType(int(prime_type_str))
    except (ValueError, TypeError):
      pass
    return PrimeType.UNKNOWN

  def _handle_reply(self, response: str, success: bool):
    if not success:
      return

    try:
      data = json.loads(response)
      is_paired = data.get("is_paired", False)
      prime_type = data.get("prime_type", 0)
      self.set_type(PrimeType(prime_type) if is_paired else PrimeType.UNPAIRED)
    except Exception as e:
      cloudlog.error(f"Failed to fetch prime status: {e}")

  def set_type(self, prime_type: PrimeType) -> None:
    if prime_type != self.prime_type:
      self.prime_type = prime_type
      self._params.put("PrimeType", int(prime_type))
      cloudlog.info(f"Prime type updated to {prime_type}")

  def start(self) -> None:
    self._request_repeater.start()

  def get_type(self) -> PrimeType:
    return self.prime_type

  def is_prime(self) -> bool:
    return bool(self.prime_type > PrimeType.NONE)

  def is_paired(self) -> bool:
    return self.prime_type > PrimeType.UNPAIRED
