import time
from functools import lru_cache
from openpilot.common.api import Api, api_get
import threading
from collections.abc import Callable
from openpilot.common.time_helpers import system_time_valid
from openpilot.common.swaglog import cloudlog

TOKEN_EXPIRY_HOURS = 2


@lru_cache(maxsize=1)
def _get_token(dongle_id: str, t: int):
  if not system_time_valid():
    raise RuntimeError("System time is not valid, cannot generate token")

  return Api(dongle_id).get_token(expiry_hours=TOKEN_EXPIRY_HOURS)


def get_token(dongle_id: str):
  return _get_token(dongle_id, int(time.monotonic() / (TOKEN_EXPIRY_HOURS / 2 * 60 * 60)))


class RequestRepeater:
  FETCH_INTERVAL = 5.0  # seconds between API calls
  API_TIMEOUT = 10.0  # seconds for API requests
  SLEEP_INTERVAL = 0.5  # seconds to sleep between checks in the worker thread

  def __init__(self, dongle_id: str, request_route: str, cache_key: str, period: int):
    self._dongle_id = dongle_id
    self._request_route = request_route
    self._cache_key = cache_key
    self._period = period

    self._request_done_callback: Callable[[str, bool], None] | None = None

    self.start()

    self._lock = threading.Lock()
    self._running = False
    self._thread = None
    self._data = None
    self._last_request_time = 0

  def start(self):
    self._running = True
    self._thread = threading.Thread(target=self._run_thread, daemon=True)
    self._thread.start()

  def _run_thread(self):
    while self._running:
      now = time.monotonic()
      if now - self._last_request_time >= self._period:
        self._last_request_time = now
        self._send_request()
      time.sleep(0.5)

  def _send_request(self):
    token = get_token(self._dongle_id)

    try:
      identity_token = get_token(self._dongle_id)
      response = api_get(f"v1.1/devices/{self._dongle_id}", timeout=self.API_TIMEOUT, access_token=identity_token)
      if response.status_code == 200:
        data = response.json()
        is_paired = data.get("is_paired", False)
        prime_type = data.get("prime_type", 0)
        self.set_type(PrimeType(prime_type) if is_paired else PrimeType.UNPAIRED)
    except Exception as e:
      cloudlog.error(f"Failed to fetch prime status: {e}")

    # try:
    #   response = Api(self._dongle_id).request('GET', self._request_route, access_token=token, timeout=5)
    #   success = response.status_code == 200
    #   data = response.text if success else None
    # except Exception as e:
    #   cloudlog.error(f"RequestRepeater request failed: {e}")
    #   success = False
    #   data = None
