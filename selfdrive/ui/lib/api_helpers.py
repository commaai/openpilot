# TODO: sort
import time
from functools import lru_cache
from openpilot.common.api import Api, api_get
import threading
from collections.abc import Callable
from openpilot.common.time_helpers import system_time_valid
from openpilot.common.swaglog import cloudlog
from openpilot.common.params import Params
from openpilot.selfdrive.ui.ui_state import ui_state, device

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
    self._period = period  # seconds

    self._request_done_callbacks: list[Callable[[str, bool], None]] = []
    self.add_request_done_callback(self._handle_reply)

    self._prev_request_text = None
    self._lock = threading.Lock()
    self._running = False
    self._thread = None
    self._data = None
    self._last_request_time = 0.0
    self._params = Params()

    self.start()

  def add_request_done_callback(self, callback: Callable[[str, bool], None]):
    self._request_done_callbacks.append(callback)

  def _handle_reply(self, response: str, success: bool):
    # Cache successful responses to params
    if success and response != self._prev_request_text:
      self._params.put(self._cache_key, response)
      self._prev_request_text = response

  def start(self):
    self._running = True
    self._thread = threading.Thread(target=self._run_thread, daemon=True)
    self._thread.start()

  def _run_thread(self):
    while self._running:
      now = time.monotonic()
      if now - self._last_request_time >= self._period:
        self._last_request_time = now

        active_request = False   # TODO: this
        if not ui_state.started and device.is_awake() and not active_request:
          print('RUNNING REQUEST')
          self._send_request()
        else:
          print('SKIPPING REQUEST')

      time.sleep(self.SLEEP_INTERVAL)

  def _send_request(self):
    try:
      identity_token = get_token(self._dongle_id)
      response = api_get(f"v1.1/devices/{self._dongle_id}", timeout=self.API_TIMEOUT, access_token=identity_token)

      if response.status_code == 200:
        for callback in self._request_done_callbacks:
          callback(response.text, True)
      else:
        for callback in self._request_done_callbacks:
          callback(response.text, False)

    except Exception as e:
      cloudlog.error(f"Failed to fetch prime status: {e}")
