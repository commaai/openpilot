import time
import threading
from collections.abc import Callable
from functools import lru_cache

from openpilot.common.api import Api, api_get
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.common.time_helpers import system_time_valid
from openpilot.system.athena.registration import UNREGISTERED_DONGLE_ID

TOKEN_EXPIRY_HOURS = 2


@lru_cache(maxsize=1)
def _get_token(dongle_id: str, t: int):
  if not system_time_valid():
    raise RuntimeError("System time is not valid, cannot generate token")

  return Api(dongle_id).get_token(expiry_hours=TOKEN_EXPIRY_HOURS)


def get_token(dongle_id: str):
  return _get_token(dongle_id, int(time.monotonic() / (TOKEN_EXPIRY_HOURS / 2 * 60 * 60)))


class RequestRepeater:
  API_TIMEOUT = 10.0  # seconds for API requests
  SLEEP_INTERVAL = 0.5  # seconds to sleep between checks in the worker thread

  def __init__(self, dongle_id: str, request_route: str, period: int, cache_key: str | None = None):
    self._dongle_id = dongle_id
    self._request_route = request_route
    self._period = period  # seconds
    self._cache_key = cache_key

    self._request_done_callbacks: list[Callable[[str, bool], None]] = []
    self._prev_response_text = None
    self._running = False
    self._thread = None
    self._params = Params()

    if self._cache_key is not None:
      # Cache successful responses to params
      def cache_response(response: str, success: bool):
        if success and response != self._prev_response_text:
          self._params.put(self._cache_key, response)
          self._prev_response_text = response

      self.add_request_done_callback(cache_response)

  def add_request_done_callback(self, callback: Callable[[str, bool], None]):
    self._request_done_callbacks.append(callback)

  def _do_callbacks(self, response_text: str, success: bool):
    for callback in self._request_done_callbacks:
      try:
        callback(response_text, success)
      except Exception as e:
        cloudlog.error(f"RequestRepeater callback error: {e}")

  def load_cache(self):
    # call callbacks with cached response
    if self._cache_key is not None:
      self._prev_response_text = self._params.get(self._cache_key)
      if self._prev_response_text:
        self._do_callbacks(self._prev_response_text, True)

  def start(self):
    if self._thread and self._thread.is_alive():
      return
    self._running = True
    self._thread = threading.Thread(target=self._worker_thread, daemon=True)
    self._thread.start()

  def stop(self):
    self._running = False
    if self._thread and self._thread.is_alive():
      self._thread.join(timeout=1.0)

  def _worker_thread(self):
    # Avoid circular imports
    from openpilot.selfdrive.ui.ui_state import ui_state, device

    while self._running:
      # Don't run when device is asleep or onroad
      if not ui_state.started and device.awake:
        self._send_request()

      for _ in range(int(self._period / self.SLEEP_INTERVAL)):
        if not self._running:
          break
        time.sleep(self.SLEEP_INTERVAL)

  def _send_request(self):
    if not self._dongle_id or self._dongle_id == UNREGISTERED_DONGLE_ID:
      return

    try:
      identity_token = get_token(self._dongle_id)
      response = api_get(self._request_route, timeout=self.API_TIMEOUT, access_token=identity_token)
      self._do_callbacks(response.text, 200 <= response.status_code < 300)

    except Exception as e:
      cloudlog.error(f"Failed to send request to {self._request_route}: {e}")
      self._do_callbacks("", False)

  def __del__(self):
    self.stop()
