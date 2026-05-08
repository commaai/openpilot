from enum import IntEnum
import os
import requests
import threading
import time

from openpilot.common.api import api_get
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.system.athena.registration import UNREGISTERED_DONGLE_ID
from openpilot.selfdrive.ui.lib.api_helpers import get_token


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
  FETCH_INTERVAL = 5.0  # seconds between API calls
  API_TIMEOUT = 10.0  # seconds for API requests
  SLEEP_INTERVAL = 0.5  # seconds to sleep between checks in the worker thread

  def __init__(self):
    self._params = Params()
    self._lock = threading.Lock()
    self._session = requests.Session()  # reuse session to reduce SSL handshake overhead
    self.prime_type: PrimeType = self._load_initial_state()

    self._running = False
    self._thread = None

  def _load_initial_state(self) -> PrimeType:
    prime_type_str = os.getenv("PRIME_TYPE") or self._params.get("PrimeType")
    try:
      if prime_type_str is not None:
        return PrimeType(int(prime_type_str))
    except (ValueError, TypeError):
      pass
    return PrimeType.UNKNOWN

  def _fetch_prime_status(self) -> None:
    dongle_id = self._params.get("DongleId")
    if not dongle_id or dongle_id == UNREGISTERED_DONGLE_ID:
      return

    try:
      identity_token = get_token(dongle_id)
      # FIXME: this api_get call (~127ms) caused multi-ms main-thread spikes when
      # this thread was enabled. Most of the time is socket I/O (GIL released), but
      # requests/urllib3/ssl Python internals hold the GIL in 1-5ms chunks between
      # syscalls — long enough to stall the render thread past CPython's 5ms
      # switch_interval. Mitigations to consider before re-enabling:
      #   1. Move the HTTPS to a subprocess (curl) — eliminates GIL impact entirely.
      #   2. Bump FETCH_INTERVAL to 60s+ and trigger on-demand refresh from screens
      #      that need fresh prime state (e.g. settings/pair).
      #   3. Use http.client/socket directly — less Python overhead than requests
      #      but won't fully eliminate GIL stalls in the SSL/header path.
      response = api_get(f"v1.1/devices/{dongle_id}", timeout=self.API_TIMEOUT, access_token=identity_token, session=self._session)
      if response.status_code == 200:
        data = response.json()
        is_paired = data.get("is_paired", False)
        prime_type = data.get("prime_type", 0)
        self.set_type(PrimeType(prime_type) if is_paired else PrimeType.UNPAIRED)
    except Exception as e:
      cloudlog.error(f"Failed to fetch prime status: {e}")

  def set_type(self, prime_type: PrimeType) -> None:
    with self._lock:
      if prime_type != self.prime_type:
        self.prime_type = prime_type
        self._params.put_nonblocking("PrimeType", int(prime_type))
        cloudlog.info(f"Prime type updated to {prime_type}")

  def _worker_thread(self) -> None:
    # Drop inherited RT99 from main thread — background work, never preempt render.
    try:
      os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
    except OSError:
      pass
    from openpilot.selfdrive.ui.ui_state import ui_state, device
    while self._running:
      # Networking gate: `touch /tmp/enable_bg_threads` to allow API calls.
      if os.path.exists('/tmp/enable_bg_threads'):
        if not ui_state.started and device._awake:
          self._fetch_prime_status()

      for _ in range(int(self.FETCH_INTERVAL / self.SLEEP_INTERVAL)):
        if not self._running:
          break
        time.sleep(self.SLEEP_INTERVAL)

  def start(self) -> None:
    return  # disabled: GIL contention from requests/json caused multi-ms main-thread spikes
    if self._thread and self._thread.is_alive():
      return
    self._running = True
    self._thread = threading.Thread(target=self._worker_thread, daemon=True)
    self._thread.start()

  def stop(self) -> None:
    self._running = False
    if self._thread and self._thread.is_alive():
      self._thread.join(timeout=1.0)

  def get_type(self) -> PrimeType:
    with self._lock:
      return self.prime_type

  def is_prime(self) -> bool:
    with self._lock:
      return bool(self.prime_type > PrimeType.NONE)

  def is_paired(self) -> bool:
    with self._lock:
      return self.prime_type > PrimeType.UNPAIRED

  def __del__(self):
    self.stop()
