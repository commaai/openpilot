from enum import IntEnum
import os
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
    t0 = time.monotonic()
    dongle_id = self._params.get("DongleId")
    t1 = time.monotonic()
    print(f"[PRIME_STATE] get dongle_id: {(t1-t0)*1000:.2f}ms")

    if not dongle_id or dongle_id == UNREGISTERED_DONGLE_ID:
      return

    try:
      t2 = time.monotonic()
      identity_token = get_token(dongle_id)
      t3 = time.monotonic()
      print(f"[PRIME_STATE] get_token: {(t3-t2)*1000:.2f}ms")

      t4 = time.monotonic()
      response = api_get(f"v1.1/devices/{dongle_id}", timeout=self.API_TIMEOUT, access_token=identity_token)
      return
      t5 = time.monotonic()
      print(f"[PRIME_STATE] api_get: {(t5-t4)*1000:.2f}ms")

      if response.status_code == 200:
        t6 = time.monotonic()
        data = response.json()
        t7 = time.monotonic()
        print(f"[PRIME_STATE] response.json(): {(t7-t6)*1000:.2f}ms")

        is_paired = data.get("is_paired", False)
        prime_type = data.get("prime_type", 0)
        t8 = time.monotonic()
        self.set_type(PrimeType(prime_type) if is_paired else PrimeType.UNPAIRED)
        t9 = time.monotonic()
        print(f"[PRIME_STATE] set_type: {(t9-t8)*1000:.2f}ms")
        print(f"[PRIME_STATE] _fetch_prime_status TOTAL: {(t9-t0)*1000:.2f}ms")
      else:
        t_end = time.monotonic()
        print(f"[PRIME_STATE] _fetch_prime_status TOTAL (non-200): {(t_end-t0)*1000:.2f}ms")
    except Exception as e:
      t_end = time.monotonic()
      print(f"[PRIME_STATE] _fetch_prime_status TOTAL (error): {(t_end-t0)*1000:.2f}ms")
      cloudlog.error(f"Failed to fetch prime status: {e}")

  def set_type(self, prime_type: PrimeType) -> None:
    t0 = time.monotonic()
    lock_acquire_start = time.monotonic()
    with self._lock:
      t1 = time.monotonic()
      print(f"[PRIME_STATE] set_type lock acquire: {(t1-lock_acquire_start)*1000:.2f}ms")
      if prime_type != self.prime_type:
        self.prime_type = prime_type
        t2 = time.monotonic()
        self._params.put("PrimeType", int(prime_type))
        t3 = time.monotonic()
        print(f"[PRIME_STATE] set_type params.put: {(t3-t2)*1000:.2f}ms")
        cloudlog.info(f"Prime type updated to {prime_type}")
      t4 = time.monotonic()
      print(f"[PRIME_STATE] set_type lock held: {(t4-t1)*1000:.2f}ms")
    t5 = time.monotonic()
    print(f"[PRIME_STATE] set_type TOTAL: {(t5-t0)*1000:.2f}ms")

  def _worker_thread(self) -> None:
    from openpilot.selfdrive.ui.ui_state import ui_state, device
    while self._running:
      t0 = time.monotonic()
      if not ui_state.started and device._awake:
        print('[PRIME_STATE] FETCH PRIME STATUS')
        self._fetch_prime_status()
        t1 = time.monotonic()
        print(f'[PRIME_STATE] worker loop iteration: {(t1-t0)*1000:.2f}ms')

      for _ in range(int(self.FETCH_INTERVAL / self.SLEEP_INTERVAL)):
        if not self._running:
          break
        time.sleep(self.SLEEP_INTERVAL)

  def start(self) -> None:
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
    t0 = time.monotonic()
    with self._lock:
      t1 = time.monotonic()
      result = self.prime_type
      t2 = time.monotonic()
    if t2 - t0 > 0.001:  # Only print if > 1ms
      print(f"[PRIME_STATE] get_type lock wait: {(t1-t0)*1000:.2f}ms, held: {(t2-t1)*1000:.2f}ms")
    return result

  def is_prime(self) -> bool:
    t0 = time.monotonic()
    with self._lock:
      t1 = time.monotonic()
      result = bool(self.prime_type > PrimeType.NONE)
      t2 = time.monotonic()
    if t2 - t0 > 0.001:  # Only print if > 1ms
      print(f"[PRIME_STATE] is_prime lock wait: {(t1-t0)*1000:.2f}ms, held: {(t2-t1)*1000:.2f}ms")
    return result

  def is_paired(self) -> bool:
    t0 = time.monotonic()
    with self._lock:
      t1 = time.monotonic()
      result = self.prime_type > PrimeType.UNPAIRED
      t2 = time.monotonic()
    if t2 - t0 > 0.001:  # Only print if > 1ms
      print(f"[PRIME_STATE] is_paired lock wait: {(t1-t0)*1000:.2f}ms, held: {(t2-t1)*1000:.2f}ms")
    return result

  def __del__(self):
    self.stop()
