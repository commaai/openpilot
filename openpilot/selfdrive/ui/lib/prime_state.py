from enum import IntEnum
import os
import requests
import threading
import time

from openpilot.common.api import api_get
from openpilot.common.params import Params
from openpilot.common.realtime import drop_realtime
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

class Provider(str):
  GOOGLE = "google"
  GITHUB = "github"
  APPLE = "apple"

class PrimeState:
  FETCH_INTERVAL = 5.0  # seconds between API calls
  API_TIMEOUT = 10.0  # seconds for API requests
  SLEEP_INTERVAL = 0.5  # seconds to sleep between checks in the worker thread

  def __init__(self):
    self._params = Params()
    self._lock = threading.Lock()
    self._session = requests.Session()  # reuse session to reduce SSL handshake overhead
    self.prime_type: PrimeType = self._load_initial_state()
    self._prime_trial_available = False
    pairing_provider = os.getenv("PAIRING_PROVIDER") or self._params.get("PairingProvider")
    self._pairing_provider: Provider | None = Provider(pairing_provider) if pairing_provider is not None else None
    self._pairing_email: str | None = self._params.get("PairingEmail")
    self._commacare = False

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
      response = api_get(f"v1.1/devices/{dongle_id}", timeout=self.API_TIMEOUT, access_token=identity_token, session=self._session)
      if response.status_code == 200:
        data = response.json()

        is_paired = data.get("is_paired", False)
        prime_type = data.get("prime_type", 0)
        self.set_type(PrimeType(prime_type) if is_paired else PrimeType.UNPAIRED)
        if not is_paired:
          self.set_provider(None, None)

        prime_trial_available = data.get("trial_claimed") is False and data.get("eligible_features", {}).get("prime", False)
        self.set_prime_trial_available(prime_trial_available)

        commacare = data.get("commacare", False)
        self.set_commacare(commacare)
    except Exception as e:
      cloudlog.error(f"Failed to fetch prime status: {e}")

  def _fetch_pairing_provider(self) -> None:
    dongle_id = self._params.get("DongleId")
    if not dongle_id or dongle_id == UNREGISTERED_DONGLE_ID:
      return

    try:
      identity_token = get_token(dongle_id)
      response = api_get(f"v1/devices/{dongle_id}/owner", timeout=self.API_TIMEOUT, access_token=identity_token, session=self._session)
      if response.status_code == 200:
        data = response.json()
        user_id = data.get("user_id", "")
        provider = Provider(user_id.partition("_")[0])
        email = data.get("email")
        self.set_provider(provider, email)
    except Exception as e:
      cloudlog.error(f"Failed to fetch pairing provider: {e}")

  def set_type(self, prime_type: PrimeType) -> None:
    with self._lock:
      if prime_type != self.prime_type:
        self.prime_type = prime_type
        self._params.put("PrimeType", int(prime_type))
        cloudlog.info(f"Prime type updated to {prime_type}")

  def set_provider(self, provider: Provider | None, email: str | None):
    with self._lock:
      if self.prime_type <= PrimeType.UNPAIRED:
        provider = None
        email = None
      email = email or None # if data.get(email) returns "" instead of None

      if self._pairing_provider != provider:
        self._pairing_provider = provider
        if provider is None:
          self._params.remove("PairingProvider")
        else:
          self._params.put("PairingProvider", str(provider))

      if self._pairing_email != email:
        self._pairing_email = email
        if email is None:
          self._params.remove("PairingEmail")
        else:
          self._params.put("PairingEmail", email)

  def set_commacare(self, has_commacare: bool):
    with self._lock:
      if self.prime_type <= PrimeType.UNPAIRED:
        self._commacare = False
      else:
        self._commacare = has_commacare

  def set_prime_trial_available(self, prime_trail_available: bool):
    with self._lock:
      if self.prime_type <= PrimeType.UNPAIRED:
        self._prime_trial_available = False
      else:
        self._prime_trial_available = prime_trail_available

  def _worker_thread(self) -> None:
    drop_realtime()
    from openpilot.selfdrive.ui.ui_state import ui_state, device
    while self._running:
      if not ui_state.started and device._awake:
        self._fetch_prime_status()
        self._fetch_pairing_provider()

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
    with self._lock:
      return self.prime_type

  def is_prime(self) -> bool:
    with self._lock:
      return bool(self.prime_type > PrimeType.NONE)

  def is_full_prime(self) -> bool:
    with self._lock:
      return self.prime_type > PrimeType.NONE and self.prime_type != PrimeType.LITE

  def is_paired(self) -> bool:
    with self._lock:
      return self.prime_type > PrimeType.UNPAIRED

  def can_claim_prime_trial(self) -> bool:
    with self._lock:
      return self._prime_trial_available

  def has_commacare(self) -> bool:
    with self._lock:
      return self._commacare

  def get_pairing_provider(self) -> str | None:
    with self._lock:
      return self._pairing_provider

  def get_pairing_account(self) -> str:
    with self._lock:
      if not self._pairing_provider:
        return "unknown"
      elif self._pairing_provider == Provider.GITHUB or not self._pairing_email:
        return f"{self._pairing_provider} account"
      return self._pairing_email

  def __del__(self):
    self.stop()
