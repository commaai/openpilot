import time
import threading
from collections.abc import Callable
from dataclasses import replace

from openpilot.common.hardware import HARDWARE
from openpilot.common.swaglog import cloudlog
from openpilot.common.esim.base import LPABase, Profile
from openpilot.common.esim.esim import execute_and_process_notifications


class CellularManager:
  PROFILE_POLL_INTERVAL_S = 5.0

  def __init__(self):
    self._lpa: LPABase | None = None
    self._profiles: list[Profile] = []
    self._busy: bool = False
    # re-probed every poll; SIM may be swapped at runtime on tray-accessible devices
    self._is_euicc: bool | None = None
    self._modem_state: dict = {}

    self._lock = threading.Lock()
    self._callback_lock = threading.Lock()
    self._callback_queue: list[Callable] = []

    self.on_profiles_updated: Callable[[], None] | None = None
    self.on_operation_error: Callable[[str], None] | None = None

    self._last_profile_poll: float = 0.0
    self._polling: bool = False

  @property
  def modem_state(self) -> dict:
    return self._modem_state

  def process_callbacks(self):
    with self._callback_lock:
      to_run, self._callback_queue = self._callback_queue, []
    for cb in to_run:
      cb()

    if not self._busy and not self._polling and time.monotonic() - self._last_profile_poll >= self.PROFILE_POLL_INTERVAL_S:
      self._last_profile_poll = time.monotonic()
      self._modem_state = HARDWARE.get_modem_state()
      self._poll_profiles()

  @property
  def profiles(self) -> list[Profile]:
    return self._profiles

  @property
  def active_profile(self) -> Profile | None:
    return next((p for p in self._profiles if p.enabled), None)

  @property
  def busy(self) -> bool:
    return self._busy

  @property
  def is_euicc(self) -> bool | None:
    return self._is_euicc

  def _ensure_lpa(self) -> LPABase:
    if self._lpa is None:
      self._lpa = HARDWARE.get_sim_lpa()
    return self._lpa

  def _enqueue(self, cb: Callable):
    with self._callback_lock:
      self._callback_queue.append(cb)

  def _stop_polling(self):
    self._polling = False

  def _set_profiles(self, profiles: list[Profile]):
    self._profiles = profiles
    if self.on_profiles_updated:
      self.on_profiles_updated()

  def _finish(self, profiles: list[Profile] | None = None, error: str | None = None):
    self._busy = False
    # defer the next poll a full interval; the eUICC can briefly report stale state after an operation
    self._last_profile_poll = time.monotonic()
    if profiles is not None:
      self._set_profiles(profiles)
    if error is not None:
      self.refresh_profiles()
      if self.on_operation_error:
        self.on_operation_error(error)

  def _run_operation(self, fn: Callable[[LPABase], None], error_msg: str, relist: bool = True):
    self._busy = True

    def worker():
      try:
        with self._lock:
          lpa = self._ensure_lpa()
          fn(lpa)
          profiles = lpa.list_profiles() if relist else None
        self._enqueue(lambda: self._finish(profiles=profiles))
      except Exception as e:
        cloudlog.exception(error_msg)
        err = str(e)
        self._enqueue(lambda: self._finish(error=err))

    threading.Thread(target=worker, daemon=True).start()

  def refresh_profiles(self):
    # next process_callbacks tick polls, respecting busy/polling guards
    self._last_profile_poll = 0.0

  def _poll_profiles(self):
    self._polling = True
    def worker():
      try:
        with self._lock:
          lpa = self._ensure_lpa()
          is_euicc = lpa.is_euicc()
          profiles = lpa.list_profiles() if is_euicc else []
        self._enqueue(lambda: self._finish_poll(is_euicc, profiles))
      except Exception:
        cloudlog.exception("Failed to poll eSIM profiles")
        self._enqueue(self._stop_polling)

    threading.Thread(target=worker, daemon=True).start()

  def _finish_poll(self, is_euicc: bool, profiles: list[Profile]):
    self._polling = False
    if self._busy:
      return
    if not is_euicc and self._is_euicc:
      # is_euicc() is False on any AT error (e.g. SIM busy during a profile refresh); confirm on the next poll
      self._is_euicc = None
      return
    self._is_euicc = is_euicc
    self._set_profiles(profiles)

  def switch_profile(self, iccid: str):
    def switch(lpa: LPABase):
      execute_and_process_notifications(lpa, lambda: lpa.switch_profile(iccid))

    # optimistic: list_profiles() can briefly return stale enabled state after a switch
    self._set_profiles([replace(p, enabled=(p.iccid == iccid)) for p in self._profiles])
    self._run_operation(switch, "Failed to switch eSIM profile", relist=False)

  def delete_profile(self, iccid: str):
    self._run_operation(lambda lpa: lpa.delete_profile(iccid), "Failed to delete eSIM profile")

  def nickname_profile(self, iccid: str, nickname: str):
    self._run_operation(lambda lpa: lpa.nickname_profile(iccid, nickname), "Failed to update eSIM profile nickname")
