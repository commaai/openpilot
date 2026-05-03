import time
import threading
from collections.abc import Callable

from openpilot.common.swaglog import cloudlog
from openpilot.system.hardware.base import LPABase, Profile


PROFILE_POLL_INTERVAL = 30.0


def _get_lpa() -> LPABase:
  from openpilot.system.hardware import HARDWARE
  return HARDWARE.get_sim_lpa()


def _get_modem_state() -> dict:
  from openpilot.system.hardware import HARDWARE
  try:
    return HARDWARE.get_modem_state()
  except Exception:
    return {}


class CellularManager:
  def __init__(self):
    self._lpa: LPABase | None = None
    self._profiles: list[Profile] = []
    self._busy: bool = False
    self._switching_iccid: str | None = None
    # None = not yet checked, True/False = cached result. SIM cannot be swapped
    # without disassembling the device, so we probe once and keep the result.
    self._is_euicc: bool | None = None
    self._modem_state: dict = {}

    self._lock = threading.Lock()
    self._callback_queue: list[Callable] = []

    self._profiles_updated_cbs: list[Callable[[list[Profile]], None]] = []
    self._operation_error_cbs: list[Callable[[str], None]] = []

    self._last_profile_poll: float = 0.0
    self._polling: bool = False

  def add_callbacks(self, profiles_updated: Callable | None = None, operation_error: Callable | None = None):
    if profiles_updated:
      self._profiles_updated_cbs.append(profiles_updated)
    if operation_error:
      self._operation_error_cbs.append(operation_error)

  @property
  def modem_ip(self) -> str:
    return self._modem_state.get("ip_address", "")

  @property
  def modem_state(self) -> dict:
    return self._modem_state

  def process_callbacks(self):
    to_run, self._callback_queue = self._callback_queue, []
    for cb in to_run:
      cb()

    self._modem_state = _get_modem_state()

    if self._switching_iccid and self._modem_state.get("iccid") == self._switching_iccid:
      self._switching_iccid = None

    if not self._busy and not self._polling and time.monotonic() - self._last_profile_poll >= PROFILE_POLL_INTERVAL:
      self._last_profile_poll = time.monotonic()
      if self._is_euicc is not False:
        self._poll_profiles()

  @property
  def profiles(self) -> list[Profile]:
    return self._profiles

  @property
  def busy(self) -> bool:
    return self._busy

  @property
  def switching_iccid(self) -> str | None:
    return self._switching_iccid

  @property
  def is_euicc(self) -> bool | None:
    return self._is_euicc

  def is_comma_profile(self, iccid: str) -> bool:
    return any(p.iccid == iccid and p.is_comma for p in self._profiles)

  def _ensure_lpa(self) -> LPABase:
    if self._lpa is None:
      self._lpa = _get_lpa()
    return self._lpa

  def _finish(self, profiles: list[Profile] | None = None, error: str | None = None):
    self._busy = False
    if profiles is not None:
      self._profiles = profiles
      for cb in self._profiles_updated_cbs:
        cb(profiles)
    if error is not None:
      for cb in self._operation_error_cbs:
        cb(error)

  def _run_operation(self, fn: Callable, error_msg: str):
    self._busy = True

    def worker():
      try:
        with self._lock:
          lpa = self._ensure_lpa()
          fn(lpa)
          lpa.process_notifications()
          profiles = lpa.list_profiles()
        self._callback_queue.append(lambda: self._finish(profiles=profiles))
      except Exception as e:
        cloudlog.exception(error_msg)
        err = str(e)
        self._callback_queue.append(lambda: self._finish(error=err))

    threading.Thread(target=worker, daemon=True).start()

  def refresh_profiles(self):
    if self._is_euicc is False:
      return
    self._poll_profiles()

  def _poll_profiles(self):
    self._polling = True

    def worker():
      try:
        with self._lock:
          lpa = self._ensure_lpa()
          if self._is_euicc is None:
            cloudlog.info("eSIM: checking eUICC presence")
            self._is_euicc = lpa.is_euicc()
            cloudlog.info(f"eSIM: is_euicc={self._is_euicc}")
          if not self._is_euicc:
            self._callback_queue.append(lambda: setattr(self, '_polling', False))
            return
          cloudlog.info("eSIM: processing notifications")
          lpa.process_notifications()
          cloudlog.info("eSIM: listing profiles")
          profiles = lpa.list_profiles()
          cloudlog.info(f"eSIM: got {len(profiles)} profiles")
        self._callback_queue.append(lambda: self._finish_poll(profiles))
      except Exception:
        cloudlog.exception("Failed to poll eSIM profiles")
        self._callback_queue.append(lambda: setattr(self, '_polling', False))

    threading.Thread(target=worker, daemon=True).start()

  def _finish_poll(self, profiles: list[Profile]):
    self._polling = False
    if self._busy:
      return
    self._profiles = profiles
    for cb in self._profiles_updated_cbs:
      cb(profiles)

  def switch_profile(self, iccid: str):
    # _switching_iccid stays set across _finish; cleared when modem state's ICCID matches (or on error)
    self._switching_iccid = iccid
    self._busy = True

    def worker():
      try:
        with self._lock:
          lpa = self._ensure_lpa()
          lpa.switch_profile(iccid)
        # optimistic update: flip enabled flags locally
        profiles = [Profile(iccid=p.iccid, nickname=p.nickname, enabled=(p.iccid == iccid), provider=p.provider) for p in self._profiles]
        self._callback_queue.append(lambda: self._finish(profiles=profiles))
      except Exception as e:
        cloudlog.exception("Failed to switch eSIM profile")
        err = str(e)
        def fail():
          self._switching_iccid = None
          self._finish(error=err)
        self._callback_queue.append(fail)

    threading.Thread(target=worker, daemon=True).start()

  def delete_profile(self, iccid: str):
    self._run_operation(lambda lpa: lpa.delete_profile(iccid), "Failed to delete eSIM profile")

  def nickname_profile(self, iccid: str, nickname: str):
    self._run_operation(lambda lpa: lpa.nickname_profile(iccid, nickname), "Failed to update eSIM profile nickname")
