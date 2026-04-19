import subprocess
import time
import threading
from collections.abc import Callable

from openpilot.common.swaglog import cloudlog
from openpilot.system.hardware.base import LPABase, Profile


def profile_display_name(profile: Profile) -> str:
  return profile.nickname or profile.provider or profile.iccid[:12]


MODEM_IP_POLL_INTERVAL = 5.0
PROFILE_POLL_INTERVAL = 30.0
SWITCH_SETTLE_S = 15.0


def _get_modem_ip() -> str:
  for iface in ("ppp0", "wwan0"):
    try:
      out = subprocess.check_output(["ip", "-4", "-o", "addr", "show", iface], timeout=1, text=True, stderr=subprocess.DEVNULL)
      parts = out.split()
      for i, part in enumerate(parts):
        if part == "inet" and i + 1 < len(parts):
          return parts[i + 1].split("/")[0]
    except Exception:
      pass
  return ""


def _get_lpa() -> LPABase:
  from openpilot.system.hardware import HARDWARE
  return HARDWARE.get_sim_lpa()


def _get_sim_info() -> dict:
  from openpilot.system.hardware import HARDWARE
  try:
    return HARDWARE.get_sim_info()
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
    self._sim_info: dict = {}

    self._lock = threading.Lock()
    self._callback_queue: list[Callable] = []

    self._profiles_updated_cbs: list[Callable[[list[Profile]], None]] = []
    self._operation_error_cbs: list[Callable[[str], None]] = []

    self._modem_ip: str = ""
    self._ip_polling: bool = False
    self._last_ip_poll: float = 0.0
    self._last_profile_poll: float = 0.0
    self._no_poll_until: float = 0.0
    self._polling: bool = False
    self._active: bool = False

  def add_callbacks(self, profiles_updated: Callable | None = None, operation_error: Callable | None = None):
    if profiles_updated:
      self._profiles_updated_cbs.append(profiles_updated)
    if operation_error:
      self._operation_error_cbs.append(operation_error)

  def set_active(self, active: bool):
    self._active = active

  @property
  def modem_ip(self) -> str:
    return self._modem_ip

  def process_callbacks(self):
    to_run, self._callback_queue = self._callback_queue, []
    for cb in to_run:
      cb()

    if not self._active:
      return

    now = time.monotonic()
    if now - self._last_ip_poll >= MODEM_IP_POLL_INTERVAL and not self._ip_polling:
      self._last_ip_poll = now
      self._ip_polling = True
      threading.Thread(target=self._poll_modem_ip, daemon=True).start()

    if not self._busy and not self._polling and now >= self._no_poll_until and now - self._last_profile_poll >= PROFILE_POLL_INTERVAL:
      self._last_profile_poll = now
      if self._is_euicc is False:
        # confirmed non-eUICC: don't touch LPA, but keep sim_info fresh for the UI
        self._sim_info = _get_sim_info()
      else:
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

  @property
  def sim_info(self) -> dict:
    return self._sim_info

  def _poll_modem_ip(self):
    ip = _get_modem_ip()
    self._callback_queue.append(lambda: self._finish_ip_poll(ip))

  def _finish_ip_poll(self, ip: str):
    self._ip_polling = False
    self._modem_ip = ip

  def is_comma_profile(self, iccid: str) -> bool:
    return any(p.iccid == iccid and p.provider == 'Webbing' for p in self._profiles)

  def _ensure_lpa(self) -> LPABase:
    if self._lpa is None:
      self._lpa = _get_lpa()
    return self._lpa

  def _finish(self, profiles: list[Profile] | None = None, error: str | None = None):
    self._busy = False
    self._switching_iccid = None
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
    self._poll_profiles(is_refresh=True)

  def _poll_profiles(self, is_refresh: bool = False):
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
            sim_info = _get_sim_info()
            self._callback_queue.append(lambda: self._finish_poll_non_euicc(sim_info))
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

  def _finish_poll_non_euicc(self, sim_info: dict):
    self._polling = False
    self._sim_info = sim_info
    for cb in self._profiles_updated_cbs:
      cb(self._profiles)

  def _finish_poll(self, profiles: list[Profile]):
    self._polling = False
    if self._busy or time.monotonic() < self._no_poll_until:
      return
    self._profiles = profiles
    for cb in self._profiles_updated_cbs:
      cb(profiles)

  def switch_profile(self, iccid: str):
    self._switching_iccid = iccid
    self._busy = True

    def worker():
      try:
        with self._lock:
          lpa = self._ensure_lpa()
          lpa.switch_profile(iccid)
        # optimistic update: flip enabled flags locally
        profiles = [Profile(iccid=p.iccid, nickname=p.nickname, enabled=(p.iccid == iccid), provider=p.provider) for p in self._profiles]
        self._no_poll_until = time.monotonic() + SWITCH_SETTLE_S
        self._callback_queue.append(lambda: self._finish(profiles=profiles))
      except Exception as e:
        cloudlog.exception("Failed to switch eSIM profile")
        err = str(e)
        self._callback_queue.append(lambda: self._finish(error=err))

    threading.Thread(target=worker, daemon=True).start()

  def delete_profile(self, iccid: str):
    self._run_operation(lambda lpa: lpa.delete_profile(iccid), "Failed to delete eSIM profile")

  def nickname_profile(self, iccid: str, nickname: str):
    self._run_operation(lambda lpa: lpa.nickname_profile(iccid, nickname), "Failed to update eSIM profile nickname")
