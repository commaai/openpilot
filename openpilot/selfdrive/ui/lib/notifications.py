import hashlib
import os
import threading
import time
import requests

from openpilot.common.api import api_get
from openpilot.common.hardware import PC
from openpilot.common.hardware.hw import Paths
from openpilot.common.params import Params
from openpilot.common.realtime import drop_realtime
from openpilot.system.athena.registration import UNREGISTERED_DONGLE_ID
from openpilot.selfdrive.ui.lib.api_helpers import get_token

ASSETS_ROOT = os.path.join(Paths.comma_home() if PC else "/data", "notifications")
ASSET_KEYS = ("icon", "image")  # urls, replaced with the path of the downloaded file


class Notifications:
  """Keeps the notifications from the server in a param, with their images downloaded. The comma four gets an "icon", "image" and "menu",
  the 3X only an "image"."""
  FETCH_INTERVAL = 30.0  # seconds between API calls
  API_TIMEOUT = 10.0  # seconds for API requests
  SLEEP_INTERVAL = 0.5  # seconds to sleep between checks in the worker thread

  def __init__(self):
    self._params = Params()
    self._session = requests.Session()  # reuse session to reduce SSL handshake overhead
    self._lock = threading.Lock()  # the worker thread and dismiss() both change the notifications
    self._notifications: list[dict] = self._params.get("Notifications") or []

    self._thread = None

  def _fetch_asset(self, url: str) -> str:
    path = os.path.join(ASSETS_ROOT, hashlib.sha256(url.encode()).hexdigest() + ".png")
    if not os.path.exists(path):
      resp = self._session.get(url, timeout=self.API_TIMEOUT)
      resp.raise_for_status()
      with open(path + ".tmp", "wb") as f:
        f.write(resp.content)
      os.replace(path + ".tmp", path)
    return path

  def _fetch_notifications(self) -> None:
    dongle_id = self._params.get("DongleId")
    if not dongle_id or dongle_id == UNREGISTERED_DONGLE_ID:
      return

    resp = api_get(f"v1/devices/{dongle_id}/notifications", timeout=self.API_TIMEOUT, access_token=get_token(dongle_id), session=self._session)
    resp.raise_for_status()

    os.makedirs(ASSETS_ROOT, exist_ok=True)
    notifications = [n | {key: self._fetch_asset(n[key]) for key in ASSET_KEYS if key in n} for n in resp.json()]
    with self._lock:
      # the ones the user dismissed stay dismissed
      dismissed = {n["id"] for n in self._notifications if n["dismissed"]}
      notifications = [n | {"dismissed": n["id"] in dismissed} for n in notifications]
      if notifications != self._notifications:
        self._set(notifications)

    used = {n[key] for n in notifications for key in ASSET_KEYS if key in n}
    for fn in os.listdir(ASSETS_ROOT):
      if os.path.join(ASSETS_ROOT, fn) not in used:
        os.remove(os.path.join(ASSETS_ROOT, fn))

  def _worker_thread(self) -> None:
    drop_realtime()
    from openpilot.selfdrive.ui.ui_state import ui_state, device
    next_fetch = 0.
    while True:
      # like PrimeState only while offroad and awake. right away on start, when the device wakes up and when it goes offroad
      if ui_state.started or not device._awake:
        next_fetch = 0.
      elif time.monotonic() >= next_fetch:
        next_fetch = time.monotonic() + self.FETCH_INTERVAL
        try:
          self._fetch_notifications()
        except Exception:
          pass  # e.g. the endpoint isn't deployed yet, try again next time
      time.sleep(self.SLEEP_INTERVAL)

  def start(self) -> None:
    if self._thread and self._thread.is_alive():
      return
    self._thread = threading.Thread(target=self._worker_thread, daemon=True)
    self._thread.start()

  def _set(self, notifications: list[dict]) -> None:
    self._notifications = notifications
    self._params.put("Notifications", notifications, block=True)

  @property
  def active(self) -> list[dict]:
    """The ones the user hasn't dismissed."""
    return [n for n in self._notifications if not n["dismissed"]]

  def dismiss(self, notification_id: str) -> None:
    """Dismisses the notification for good."""
    with self._lock:
      self._set([n | {"dismissed": True} if n["id"] == notification_id else n for n in self._notifications])
