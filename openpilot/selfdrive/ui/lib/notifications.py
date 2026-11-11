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
from openpilot.common.swaglog import cloudlog
from openpilot.system.athena.registration import UNREGISTERED_DONGLE_ID
from openpilot.selfdrive.ui.lib.api_helpers import get_token

ASSETS_ROOT = os.path.join(Paths.comma_home() if PC else "/data", "notifications")
ASSET_KEYS = ("icon", "image")  # sha256 of a png, replaced with the path of the saved file


class Notifications:
  FETCH_INTERVAL = 30.0  # seconds between API calls
  API_TIMEOUT = 10.0  # seconds for API requests
  SLEEP_INTERVAL = 0.5  # seconds between checks for waking up or going offroad

  def __init__(self):
    self._params = Params()
    self._session = requests.Session()
    self._lock = threading.Lock()
    self._notifications: list[dict] = self._params.get("Notifications") or []

    self._thread = None

  def _save_asset(self, dongle_id: str, sha256: str) -> str:
    path = os.path.join(ASSETS_ROOT, sha256 + ".png")
    if not os.path.exists(path):
      resp = api_get(f"v1/devices/{dongle_id}/notifications/assets/{sha256}", timeout=self.API_TIMEOUT,
                     access_token=get_token(dongle_id), session=self._session)
      resp.raise_for_status()
      if hashlib.sha256(resp.content).hexdigest() != sha256:
        raise ValueError(f"notification asset {sha256} doesn't match its hash")
      with open(path + ".tmp", "wb") as f:
        f.write(resp.content)
      os.replace(path + ".tmp", path)
    return path

  def _fetch_notifications(self) -> None:
    dongle_id = self._params.get("DongleId")
    if not dongle_id or dongle_id == UNREGISTERED_DONGLE_ID:
      return

    # 304 if it matches the ETag of the last one. images are by sha256, only the ones not saved yet are downloaded
    resp = api_get(f"v1/devices/{dongle_id}/notifications", timeout=self.API_TIMEOUT, access_token=get_token(dongle_id), session=self._session)
    if resp.status_code == 304:
      return
    resp.raise_for_status()

    os.makedirs(ASSETS_ROOT, exist_ok=True)
    notifications = [n | {key: self._save_asset(dongle_id, n[key]) for key in ASSET_KEYS if key in n} for n in resp.json()]
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

    # only once it's all saved, so a failure above fetches it again next time
    if "ETag" in resp.headers:
      self._session.headers["If-None-Match"] = resp.headers["ETag"]

  def _worker_thread(self) -> None:
    drop_realtime()
    from openpilot.selfdrive.ui.ui_state import ui_state, device
    while True:
      if not ui_state.started and device.awake:
        try:
          self._fetch_notifications()
        except Exception as e:
          cloudlog.error(f"Failed to fetch notifications: {e}")
        time.sleep(self.FETCH_INTERVAL)
      else:
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
