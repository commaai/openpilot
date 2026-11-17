#!/usr/bin/env python3
"""
DashC uploader: sends finished full-quality camera segments to your own Google Drive.

Place at: openpilot/system/loggerd/dashcam_uploader.py

Needs /data/dashcam/google_token.json, created once by dashcam_login.py:
  {"client_id": "...", "client_secret": "...", "refresh_token": "...", "folder_name": "DashC"}

Files uploaded per 60 s segment: fcamera.hevc (road), ecamera.hevc (wide road), dcamera.hevc (driver).
Works on home Wi-Fi, and on a phone hotspot when "Upload on Mobile Hotspot" is on in the Dashcam settings. Segments still being written
are skipped (loggerd keeps a .lock file in the segment folder until it is done).
"""
import json
import os
import time
import traceback

import requests

import openpilot.cereal.messaging as messaging
from openpilot.common.params import Params
from openpilot.common.hardware.hw import Paths
from openpilot.common.swaglog import cloudlog
from openpilot.system.loggerd.uploader import listdir_by_creation
from openpilot.system.loggerd.xattr_cache import getxattr, setxattr

TOKEN_PATH = "/data/dashcam/google_token.json"
TOKEN_URL = "https://oauth2.googleapis.com/token"
DRIVE_FILES = "https://www.googleapis.com/drive/v3/files"
DRIVE_UPLOAD = "https://www.googleapis.com/upload/drive/v3/files"

VIDEO_FILES = ("fcamera.hevc", "ecamera.hevc", "dcamera.hevc")
ACTIVE_PATH = "/tmp/dashcam_uploading"  # exists only while a file is being uploaded
EVENT_TAG = "dashcam_event.json"  # smart event classification written by dashcam_events
ATTR_NAME = "user.gdrive"
ATTR_VALUE = b"1"

CHUNK_SIZE = 8 * 1024 * 1024  # must be a multiple of 256 KiB
MAX_CHUNK_RETRIES = 8
MAX_FILE_ATTEMPTS = 3  # a broken segment must not block the whole queue
QUOTA_COOLDOWN = 3600
ERROR_ATTR = "user.gdrive_error"


class QuotaExceeded(Exception):
  pass


def raise_if_quota(resp):
  """Translate a Drive quota rejection into a signal that pauses the whole uploader."""
  if resp.status_code not in (403, 413):
    return
  try:
    reason = resp.json().get("error", {}).get("errors", [{}])[0].get("reason", "")
  except ValueError:
    reason = ""
  if reason in ("storageQuotaExceeded", "quotaExceeded"):
    raise QuotaExceeded("Google Drive storage quota exceeded")


class DriveClient:
  def __init__(self):
    self.cfg = {}
    self.access_token = None
    self.token_expiry = 0.0
    self.folder_id = None

  def load_config(self) -> bool:
    try:
      with open(TOKEN_PATH) as f:
        self.cfg = json.load(f)
      return all(k in self.cfg for k in ("client_id", "client_secret", "refresh_token"))
    except (OSError, ValueError):
      return False

  def _auth(self) -> dict:
    if self.access_token is None or time.time() > self.token_expiry - 60:
      r = requests.post(TOKEN_URL, data={
        "client_id": self.cfg["client_id"],
        "client_secret": self.cfg["client_secret"],
        "refresh_token": self.cfg["refresh_token"],
        "grant_type": "refresh_token",
      }, timeout=15)
      raise_if_quota(r)
      r.raise_for_status()
      j = r.json()
      self.access_token = j["access_token"]
      self.token_expiry = time.time() + int(j.get("expires_in", 3600))
    return {"Authorization": f"Bearer {self.access_token}"}

  def ensure_folder(self) -> str:
    if self.folder_id:
      return self.folder_id
    name = self.cfg.get("folder_name", "DashC")
    q = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    r = requests.get(DRIVE_FILES, params={"q": q, "fields": "files(id)"}, headers=self._auth(), timeout=15)
    raise_if_quota(r)
    r.raise_for_status()
    files = r.json().get("files", [])
    if files:
      self.folder_id = files[0]["id"]
    else:
      r = requests.post(DRIVE_FILES, headers=self._auth(), timeout=15,
                        json={"name": name, "mimeType": "application/vnd.google-apps.folder"})
      raise_if_quota(r)
      r.raise_for_status()
      self.folder_id = r.json()["id"]
    return self.folder_id

  def upload_file(self, path: str, drive_name: str) -> bool:
    size = os.path.getsize(path)
    if size == 0:
      return True

    meta = {"name": drive_name, "parents": [self.ensure_folder()]}
    r = requests.post(
      DRIVE_UPLOAD + "?uploadType=resumable",
      headers={**self._auth(), "X-Upload-Content-Type": "video/hevc", "X-Upload-Content-Length": str(size)},
      json=meta, timeout=15)
    raise_if_quota(r)
    r.raise_for_status()
    session_url = r.headers["Location"]

    offset = 0
    retries = 0
    with open(path, "rb") as f:
      while offset < size:
        f.seek(offset)
        data = f.read(CHUNK_SIZE)
        end = offset + len(data) - 1
        try:
          resp = requests.put(session_url, data=data, timeout=60, headers={
            "Content-Length": str(len(data)),
            "Content-Range": f"bytes {offset}-{end}/{size}",
          })
          raise_if_quota(resp)
          if resp.status_code in (200, 201):
            return True
          if resp.status_code == 308:  # chunk accepted, more expected
            rng = resp.headers.get("Range")
            offset = int(rng.split("-")[1]) + 1 if rng else 0
            retries = 0
            continue
          raise RuntimeError(f"unexpected status {resp.status_code}: {resp.text[:200]}")
        except Exception:
          retries += 1
          if retries > MAX_CHUNK_RETRIES:
            raise
          time.sleep(min(2 ** retries, 30))
          # ask Drive how much it actually has, then continue from there
          try:
            q = requests.put(session_url, timeout=30, headers={
              "Content-Length": "0", "Content-Range": f"bytes */{size}"})
            raise_if_quota(q)
            if q.status_code in (200, 201):
              return True
            if q.status_code == 308:
              rng = q.headers.get("Range")
              offset = int(rng.split("-")[1]) + 1 if rng else 0
          except Exception:
            pass
    return False


def set_active(active: bool):
  """Create or remove the marker the screen uses to show the upload dot."""
  try:
    if active:
      with open(ACTIVE_PATH, "w") as f:
        f.write("1")
    elif os.path.exists(ACTIVE_PATH):
      os.remove(ACTIVE_PATH)
  except OSError:
    pass


def event_prefix(path: str) -> str:
  """Drive filename prefix for a segment's driver-monitoring event, if any."""
  try:
    with open(os.path.join(path, EVENT_TAG)) as f:
      return json.load(f).get("prefix", "")
  except (OSError, ValueError, AttributeError):
    return ""


def next_file():
  """Oldest finished, not-yet-uploaded video file, or None."""
  root = Paths.log_root()
  for logdir in listdir_by_creation(root):
    path = os.path.join(root, logdir)
    try:
      names = os.listdir(path)
    except OSError:
      continue
    if any(n.endswith(".lock") for n in names):
      continue  # still being recorded
    for name in VIDEO_FILES:
      if name not in names:
        continue
      fn = os.path.join(path, name)
      try:
        if getxattr(fn, ATTR_NAME) == ATTR_VALUE or getxattr(fn, ERROR_ATTR) == ATTR_VALUE:
          continue
      except OSError:
        continue
      yield logdir, name, fn


def main():
  try:
    os.nice(10)  # never compete with recording/encoding
  except OSError:
    pass

  client = DriveClient()
  params = Params()
  sm = messaging.SubMaster(["deviceState"])
  backoff = 5.0
  quota_until = 0.0
  failures: dict[str, int] = {}

  while True:
    sm.update(0)
    ds = sm["deviceState"]

    if time.time() < quota_until:
      time.sleep(60)
      continue

    if not client.load_config():
      time.sleep(30)  # no token yet, run dashcam_login.py
      continue

    allow_metered = params.get_bool("DashcamAllowMetered")  # Dashcam settings: "Upload on Mobile Hotspot"
    if ds.networkType == 0 or (ds.networkMetered and not allow_metered):  # 0 = no network
      time.sleep(15)
      continue

    item = next(next_file(), None)
    if item is None:
      time.sleep(10)
      continue

    logdir, name, fn = item
    drive_name = f"{event_prefix(os.path.dirname(fn))}{logdir}--{name}"
    try:
      cloudlog.event("dashcam_upload_start", fn=fn, sz=os.path.getsize(fn))
      set_active(True)
      if client.upload_file(fn, drive_name):
        setxattr(fn, ATTR_NAME, ATTR_VALUE)
        failures.pop(fn, None)
        cloudlog.event("dashcam_upload_success", fn=fn)
        backoff = 5.0
        continue
    except QuotaExceeded as e:
      quota_until = time.time() + QUOTA_COOLDOWN
      cloudlog.warning(f"dashcam_upload_paused: {e}, retrying in {QUOTA_COOLDOWN // 60} minutes")
    except Exception:
      failures[fn] = failures.get(fn, 0) + 1
      cloudlog.event("dashcam_upload_failed", fn=fn, attempt=failures[fn], exc=traceback.format_exc())
      if failures[fn] >= MAX_FILE_ATTEMPTS:
        setxattr(fn, ERROR_ATTR, ATTR_VALUE)
        cloudlog.warning(f"dashcam_upload_skipped after {MAX_FILE_ATTEMPTS} attempts: {fn}")
    finally:
      set_active(False)

    time.sleep(backoff)
    backoff = min(backoff * 2, 120)


if __name__ == "__main__":
  main()
