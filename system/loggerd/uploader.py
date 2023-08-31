#!/usr/bin/env python3
import bz2
import io
import json
import os
import random
import requests
import threading
import time
import traceback
from pathlib import Path
from typing import BinaryIO, Iterator, List, Optional, Tuple, Union

from cereal import log
import cereal.messaging as messaging
from openpilot.common.api import Api
from openpilot.common.params import Params
from openpilot.common.realtime import set_core_affinity
from openpilot.system.hardware import TICI
from openpilot.system.loggerd.xattr_cache import getxattr, setxattr
from openpilot.system.loggerd.config import ROOT
from openpilot.system.swaglog import cloudlog

NetworkType = log.DeviceState.NetworkType
UPLOAD_ATTR_NAME = 'user.upload'
UPLOAD_ATTR_VALUE = b'1'

UPLOAD_QLOG_QCAM_MAX_SIZE = 100 * 1e6  # MB

allow_sleep = bool(os.getenv("UPLOADER_SLEEP", "1"))
force_wifi = os.getenv("FORCEWIFI") is not None
fake_upload = os.getenv("FAKEUPLOAD") is not None


class FakeRequest:
  def __init__(self):
    self.headers = {"Content-Length": "0"}


class FakeResponse:
  def __init__(self):
    self.status_code = 200
    self.request = FakeRequest()


UploadResponse = Union[requests.Response, FakeResponse]

def get_directory_sort(d: str) -> List[str]:
  return [s.rjust(10, '0') for s in d.rsplit('--', 1)]

def listdir_by_creation(d: str) -> List[str]:
  try:
    paths = os.listdir(d)
    paths = sorted(paths, key=get_directory_sort)
    return paths
  except OSError:
    cloudlog.exception("listdir_by_creation failed")
    return list()

def clear_locks(root: str) -> None:
  for logname in os.listdir(root):
    path = os.path.join(root, logname)
    try:
      for fname in os.listdir(path):
        if fname.endswith(".lock"):
          os.unlink(os.path.join(path, fname))
    except OSError:
      cloudlog.exception("clear_locks failed")


class Uploader:
  def __init__(self, dongle_id: str, root: str):
    self.dongle_id = dongle_id
    self.api = Api(dongle_id)
    self.root = root

    self.last_resp: Optional[UploadResponse] = None
    self.last_exc: Optional[Tuple[Exception, str]] = None

    self.immediate_size = 0
    self.immediate_count = 0

    # stats for last successfully uploaded file
    self.last_time = 0.0
    self.last_speed = 0.0
    self.last_filename = ""

    self.immediate_folders = ["crash/", "boot/"]
    self.immediate_priority = {"qlog": 0, "qlog.bz2": 0, "qcamera.ts": 1}

  def get_upload_sort(self, name: str) -> int:
    if name in self.immediate_priority:
      return self.immediate_priority[name]
    return 1000

  def list_upload_files(self) -> Iterator[Tuple[str, str, str]]:
    if not os.path.isdir(self.root):
      return

    self.immediate_size = 0
    self.immediate_count = 0

    for logname in listdir_by_creation(self.root):
      path = os.path.join(self.root, logname)
      try:
        names = os.listdir(path)
      except OSError:
        continue

      if any(name.endswith(".lock") for name in names):
        continue

      for name in sorted(names, key=self.get_upload_sort):
        key = os.path.join(logname, name)
        fn = os.path.join(path, name)
        # skip files already uploaded
        try:
          is_uploaded = getxattr(fn, UPLOAD_ATTR_NAME) == UPLOAD_ATTR_VALUE
        except OSError:
          cloudlog.event("uploader_getxattr_failed", exc=self.last_exc, key=key, fn=fn)
          is_uploaded = True  # deleter could have deleted
        if is_uploaded:
          continue

        try:
          if name in self.immediate_priority:
            self.immediate_count += 1
            self.immediate_size += os.path.getsize(fn)
        except OSError:
          pass

        yield name, key, fn

  def next_file_to_upload(self) -> Optional[Tuple[str, str, str]]:
    upload_files = list(self.list_upload_files())

    for name, key, fn in upload_files:
      if any(f in fn for f in self.immediate_folders):
        return name, key, fn

    for name, key, fn in upload_files:
      if name in self.immediate_priority:
        return name, key, fn

    return None

  def do_upload(self, key: str, fn: str) -> None:
    try:
      url_resp = self.api.get("v1.4/" + self.dongle_id + "/upload_url/", timeout=10, path=key, access_token=self.api.get_token())
      if url_resp.status_code == 412:
        self.last_resp = url_resp
        return

      url_resp_json = json.loads(url_resp.text)
      url = url_resp_json['url']
      headers = url_resp_json['headers']
      cloudlog.debug("upload_url v1.4 %s %s", url, str(headers))

      if fake_upload:
        cloudlog.debug(f"*** WARNING, THIS IS A FAKE UPLOAD TO {url} ***")
        self.last_resp = FakeResponse()
      else:
        with open(fn, "rb") as f:
          data: BinaryIO
          if key.endswith('.bz2') and not fn.endswith('.bz2'):
            compressed = bz2.compress(f.read())
            data = io.BytesIO(compressed)
          else:
            data = f

          self.last_resp = requests.put(url, data=data, headers=headers, timeout=10)
    except Exception as e:
      self.last_exc = (e, traceback.format_exc())
      raise

  def normal_upload(self, key: str, fn: str) -> Optional[UploadResponse]:
    self.last_resp = None
    self.last_exc = None

    try:
      self.do_upload(key, fn)
    except Exception:
      pass

    return self.last_resp

  def upload(self, name: str, key: str, fn: str, network_type: int, metered: bool) -> bool:
    try:
      sz = os.path.getsize(fn)
    except OSError:
      cloudlog.exception("upload: getsize failed")
      return False

    cloudlog.event("upload_start", key=key, fn=fn, sz=sz, network_type=network_type, metered=metered)

    if sz == 0:
      # tag files of 0 size as uploaded
      success = True
    elif name in self.immediate_priority and sz > UPLOAD_QLOG_QCAM_MAX_SIZE:
      cloudlog.event("uploader_too_large", key=key, fn=fn, sz=sz)
      success = True
    else:
      start_time = time.monotonic()
      stat = self.normal_upload(key, fn)
      if stat is not None and stat.status_code in (200, 201, 401, 403, 412):
        self.last_filename = fn
        self.last_time = time.monotonic() - start_time
        if stat.status_code == 412:
          self.last_speed = 0
          cloudlog.event("upload_ignored", key=key, fn=fn, sz=sz, network_type=network_type, metered=metered)
        else:
          content_length = int(stat.request.headers.get("Content-Length", 0))
          self.last_speed = (content_length / 1e6) / self.last_time
          cloudlog.event("upload_success", key=key, fn=fn, sz=sz, content_length=content_length,
                         network_type=network_type, metered=metered, speed=self.last_speed)
        success = True
      else:
        success = False
        cloudlog.event("upload_failed", stat=stat, exc=self.last_exc, key=key, fn=fn, sz=sz, network_type=network_type, metered=metered)

    if success:
      # tag file as uploaded
      try:
        setxattr(fn, UPLOAD_ATTR_NAME, UPLOAD_ATTR_VALUE)
      except OSError:
        cloudlog.event("uploader_setxattr_failed", exc=self.last_exc, key=key, fn=fn, sz=sz)

    return success

  def get_msg(self):
    msg = messaging.new_message("uploaderState")
    us = msg.uploaderState
    us.immediateQueueSize = int(self.immediate_size / 1e6)
    us.immediateQueueCount = self.immediate_count
    us.lastTime = self.last_time
    us.lastSpeed = self.last_speed
    us.lastFilename = self.last_filename
    return msg


def uploader_fn(exit_event: threading.Event) -> None:
  try:
    set_core_affinity([0, 1, 2, 3])
  except Exception:
    cloudlog.exception("failed to set core affinity")

  clear_locks(ROOT)

  params = Params()
  dongle_id = params.get("DongleId", encoding='utf8')

  if dongle_id is None:
    cloudlog.info("uploader missing dongle_id")
    raise Exception("uploader can't start without dongle id")

  if TICI and not Path("/data/media").is_mount():
    cloudlog.warning("NVME not mounted")

  sm = messaging.SubMaster(['deviceState'])
  pm = messaging.PubMaster(['uploaderState'])
  uploader = Uploader(dongle_id, ROOT)

  backoff = 0.1
  while not exit_event.is_set():
    sm.update(0)
    offroad = params.get_bool("IsOffroad")
    network_type = sm['deviceState'].networkType if not force_wifi else NetworkType.wifi
    if network_type == NetworkType.none:
      if allow_sleep:
        time.sleep(60 if offroad else 5)
      continue

    d = uploader.next_file_to_upload()
    if d is None:  # Nothing to upload
      if allow_sleep:
        time.sleep(60 if offroad else 5)
      continue

    name, key, fn = d

    # qlogs and bootlogs need to be compressed before uploading
    if key.endswith(('qlog', 'rlog')) or (key.startswith('boot/') and not key.endswith('.bz2')):
      key += ".bz2"

    success = uploader.upload(name, key, fn, sm['deviceState'].networkType.raw, sm['deviceState'].networkMetered)
    if success:
      backoff = 0.1
    elif allow_sleep:
      cloudlog.info("upload backoff %r", backoff)
      time.sleep(backoff + random.uniform(0, backoff))
      backoff = min(backoff*2, 120)

    pm.send("uploaderState", uploader.get_msg())


def main() -> None:
  uploader_fn(threading.Event())


if __name__ == "__main__":
  main()
