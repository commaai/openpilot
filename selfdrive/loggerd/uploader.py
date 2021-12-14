#!/usr/bin/env python3
import json
import os
import random
import requests
import threading
import time
import traceback
from pathlib import Path

from cereal import log
import cereal.messaging as messaging
from common.api import Api
from common.params import Params
from selfdrive.hardware import TICI
from selfdrive.loggerd.xattr_cache import getxattr, setxattr
from selfdrive.loggerd.config import ROOT
from selfdrive.swaglog import cloudlog

NetworkType = log.DeviceState.NetworkType
UPLOAD_ATTR_NAME = 'user.upload'
UPLOAD_ATTR_VALUE = b'1'

allow_sleep = bool(os.getenv("UPLOADER_SLEEP", "1"))
force_wifi = os.getenv("FORCEWIFI") is not None
fake_upload = os.getenv("FAKEUPLOAD") is not None


def get_directory_sort(d):
  return list(map(lambda s: s.rjust(10, '0'), d.rsplit('--', 1)))

def listdir_by_creation(d):
  try:
    paths = os.listdir(d)
    paths = sorted(paths, key=get_directory_sort)
    return paths
  except OSError:
    cloudlog.exception("listdir_by_creation failed")
    return list()

def clear_locks(root):
  for logname in os.listdir(root):
    path = os.path.join(root, logname)
    try:
      for fname in os.listdir(path):
        if fname.endswith(".lock"):
          os.unlink(os.path.join(path, fname))
    except OSError:
      cloudlog.exception("clear_locks failed")


class Uploader():
  def __init__(self, dongle_id, root):
    self.dongle_id = dongle_id
    self.api = Api(dongle_id)
    self.root = root

    self.upload_thread = None

    self.last_resp = None
    self.last_exc = None

    self.immediate_size = 0
    self.immediate_count = 0

    # stats for last successfully uploaded file
    self.last_time = 0
    self.last_speed = 0
    self.last_filename = ""

    self.immediate_folders = ["crash/", "boot/"]
    self.immediate_priority = {"qlog.bz2": 0, "qcamera.ts": 1}

  def get_upload_sort(self, name):
    if name in self.immediate_priority:
      return self.immediate_priority[name]
    return 1000

  def list_upload_files(self):
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
          is_uploaded = getxattr(fn, UPLOAD_ATTR_NAME)
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

        yield (name, key, fn)

  def next_file_to_upload(self):
    upload_files = list(self.list_upload_files())

    for name, key, fn in upload_files:
      if any(f in fn for f in self.immediate_folders):
        return (key, fn)

    for name, key, fn in upload_files:
      if name in self.immediate_priority:
        return (key, fn)

    return None

  def do_upload(self, key, fn):
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
        cloudlog.debug("*** WARNING, THIS IS A FAKE UPLOAD TO %s ***" % url)

        class FakeResponse():
          def __init__(self):
            self.status_code = 200

        self.last_resp = FakeResponse()
      else:
        with open(fn, "rb") as f:
          self.last_resp = requests.put(url, data=f, headers=headers, timeout=10)
    except Exception as e:
      self.last_exc = (e, traceback.format_exc())
      raise

  def normal_upload(self, key, fn):
    self.last_resp = None
    self.last_exc = None

    try:
      self.do_upload(key, fn)
    except Exception:
      pass

    return self.last_resp

  def upload(self, key, fn):
    try:
      sz = os.path.getsize(fn)
    except OSError:
      cloudlog.exception("upload: getsize failed")
      return False

    cloudlog.event("upload", key=key, fn=fn, sz=sz)

    cloudlog.debug("checking %r with size %r", key, sz)

    if sz == 0:
      try:
        # tag files of 0 size as uploaded
        setxattr(fn, UPLOAD_ATTR_NAME, UPLOAD_ATTR_VALUE)
      except OSError:
        cloudlog.event("uploader_setxattr_failed", exc=self.last_exc, key=key, fn=fn, sz=sz)
      success = True
    else:
      start_time = time.monotonic()
      cloudlog.debug("uploading %r", fn)
      stat = self.normal_upload(key, fn)
      if stat is not None and stat.status_code in (200, 201, 403, 412):
        cloudlog.event("upload_success" if stat.status_code != 412 else "upload_ignored", key=key, fn=fn, sz=sz, debug=True)
        try:
          # tag file as uploaded
          setxattr(fn, UPLOAD_ATTR_NAME, UPLOAD_ATTR_VALUE)
        except OSError:
          cloudlog.event("uploader_setxattr_failed", exc=self.last_exc, key=key, fn=fn, sz=sz)

        self.last_filename = fn
        self.last_time = time.monotonic() - start_time
        self.last_speed = (sz / 1e6) / self.last_time
        success = True
      else:
        cloudlog.event("upload_failed", stat=stat, exc=self.last_exc, key=key, fn=fn, sz=sz, debug=True)
        success = False

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

def uploader_fn(exit_event):
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

    key, fn = d

    cloudlog.debug("upload %r over %s", d, network_type)
    success = uploader.upload(key, fn)
    if success:
      backoff = 0.1
    elif allow_sleep:
      cloudlog.info("upload backoff %r", backoff)
      time.sleep(backoff + random.uniform(0, backoff))
      backoff = min(backoff*2, 120)

    pm.send("uploaderState", uploader.get_msg())
    cloudlog.info("upload done, success=%r", success)

def main():
  uploader_fn(threading.Event())


if __name__ == "__main__":
  main()
