#!/usr/bin/env python3
import ctypes
import inspect
import json
import os
import random
import requests
import threading
import time
import traceback

from cereal import log
from common.hardware import HARDWARE
from common.api import Api
from common.params import Params
from selfdrive.loggerd.xattr_cache import getxattr, setxattr
from selfdrive.loggerd.config import ROOT
from selfdrive.swaglog import cloudlog

NetworkType = log.ThermalData.NetworkType
UPLOAD_ATTR_NAME = 'user.upload'
UPLOAD_ATTR_VALUE = b'1'

fake_upload = os.getenv("FAKEUPLOAD") is not None


def raise_on_thread(t, exctype):
  '''Raises an exception in the threads with id tid'''
  for ctid, tobj in threading._active.items():
    if tobj is t:
      tid = ctid
      break
  else:
    raise Exception("Could not find thread")

  if not inspect.isclass(exctype):
    raise TypeError("Only types can be raised (not instances)")

  res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid),
                                                   ctypes.py_object(exctype))
  if res == 0:
    raise ValueError("invalid thread id")
  elif res != 1:
    # "if it returns a number greater than one, you're in trouble,
    # and you should call it again with exc=NULL to revert the effect"
    ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, 0)
    raise SystemError("PyThreadState_SetAsyncExc failed")

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

def is_on_wifi():
  return HARDWARE.get_network_type() == NetworkType.wifi

class Uploader():
  def __init__(self, dongle_id, root):
    self.dongle_id = dongle_id
    self.api = Api(dongle_id)
    self.root = root

    self.upload_thread = None

    self.last_resp = None
    self.last_exc = None

    self.immediate_priority = {"qlog.bz2": 0, "qcamera.ts": 1}
    self.high_priority = {"rlog.bz2": 0, "fcamera.hevc": 1, "dcamera.hevc": 2, "ecamera.hevc": 3}

  def get_upload_sort(self, name):
    if name in self.immediate_priority:
      return self.immediate_priority[name]
    if name in self.high_priority:
      return self.high_priority[name] + 100
    return 1000

  def gen_upload_files(self):
    if not os.path.isdir(self.root):
      return
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
        yield (name, key, fn)

  def next_file_to_upload(self, with_raw):
    upload_files = list(self.gen_upload_files())

    # try to upload qlog files first
    for name, key, fn in upload_files:
      if name in self.immediate_priority:
        return (key, fn)

    if with_raw:
      # then upload the full log files, rear and front camera files
      for name, key, fn in upload_files:
        if name in self.high_priority:
          return (key, fn)

      # then upload other files
      for name, key, fn in upload_files:
        if not name.endswith('.lock') and not name.endswith(".tmp"):
          return (key, fn)

    return None

  def do_upload(self, key, fn):
    try:
      url_resp = self.api.get("v1.3/"+self.dongle_id+"/upload_url/", timeout=10, path=key, access_token=self.api.get_token())
      if url_resp.status_code == 412:
        self.last_resp = url_resp
        return

      url_resp_json = json.loads(url_resp.text)
      url = url_resp_json['url']
      headers = url_resp_json['headers']
      cloudlog.info("upload_url v1.3 %s %s", url, str(headers))

      if fake_upload:
        cloudlog.info("*** WARNING, THIS IS A FAKE UPLOAD TO %s ***" % url)

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

    cloudlog.info("checking %r with size %r", key, sz)

    if sz == 0:
      try:
        # tag files of 0 size as uploaded
        setxattr(fn, UPLOAD_ATTR_NAME, UPLOAD_ATTR_VALUE)
      except OSError:
        cloudlog.event("uploader_setxattr_failed", exc=self.last_exc, key=key, fn=fn, sz=sz)
      success = True
    else:
      cloudlog.info("uploading %r", fn)
      stat = self.normal_upload(key, fn)
      if stat is not None and stat.status_code in (200, 201, 412):
        cloudlog.event("upload_success" if stat.status_code != 412 else "upload_ignored", key=key, fn=fn, sz=sz)
        try:
          # tag file as uploaded
          setxattr(fn, UPLOAD_ATTR_NAME, UPLOAD_ATTR_VALUE)
        except OSError:
          cloudlog.event("uploader_setxattr_failed", exc=self.last_exc, key=key, fn=fn, sz=sz)
        success = True
      else:
        cloudlog.event("upload_failed", stat=stat, exc=self.last_exc, key=key, fn=fn, sz=sz)
        success = False

    return success

def uploader_fn(exit_event):
  cloudlog.info("uploader_fn")

  params = Params()
  dongle_id = params.get("DongleId").decode('utf8')

  if dongle_id is None:
    cloudlog.info("uploader missing dongle_id")
    raise Exception("uploader can't start without dongle id")

  uploader = Uploader(dongle_id, ROOT)

  backoff = 0.1
  counter = 0
  on_wifi = False
  while not exit_event.is_set():
    offroad = params.get("IsOffroad") == b'1'
    allow_raw_upload = (params.get("IsUploadRawEnabled") != b"0") and offroad
    if offroad and counter % 12 == 0:
      on_wifi = is_on_wifi()
    counter += 1

    d = uploader.next_file_to_upload(with_raw=allow_raw_upload and on_wifi and offroad)
    if d is None:  # Nothing to upload
      time.sleep(60 if offroad else 5)
      continue

    key, fn = d

    cloudlog.event("uploader_netcheck", is_on_wifi=on_wifi)
    cloudlog.info("to upload %r", d)
    success = uploader.upload(key, fn)
    if success:
      backoff = 0.1
    else:
      cloudlog.info("backoff %r", backoff)
      time.sleep(backoff + random.uniform(0, backoff))
      backoff = min(backoff*2, 120)
    cloudlog.info("upload done, success=%r", success)

def main():
  uploader_fn(threading.Event())


if __name__ == "__main__":
  main()
