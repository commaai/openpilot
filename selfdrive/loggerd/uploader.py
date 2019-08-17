#!/usr/bin/env python
import os
import re
import time
import stat
import json
import random
import ctypes
import inspect
import requests
import traceback
import threading
import subprocess

from collections import Counter
from selfdrive.swaglog import cloudlog
from selfdrive.loggerd.config import ROOT

from common.params import Params
from common.api import Api

fake_upload = os.getenv("FAKEUPLOAD") is not None

def raise_on_thread(t, exctype):
  for ctid, tobj in threading._active.items():
    if tobj is t:
      tid = ctid
      break
  else:
    raise Exception("Could not find thread")

  '''Raises an exception in the threads with id tid'''
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

def listdir_with_creation_date(d):
  lst = os.listdir(d)
  for fn in lst:
    try:
      st = os.stat(os.path.join(d, fn))
      ctime = st[stat.ST_CTIME]
      yield (ctime, fn)
    except OSError:
      cloudlog.exception("listdir_with_creation_date: stat failed?")
      yield (None, fn)

def listdir_by_creation_date(d):
  times_and_paths = list(listdir_with_creation_date(d))
  return [path for _, path in sorted(times_and_paths)]

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
  # ConnectivityManager.getActiveNetworkInfo()
  try:
    result = subprocess.check_output(["service", "call", "connectivity", "2"]).strip().split("\n")
  except subprocess.CalledProcessError:
    return False

  data = ''.join(''.join(w.decode("hex")[::-1] for w in l[14:49].split()) for l in result[1:])

  return "\x00".join("WIFI") in data

def is_on_hotspot():
  try:
    result = subprocess.check_output(["ifconfig", "wlan0"])
    result = re.findall(r"inet addr:((\d+\.){3}\d+)", result)[0][0]

    is_android = result.startswith('192.168.43.')
    is_ios = result.startswith('172.20.10.')
    is_entune = result.startswith('10.0.2.')

    return (is_android or is_ios or is_entune)
  except:
    return False

class Uploader(object):
  def __init__(self, dongle_id, private_key, root):
    self.dongle_id = dongle_id
    self.api = Api(dongle_id, private_key)
    self.root = root

    self.upload_thread = None

    self.last_resp = None
    self.last_exc = None

  def clean_dirs(self):
    try:
      for logname in os.listdir(self.root):
        path = os.path.join(self.root, logname)
        # remove empty directories
        if not os.listdir(path):
          os.rmdir(path)
    except OSError:
      cloudlog.exception("clean_dirs failed")

  def gen_upload_files(self):
    if not os.path.isdir(self.root):
      return
    for logname in listdir_by_creation_date(self.root):
      path = os.path.join(self.root, logname)
      try:
        names = os.listdir(path)
      except OSError:
        continue
      if any(name.endswith(".lock") for name in names):
        continue

      for name in names:
        key = os.path.join(logname, name)
        fn = os.path.join(path, name)

        yield (name, key, fn)

  def get_data_stats(self):
    name_counts = Counter()
    total_size = 0
    for name, key, fn in self.gen_upload_files():
      name_counts[name] += 1
      total_size += os.stat(fn).st_size
    return dict(name_counts), total_size

  def next_file_to_upload(self, with_raw):
    # try to upload qlog files first
    for name, key, fn in self.gen_upload_files():
      if name  == "qlog.bz2":
        return (key, fn, 0)

    if with_raw:
      # then upload log files
      for name, key, fn in self.gen_upload_files():
        if name  == "rlog.bz2":
          return (key, fn, 1)

      # then upload rear and front camera files
      for name, key, fn in self.gen_upload_files():
        if name == "fcamera.hevc":
          return (key, fn, 2)
        elif name == "dcamera.hevc":
          return (key, fn, 3)

      # then upload other files
      for name, key, fn in self.gen_upload_files():
        if not name.endswith('.lock') and not name.endswith(".tmp"):
          return (key, fn, 4)

    return None


  def do_upload(self, key, fn):
    try:
      url_resp = self.api.get("v1.3/"+self.dongle_id+"/upload_url/", timeout=10, path=key, access_token=self.api.get_token())
      url_resp_json = json.loads(url_resp.text)
      url = url_resp_json['url']
      headers = url_resp_json['headers']
      cloudlog.info("upload_url v1.3 %s %s", url, str(headers))

      if fake_upload:
        cloudlog.info("*** WARNING, THIS IS A FAKE UPLOAD TO %s ***" % url)
        class FakeResponse(object):
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
      # can't upload files of 0 size
      os.unlink(fn) # delete the file
      success = True
    else:
      cloudlog.info("uploading %r", fn)
      stat = self.normal_upload(key, fn)
      if stat is not None and stat.status_code in (200, 201):
        cloudlog.event("upload_success", key=key, fn=fn, sz=sz)

        # delete the file
        try:
          os.unlink(fn)
        except OSError:
          cloudlog.event("delete_failed", stat=stat, exc=self.last_exc, key=key, fn=fn, sz=sz)

        success = True
      else:
        cloudlog.event("upload_failed", stat=stat, exc=self.last_exc, key=key, fn=fn, sz=sz)
        success = False

    self.clean_dirs()

    return success



def uploader_fn(exit_event):
  cloudlog.info("uploader_fn")

  params = Params()
  dongle_id = params.get("DongleId")
  private_key = open("/persist/comma/id_rsa").read()

  if dongle_id is None or private_key is None:
    cloudlog.info("uploader missing dongle_id or private_key")
    raise Exception("uploader can't start without dongle id and private key")

  uploader = Uploader(dongle_id, private_key, ROOT)

  backoff = 0.1
  while True:
    allow_raw_upload = (params.get("IsUploadRawEnabled") != "0")
    allow_cellular = (params.get("IsUploadVideoOverCellularEnabled") != "0")
    on_hotspot = is_on_hotspot()
    on_wifi = is_on_wifi()
    should_upload = allow_cellular or (on_wifi and not on_hotspot)

    if exit_event.is_set():
      return

    d = uploader.next_file_to_upload(with_raw=allow_raw_upload and should_upload)
    if d is None:
      time.sleep(5)
      continue

    key, fn, _ = d

    cloudlog.event("uploader_netcheck", allow_cellular=allow_cellular, is_on_hotspot=on_hotspot, is_on_wifi=on_wifi)
    cloudlog.info("to upload %r", d)
    success = uploader.upload(key, fn)
    if success:
      backoff = 0.1
    else:
      cloudlog.info("backoff %r", backoff)
      time.sleep(backoff + random.uniform(0, backoff))
      backoff = min(backoff*2, 120)
    cloudlog.info("upload done, success=%r", success)

def main(gctx=None):
  uploader_fn(threading.Event())

if __name__ == "__main__":
  main()
