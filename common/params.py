#!/usr/bin/env python
# ROS has a parameter server, we have a file
import time
from enum import Enum
import os
import sys
import json
import fcntl

class TxType(Enum):
  PERSISTANT = 1
  CLEAR_ON_MANAGER_START = 2
  CLEAR_ON_CAR_START = 3

class UnknownKeyName(Exception):
  pass

keys = {
# written: manager
# read:    loggerd, uploaderd
  "DongleId": TxType.PERSISTANT,
  "DongleSecret": TxType.PERSISTANT,
# written: calibrationd
# read:    calibrationd
  "CalibrationParams": TxType.PERSISTANT,
# written: controlsd
# read:    radard
  "CarParams": TxType.CLEAR_ON_CAR_START}


# use this instead of lmdb
class JSFile(object):
  def __init__(self, fn, write=False):
    self._fn = fn
    self._vals = None
    self._write = write

  def keys(self):
    return self._vals.keys()

  def get(self, key):
    if key in self._vals:
      return self._vals[key]
    else:
      return None

  def put(self, key, value):
    self._vals[key] = value

  def delete(self, key):
    if key in self._vals:
      del self._vals[key]

  def __enter__(self):
    self._f = None
    try:
      self._f = open(self._fn)
      if self._write:
        fcntl.flock(self._f, fcntl.LOCK_EX)
      self._vals = json.loads(self._f.read())
    except Exception:
      self._vals = {}
    return self

  def __exit__(self, type, value, traceback):
    if self._write:
      with open(self._fn+".staging", "w") as f:
        f.write(json.dumps(self._vals))
      os.rename(self._fn+".staging", self._fn)
    if self._f is not None:
      self._f.close()

class JSDB(object):
  def __init__(self, fn):
    self._fn = fn

  def begin(self, write=False):
    return JSFile(self._fn, write)

class Params(object):
  def __init__(self, db='/sdcard/params.json'):
    self.env = JSDB(db)

  def _clear_keys_with_type(self, tx_type):
    with self.env.begin(write=True) as txn:
      for key in keys:
        if keys[key] == tx_type:
          txn.delete(key)

  def manager_start(self):
    self._clear_keys_with_type(TxType.CLEAR_ON_MANAGER_START)

  def car_start(self):
    self._clear_keys_with_type(TxType.CLEAR_ON_CAR_START)

  def get(self, key, block=False):
    if key not in keys:
      raise UnknownKeyName(key)

    while 1:
      with self.env.begin() as txn:
        ret = txn.get(key)
      if not block or ret is not None:
        break
      # is polling really the best we can do?
      time.sleep(0.05)
    return ret

  def put(self, key, dat):
    if key not in keys:
      raise UnknownKeyName(key)

    with self.env.begin(write=True) as txn:
      txn.put(key, dat)
    print "set", key

if __name__ == "__main__":
  params = Params()
  if len(sys.argv) > 2:
    params.put(sys.argv[1], sys.argv[2])
  else:
    for k in keys:
      pp = params.get(k)
      if pp is None:
        print k, "is None"
      elif all(ord(c) < 128 and ord(c) >= 32 for c in pp):
        print k, pp
      else:
        print k, pp.encode("hex")

