#!/usr/bin/env python
"""ROS has a parameter server, we have files.

The parameter store is a persistent key value store, implemented as a directory with a writer lock.
On Android, we store params under params_dir = /data/params. The writer lock is a file
"<params_dir>/.lock" taken using flock(), and data is stored in a directory symlinked to by
"<params_dir>/d".

Each key, value pair is stored as a file with named <key> with contents <value>, located in
  <params_dir>/d/<key>

Readers of a single key can just open("<params_dir>/d/<key>") and read the file contents.
Readers who want a consistent snapshot of multiple keys should take the lock.

Writers should take the lock before modifying anything. Writers should also leave the DB in a
consistent state after a crash. The implementation below does this by copying all params to a temp
directory <params_dir>/<tmp>, then atomically symlinking <params_dir>/<d> to <params_dir>/<tmp>
before deleting the old <params_dir>/<d> directory.

Writers that only modify a single key can simply take the lock, then swap the corresponding value
file in place without messing with <params_dir>/d.
"""
import time
import os
import errno
import sys
import shutil
import fcntl
import tempfile
from enum import Enum

def mkdirs_exists_ok(path):
  try:
    os.makedirs(path)
  except OSError:
    if not os.path.isdir(path):
      raise

class TxType(Enum):
  PERSISTANT = 1
  CLEAR_ON_MANAGER_START = 2
  CLEAR_ON_CAR_START = 3

class UnknownKeyName(Exception):
  pass

keys = {
# written: manager
# read:    loggerd, uploaderd, baseui
  "DongleId": TxType.PERSISTANT,
  "AccessToken": TxType.PERSISTANT,
  "Version": TxType.PERSISTANT,
  "GitCommit": TxType.PERSISTANT,
  "GitBranch": TxType.PERSISTANT,
# written: baseui
# read:    ui, controls
  "IsMetric": TxType.PERSISTANT,
  "IsRearViewMirror": TxType.PERSISTANT,
# written: visiond
# read:    visiond, controlsd
  "CalibrationParams": TxType.PERSISTANT,
# written: visiond
# read:    visiond, ui
  "CloudCalibration": TxType.PERSISTANT,
# written: controlsd
# read:    radard
  "CarParams": TxType.CLEAR_ON_CAR_START}


class FileLock(object):
  def __init__(self, path, create):
    self._path = path
    self._create = create
    self._fd = None

  def acquire(self):
    self._fd = os.open(self._path, os.O_CREAT if self._create else 0)
    fcntl.flock(self._fd, fcntl.LOCK_EX)

  def release(self):
    if self._fd is not None:
      os.close(self._fd)
      self._fd = None


class DBAccessor(object):
  def __init__(self, path):
    self._path = path
    self._vals = None

  def keys(self):
    self._check_entered()
    return self._vals.keys()

  def get(self, key):
    self._check_entered()
    try:
      return self._vals[key]
    except KeyError:
      return None

  def _get_lock(self, create):
    lock = FileLock(os.path.join(self._path, ".lock"), create)
    lock.acquire()
    return lock

  def _read_values_locked(self):
    """Callers should hold a lock while calling this method."""
    vals = {}
    try:
      data_path = self._data_path()
      keys = os.listdir(data_path)
      for key in keys:
        with open(os.path.join(data_path, key), "rb") as f:
          vals[key] = f.read()
    except (OSError, IOError) as e:
      # Either the DB hasn't been created yet, or somebody wrote a bug and left the DB in an
      # inconsistent state. Either way, return empty.
      if e.errno == errno.ENOENT:
        return {}

    return vals

  def _data_path(self):
    return os.path.join(self._path, "d")

  def _check_entered(self):
    if self._vals is None:
      raise Exception("Must call __enter__ before using DB")


class DBReader(DBAccessor):
  def __enter__(self):
    try:
      lock = self._get_lock(False)
    except OSError as e:
      # Do not create lock if it does not exist.
      if e.errno == errno.ENOENT:
        self._vals = {}
        return self

    try:
      # Read everything.
      self._vals = self._read_values_locked()
      return self
    finally:
      lock.release()

  def __exit__(self, type, value, traceback): pass


class DBWriter(DBAccessor):
  def __init__(self, path):
    super(DBWriter, self).__init__(path)
    self._lock = None
    self._prev_umask = None

  def put(self, key, value):
    self._vals[key] = value

  def delete(self, key):
    self._vals.pop(key, None)

  def __enter__(self):
    mkdirs_exists_ok(self._path)

    # Make sure we can write and that permissions are correct.
    self._prev_umask = os.umask(0)

    try:
      os.chmod(self._path, 0o777)
      self._lock = self._get_lock(True)
      self._vals = self._read_values_locked()
    except:
      os.umask(self._prev_umask)
      self._prev_umask = None
      raise

    return self

  def __exit__(self, type, value, traceback):
    self._check_entered()

    try:
      old_data_path = None
      new_data_path = None
      tempdir_path = tempfile.mkdtemp(prefix=".tmp", dir=self._path)
      try:
        # Write back all keys.
        os.chmod(tempdir_path, 0o777)
        for k, v in self._vals.items():
          with open(os.path.join(tempdir_path, k), "wb") as f:
            f.write(v)

        data_path = self._data_path()
        try:
          old_data_path = os.path.join(self._path, os.readlink(data_path))
        except (OSError, IOError) as e:
          # NOTE(mgraczyk): If other DB implementations have bugs, this could cause
          #                 copies to be left behind, but we still want to overwrite.
          pass

        new_data_path = "{}.link".format(tempdir_path)
        os.symlink(os.path.basename(tempdir_path), new_data_path)
        os.rename(new_data_path, data_path)
      # TODO(mgraczyk): raise useful error when values are bad.
      except:
        shutil.rmtree(tempdir_path)
        if new_data_path is not None:
          os.remove(new_data_path)
        raise

      # Keep holding the lock while we clean up the old data.
      if old_data_path is not None:
        shutil.rmtree(old_data_path)
    finally:
      os.umask(self._prev_umask)
      self._prev_umask = None

      # Always release the lock.
      self._lock.release()
      self._lock = None



class JSDB(object):
  def __init__(self, fn):
    self._fn = fn

  def begin(self, write=False):
    if write:
      return DBWriter(self._fn)
    else:
      return DBReader(self._fn)

class Params(object):
  def __init__(self, db='/data/params'):
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

  # Test multiprocess:
  # seq 0 100000 | xargs -P20 -I{} python common/params.py DongleId {} && sleep 0.05
  # while python common/params.py DongleId; do sleep 0.05; done
