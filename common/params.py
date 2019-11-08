#!/usr/bin/env python3
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
import threading
from enum import Enum


def mkdirs_exists_ok(path):
  try:
    os.makedirs(path)
  except OSError:
    if not os.path.isdir(path):
      raise


class TxType(Enum):
  PERSISTENT = 1
  CLEAR_ON_MANAGER_START = 2
  CLEAR_ON_PANDA_DISCONNECT = 3


class UnknownKeyName(Exception):
  pass


keys = {
  "AccessToken": [TxType.PERSISTENT],
  "AthenadPid": [TxType.PERSISTENT],
  "CalibrationParams": [TxType.PERSISTENT],
  "CarParams": [TxType.CLEAR_ON_MANAGER_START, TxType.CLEAR_ON_PANDA_DISCONNECT],
  "CarVin": [TxType.CLEAR_ON_MANAGER_START, TxType.CLEAR_ON_PANDA_DISCONNECT],
  "CompletedTrainingVersion": [TxType.PERSISTENT],
  "ControlsParams": [TxType.PERSISTENT],
  "DoUninstall": [TxType.CLEAR_ON_MANAGER_START],
  "DongleId": [TxType.PERSISTENT],
  "GitBranch": [TxType.PERSISTENT],
  "GitCommit": [TxType.PERSISTENT],
  "GitRemote": [TxType.PERSISTENT],
  "GithubSshKeys": [TxType.PERSISTENT],
  "HasAcceptedTerms": [TxType.PERSISTENT],
  "HasCompletedSetup": [TxType.PERSISTENT],
  "IsGeofenceEnabled": [TxType.PERSISTENT],
  "IsMetric": [TxType.PERSISTENT],
  "IsRHD": [TxType.PERSISTENT],
  "IsUpdateAvailable": [TxType.PERSISTENT],
  "IsUploadRawEnabled": [TxType.PERSISTENT],
  "IsUploadVideoOverCellularEnabled": [TxType.PERSISTENT],
  "LastUpdateTime": [TxType.PERSISTENT],
  "LimitSetSpeed": [TxType.PERSISTENT],
  "LimitSetSpeedNeural": [TxType.PERSISTENT],
  "LiveParameters": [TxType.PERSISTENT],
  "LongitudinalControl": [TxType.PERSISTENT],
  "OpenpilotEnabledToggle": [TxType.PERSISTENT],
  "PandaFirmware": [TxType.CLEAR_ON_MANAGER_START, TxType.CLEAR_ON_PANDA_DISCONNECT],
  "PandaDongleId": [TxType.CLEAR_ON_MANAGER_START, TxType.CLEAR_ON_PANDA_DISCONNECT],
  "Passive": [TxType.PERSISTENT],
  "RecordFront": [TxType.PERSISTENT],
  "ReleaseNotes": [TxType.PERSISTENT],
  "ShouldDoUpdate": [TxType.CLEAR_ON_MANAGER_START],
  "SpeedLimitOffset": [TxType.PERSISTENT],
  "SubscriberInfo": [TxType.PERSISTENT],
  "TermsVersion": [TxType.PERSISTENT],
  "TrainingVersion": [TxType.PERSISTENT],
  "UpdateAvailable": [TxType.CLEAR_ON_MANAGER_START],
  "Version": [TxType.PERSISTENT],
  "Offroad_ChargeDisabled": [TxType.CLEAR_ON_MANAGER_START, TxType.CLEAR_ON_PANDA_DISCONNECT],
  "Offroad_ConnectivityNeeded": [TxType.CLEAR_ON_MANAGER_START],
  "Offroad_ConnectivityNeededPrompt": [TxType.CLEAR_ON_MANAGER_START],
  "Offroad_TemperatureTooHigh": [TxType.CLEAR_ON_MANAGER_START],
  "Offroad_PandaFirmwareMismatch": [TxType.CLEAR_ON_MANAGER_START, TxType.CLEAR_ON_PANDA_DISCONNECT],
  "Offroad_InvalidTime": [TxType.CLEAR_ON_MANAGER_START],
}


def fsync_dir(path):
  fd = os.open(path, os.O_RDONLY)
  try:
    os.fsync(fd)
  finally:
    os.close(fd)


class FileLock():
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


class DBAccessor():
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
      # data_path refers to the externally used path to the params. It is a symlink.
      # old_data_path is the path currently pointed to by data_path.
      # tempdir_path is a path where the new params will go, which the new data path will point to.
      # new_data_path is a temporary symlink that will atomically overwrite data_path.
      #
      # The current situation is:
      #   data_path -> old_data_path
      # We're going to write params data to tempdir_path
      #   tempdir_path -> params data
      # Then point new_data_path to tempdir_path
      #   new_data_path -> tempdir_path
      # Then atomically overwrite data_path with new_data_path
      #   data_path -> tempdir_path
      old_data_path = None
      new_data_path = None
      tempdir_path = tempfile.mkdtemp(prefix=".tmp", dir=self._path)

      try:
        # Write back all keys.
        os.chmod(tempdir_path, 0o777)
        for k, v in self._vals.items():
          with open(os.path.join(tempdir_path, k), "wb") as f:
            f.write(v)
            f.flush()
            os.fsync(f.fileno())
        fsync_dir(tempdir_path)

        data_path = self._data_path()
        try:
          old_data_path = os.path.join(self._path, os.readlink(data_path))
        except (OSError, IOError):
          # NOTE(mgraczyk): If other DB implementations have bugs, this could cause
          #                 copies to be left behind, but we still want to overwrite.
          pass

        new_data_path = "{}.link".format(tempdir_path)
        os.symlink(os.path.basename(tempdir_path), new_data_path)
        os.rename(new_data_path, data_path)
        fsync_dir(self._path)
      finally:
        # If the rename worked, we can delete the old data. Otherwise delete the new one.
        success = new_data_path is not None and os.path.exists(data_path) and (
          os.readlink(data_path) == os.path.basename(tempdir_path))

        if success:
          if old_data_path is not None:
            shutil.rmtree(old_data_path)
        else:
          shutil.rmtree(tempdir_path)

        # Regardless of what happened above, there should be no link at new_data_path.
        if new_data_path is not None and os.path.islink(new_data_path):
          os.remove(new_data_path)
    finally:
      os.umask(self._prev_umask)
      self._prev_umask = None

      # Always release the lock.
      self._lock.release()
      self._lock = None


def read_db(params_path, key):
  path = "%s/d/%s" % (params_path, key)
  try:
    with open(path, "rb") as f:
      return f.read()
  except IOError:
    return None

def write_db(params_path, key, value):
  if isinstance(value, str):
    value = value.encode('utf8')

  prev_umask = os.umask(0)
  lock = FileLock(params_path+"/.lock", True)
  lock.acquire()

  try:
    tmp_path = tempfile.mktemp(prefix=".tmp", dir=params_path)
    with open(tmp_path, "wb") as f:
      f.write(value)
      f.flush()
      os.fsync(f.fileno())

    path = "%s/d/%s" % (params_path, key)
    os.rename(tmp_path, path)
    fsync_dir(os.path.dirname(path))
  finally:
    os.umask(prev_umask)
    lock.release()

class Params():
  def __init__(self, db='/data/params'):
    self.db = db

    # create the database if it doesn't exist...
    if not os.path.exists(self.db+"/d"):
      with self.transaction(write=True):
        pass

  def transaction(self, write=False):
    if write:
      return DBWriter(self.db)
    else:
      return DBReader(self.db)

  def _clear_keys_with_type(self, tx_type):
    with self.transaction(write=True) as txn:
      for key in keys:
        if tx_type in keys[key]:
          txn.delete(key)

  def manager_start(self):
    self._clear_keys_with_type(TxType.CLEAR_ON_MANAGER_START)

  def panda_disconnect(self):
    self._clear_keys_with_type(TxType.CLEAR_ON_PANDA_DISCONNECT)

  def delete(self, key):
    with self.transaction(write=True) as txn:
      txn.delete(key)

  def get(self, key, block=False, encoding=None):
    if key not in keys:
      raise UnknownKeyName(key)

    while 1:
      ret = read_db(self.db, key)
      if not block or ret is not None:
        break
      # is polling really the best we can do?
      time.sleep(0.05)

    if ret is not None and encoding is not None:
      ret = ret.decode(encoding)

    return ret

  def put(self, key, dat):
    """
    Warning: This function blocks until the param is written to disk!
    In very rare cases this can take over a second, and your code will hang.

    Use the put_nonblocking helper function in time sensitive code, but
    in general try to avoid writing params as much as possible.
    """

    if key not in keys:
      raise UnknownKeyName(key)

    write_db(self.db, key, dat)


def put_nonblocking(key, val):
  def f(key, val):
    params = Params()
    params.put(key, val)

  t = threading.Thread(target=f, args=(key, val))
  t.start()
  return t


if __name__ == "__main__":
  params = Params()
  if len(sys.argv) > 2:
    params.put(sys.argv[1], sys.argv[2])
  else:
    for k in keys:
      pp = params.get(k)
      if pp is None:
        print("%s is None" % k)
      elif all(ord(c) < 128 and ord(c) >= 32 for c in pp):
        print("%s = %s" % (k, pp))
      else:
        print("%s = %s" % (k, pp.encode("hex")))

  # Test multiprocess:
  # seq 0 100000 | xargs -P20 -I{} python common/params.py DongleId {} && sleep 0.05
  # while python common/params.py DongleId; do sleep 0.05; done
