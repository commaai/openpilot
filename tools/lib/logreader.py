import os
import sys
import gzip
import zlib
import json
import bz2
import tempfile
import requests
import subprocess
from aenum import Enum
import capnp
import numpy as np

import platform

from tools.lib.exceptions import DataUnreadableError
try:
  from xx.chffr.lib.filereader import FileReader
except ImportError:
  from tools.lib.filereader import FileReader
from tools.lib.log_util import convert_old_pkt_to_new
from cereal import log as capnp_log

OP_PATH = os.path.dirname(os.path.dirname(capnp_log.__file__))

def index_log(fn):
  index_log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "index_log")
  index_log = os.path.join(index_log_dir, "index_log")
  phonelibs_dir = os.path.join(OP_PATH, 'phonelibs')

  subprocess.check_call(["make", "PHONELIBS=" + phonelibs_dir], cwd=index_log_dir, stdout=subprocess.DEVNULL)

  try:
    dat = subprocess.check_output([index_log, fn, "-"])
  except subprocess.CalledProcessError:
    raise DataUnreadableError("%s capnp is corrupted/truncated" % fn)
  return np.frombuffer(dat, dtype=np.uint64)

def event_read_multiple_bytes(dat):
  with tempfile.NamedTemporaryFile() as dat_f:
    dat_f.write(dat)
    dat_f.flush()
    idx = index_log(dat_f.name)

  end_idx = np.uint64(len(dat))
  idx = np.append(idx, end_idx)

  return [capnp_log.Event.from_bytes(dat[idx[i]:idx[i+1]])
          for i in range(len(idx)-1)]


# this is an iterator itself, and uses private variables from LogReader
class MultiLogIterator(object):
  def __init__(self, log_paths, wraparound=True):
    self._log_paths = log_paths
    self._wraparound = wraparound

    self._first_log_idx = next(i for i in range(len(log_paths)) if log_paths[i] is not None)
    self._current_log = self._first_log_idx
    self._idx = 0
    self._log_readers = [None]*len(log_paths)
    self.start_time = self._log_reader(self._first_log_idx)._ts[0]

  def _log_reader(self, i):
    if self._log_readers[i] is None and self._log_paths[i] is not None:
      log_path = self._log_paths[i]
      print("LogReader:%s" % log_path)
      self._log_readers[i] = LogReader(log_path)

    return self._log_readers[i]

  def __iter__(self):
    return self

  def _inc(self):
    lr = self._log_reader(self._current_log)
    if self._idx < len(lr._ents)-1:
      self._idx += 1
    else:
      self._idx = 0
      self._current_log = next(i for i in range(self._current_log + 1, len(self._log_readers) + 1) if i == len(self._log_readers) or self._log_paths[i] is not None)
      # wraparound
      if self._current_log == len(self._log_readers):
        if self._wraparound:
          self._current_log = self._first_log_idx
        else:
          raise StopIteration

  def __next__(self):
    while 1:
      lr = self._log_reader(self._current_log)
      ret = lr._ents[self._idx]
      if lr._do_conversion:
        ret = convert_old_pkt_to_new(ret, lr.data_version)
      self._inc()
      return ret

  def tell(self):
    # returns seconds from start of log
    return (self._log_reader(self._current_log)._ts[self._idx] - self.start_time) * 1e-9

  def seek(self, ts):
    # seek to nearest minute
    minute = int(ts/60)
    if minute >= len(self._log_paths) or self._log_paths[minute] is None:
      return False

    self._current_log = minute

    # HACK: O(n) seek afterward
    self._idx = 0
    while self.tell() < ts:
      self._inc()
    return True


class LogReader(object):
  def __init__(self, fn, canonicalize=True, only_union_types=False):
    _, ext = os.path.splitext(fn)
    data_version = None

    with FileReader(fn) as f:
      dat = f.read()

    # decompress file
    if ext == ".gz" and ("log_" in fn or "log2" in fn):
      dat = zlib.decompress(dat, zlib.MAX_WBITS|32)
    elif ext == ".bz2":
      dat = bz2.decompress(dat)
    elif ext == ".7z":
      if platform.system() == "Darwin":
        os.environ["LA_LIBRARY_FILEPATH"] = "/usr/local/opt/libarchive/lib/libarchive.dylib"
      import libarchive.public
      with libarchive.public.memory_reader(dat) as aa:
        mdat = []
        for it in aa:
          for bb in it.get_blocks():
            mdat.append(bb)
      dat = ''.join(mdat)

    # TODO: extension shouln't be a proxy for DeviceType
    if ext == "":
      if dat[0] == "[":
        needs_conversion = True
        ents = [json.loads(x) for x in dat.strip().split("\n")[:-1]]
        if "_" in fn:
          data_version = fn.split("_")[1]
      else:
        # old rlogs weren't bz2 compressed
        needs_conversion = False
        ents = event_read_multiple_bytes(dat)
    elif ext == ".gz":
      if "log_" in fn:
        # Zero data file.
        ents = [json.loads(x) for x in dat.strip().split("\n")[:-1]]
        needs_conversion = True
      elif "log2" in fn:
        needs_conversion = False
        ents = event_read_multiple_bytes(dat)
      else:
        raise Exception("unknown extension")
    elif ext == ".bz2":
      needs_conversion = False
      ents = event_read_multiple_bytes(dat)
    elif ext == ".7z":
      needs_conversion = True
      ents = [json.loads(x) for x in dat.strip().split("\n")]
    else:
      raise Exception("unknown extension")

    if needs_conversion:
      # TODO: should we call convert_old_pkt_to_new to generate this?
      self._ts = [x[0][0]*1e9 for x in ents]
    else:
      self._ts = [x.logMonoTime for x in ents]

    self.data_version = data_version
    self._do_conversion = needs_conversion and canonicalize
    self._only_union_types = only_union_types
    self._ents = ents

  def __iter__(self):
    for ent in self._ents:
      if self._do_conversion:
        yield convert_old_pkt_to_new(ent, self.data_version)
      elif self._only_union_types:
        try:
          ent.which()
          yield ent
        except capnp.lib.capnp.KjException:
          pass
      else:
        yield ent

def load_many_logs_canonical(log_paths):
  """Load all logs for a sequence of log paths."""
  for log_path in log_paths:
    for msg in LogReader(log_path):
      yield msg

if __name__ == "__main__":
  log_path = sys.argv[1]
  lr = LogReader(log_path)
  for msg in lr:
    print(msg)
