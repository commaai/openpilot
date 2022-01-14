#!/usr/bin/env python3
import sys
from selfdrive.ui.replay.logreader_pyx import LogReader # pylint: disable=no-name-in-module, import-error

# this is an iterator itself, and uses private variables from LogReader
class MultiLogIterator:
  def __init__(self, log_paths, sort_by_time=False, cache_to_local=True):
    self._log_paths = log_paths
    self.sort_by_time = sort_by_time
    self.cache_to_local = cache_to_local

    self._first_log_idx = next(i for i in range(len(log_paths)) if log_paths[i] is not None)
    self._current_log = self._first_log_idx
    self._idx = 0
    self._log_readers = [None]*len(log_paths)
    self.start_time = self._log_reader(self._first_log_idx).ts(0)

  def _log_reader(self, i):
    if self._log_readers[i] is None and self._log_paths[i] is not None:
      log_path = self._log_paths[i]
      self._log_readers[i] = LogReader(log_path, sort_by_time=self.sort_by_time, cache_to_local=self.cache_to_local)

    return self._log_readers[i]

  def __iter__(self):
    return self

  def _inc(self):
    lr = self._log_reader(self._current_log)
    if self._idx < len(lr)-1:
      self._idx += 1
    else:
      self._idx = 0
      self._current_log = next(i for i in range(self._current_log + 1, len(self._log_readers) + 1)
                               if i == len(self._log_readers) or self._log_paths[i] is not None)
      if self._current_log == len(self._log_readers):
        raise StopIteration

  def __next__(self):
    while 1:
      lr = self._log_reader(self._current_log)
      ret = lr[self._idx]
      self._inc()
      return ret

  def tell(self):
    # returns seconds from start of log
    return (self._log_reader(self._current_log).ts(self._idx) - self.start_time) * 1e-9

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

if __name__ == "__main__":
  import codecs
  # capnproto <= 0.8.0 throws errors converting byte data to string
  # below line catches those errors and replaces the bytes with \x__
  codecs.register_error("strict", codecs.backslashreplace_errors)
  log_path = sys.argv[1]
  lr = LogReader(log_path, sort_by_time=True)
  for msg in lr:
    print(msg)
