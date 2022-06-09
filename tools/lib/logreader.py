#!/usr/bin/env python3
import os
import sys
import bz2
import urllib.parse
import capnp
import warnings


from cereal import log as capnp_log
from tools.lib.filereader import FileReader
from tools.lib.route import Route, SegmentName

# this is an iterator itself, and uses private variables from LogReader
class MultiLogIterator:
  def __init__(self, log_paths, sort_by_time=False):
    self._log_paths = log_paths
    self.sort_by_time = sort_by_time

    self._first_log_idx = next(i for i in range(len(log_paths)) if log_paths[i] is not None)
    self._current_log = self._first_log_idx
    self._idx = 0
    self._log_readers = [None]*len(log_paths)
    self.start_time = self._log_reader(self._first_log_idx)._ts[0]

  def _log_reader(self, i):
    if self._log_readers[i] is None and self._log_paths[i] is not None:
      log_path = self._log_paths[i]
      self._log_readers[i] = LogReader(log_path, sort_by_time=self.sort_by_time)

    return self._log_readers[i]

  def __iter__(self):
    return self

  def _inc(self):
    lr = self._log_reader(self._current_log)
    if self._idx < len(lr._ents)-1:
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
      ret = lr._ents[self._idx]
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

  def reset(self):
    self.__init__(self._log_paths, sort_by_time=self.sort_by_time)


class LogReader:
  def __init__(self, fn, canonicalize=True, only_union_types=False, sort_by_time=False, dat=None):
    self.data_version = None
    self._only_union_types = only_union_types

    ext = None
    if not dat:
      _, ext = os.path.splitext(urllib.parse.urlparse(fn).path)
      if ext not in ('', '.bz2'):
        # old rlogs weren't bz2 compressed
        raise Exception(f"unknown extension {ext}")

      with FileReader(fn) as f:
        dat = f.read()

    if ext == ".bz2" or dat.startswith(b'BZh9'):
      dat = bz2.decompress(dat)

    ents = capnp_log.Event.read_multiple_bytes(dat)

    _ents = []
    try:
      for e in ents:
        _ents.append(e)
    except capnp.KjException:
      warnings.warn("Corrupted events detected", RuntimeWarning)

    self._ents = list(sorted(_ents, key=lambda x: x.logMonoTime) if sort_by_time else _ents)
    self._ts = [x.logMonoTime for x in self._ents]

  @classmethod
  def from_bytes(cls, dat):
    return cls("", dat=dat)

  def __iter__(self):
    for ent in self._ents:
      if self._only_union_types:
        try:
          ent.which()
          yield ent
        except capnp.lib.capnp.KjException:
          pass
      else:
        yield ent

def logreader_from_route_or_segment(r, sort_by_time=False):
  sn = SegmentName(r, allow_route_name=True)
  route = Route(sn.route_name.canonical_name)
  if sn.segment_num < 0:
    return MultiLogIterator(route.log_paths(), sort_by_time=sort_by_time)
  else:
    return LogReader(route.log_paths()[sn.segment_num], sort_by_time=sort_by_time)


if __name__ == "__main__":
  import codecs
  # capnproto <= 0.8.0 throws errors converting byte data to string
  # below line catches those errors and replaces the bytes with \x__
  codecs.register_error("strict", codecs.backslashreplace_errors)
  log_path = sys.argv[1]
  lr = LogReader(log_path, sort_by_time=True)
  for msg in lr:
    print(msg)
