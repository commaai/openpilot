#!/usr/bin/env python3
import bz2
import capnp
import enum
import itertools
import numpy as np
import os
import pathlib
import re
import sys
import urllib.parse
import warnings

from typing import Iterable, Iterator
from urllib.parse import parse_qs, urlparse

from cereal import log as capnp_log
from openpilot.selfdrive.test.openpilotci import get_url
from openpilot.tools.lib.filereader import FileReader
from openpilot.tools.lib.helpers import RE
from openpilot.tools.lib.route import Route, SegmentRange

LogIterable = Iterable[capnp._DynamicStructReader]


class _LogFileReader:
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
      warnings.warn("Corrupted events detected", RuntimeWarning, stacklevel=1)

    self._ents = list(sorted(_ents, key=lambda x: x.logMonoTime) if sort_by_time else _ents)
    self._ts = [x.logMonoTime for x in self._ents]

  def __iter__(self) -> Iterator[capnp._DynamicStructReader]:
    for ent in self._ents:
      if self._only_union_types:
        try:
          ent.which()
          yield ent
        except capnp.lib.capnp.KjException:
          pass
      else:
        yield ent


class ReadMode(enum.StrEnum):
  RLOG = "r" # only read rlogs
  QLOG = "q" # only read qlogs
  #AUTO = "a" # default to rlogs, fallback to qlogs, not supported yet


def create_slice_from_string(s: str):
  m = re.fullmatch(RE.SLICE, s)
  assert m is not None, f"Invalid slice: {s}"
  start, end, step = m.groups()
  start = int(start) if start is not None else None
  end = int(end) if end is not None else None
  step = int(step) if step is not None else None

  if start is not None and ":" not in s and end is None and step is None:
    return start
  return slice(start, end, step)

def parse_slice(sr: SegmentRange):
  route = Route(sr.route_name)
  segs = np.arange(route.max_seg_number+1)
  s = create_slice_from_string(sr._slice)
  return segs[s] if isinstance(s, slice) else [segs[s]]

def comma_api_source(sr: SegmentRange, mode=ReadMode.RLOG, **kwargs):
  segs = parse_slice(sr)
  route = Route(sr.route_name)

  log_paths = route.log_paths() if mode == ReadMode.RLOG else route.qlog_paths()

  invalid_segs = [seg for seg in segs if log_paths[seg] is None]

  assert not len(invalid_segs), f"Some of the requested segments are not available: {invalid_segs}"

  for seg in segs:
    yield _LogFileReader(log_paths[seg], **kwargs)

def internal_source(sr: SegmentRange, mode=ReadMode.RLOG, **kwargs):
  segs = parse_slice(sr)

  for seg in segs:
    yield _LogFileReader(f"cd:/{sr.dongle_id}/{sr.timestamp}/{seg}/{'rlog' if mode == ReadMode.RLOG else 'qlog'}.bz2", **kwargs)

def openpilotci_source(sr: SegmentRange, mode=ReadMode.RLOG, **kwargs):
  segs = parse_slice(sr)

  for seg in segs:
    yield _LogFileReader(get_url(sr.route_name, seg, 'rlog' if mode == ReadMode.RLOG else 'qlog'), **kwargs)

def direct_source(file_or_url, **kwargs):
  yield _LogFileReader(file_or_url, **kwargs)

def auto_source(*args, **kwargs):
  # Automatically determine viable source

  try:
    next(internal_source(*args, **kwargs))
    return internal_source(*args, **kwargs)
  except Exception:
    pass

  try:
    next(openpilotci_source(*args, **kwargs))
    return openpilotci_source(*args, **kwargs)
  except Exception:
    pass

  return comma_api_source(*args, **kwargs)

def parse_useradmin(identifier):
  if "useradmin.comma.ai" in identifier:
    query = parse_qs(urlparse(identifier).query)
    return query["onebox"][0]
  return None

def parse_cabana(identifier):
  if "cabana.comma.ai" in identifier:
    query = parse_qs(urlparse(identifier).query)
    return query["route"][0]
  return None

def parse_direct(identifier):
  if identifier.startswith(("http://", "https://", "cd:/")) or pathlib.Path(identifier).exists():
    return identifier
  return None

def parse_indirect(identifier):
  parsed = parse_useradmin(identifier) or parse_cabana(identifier)

  if parsed is not None:
    return parsed, comma_api_source, True

  return identifier, None, False


class LogReader:
  def _logreaders_from_identifier(self, identifier):
    parsed, source, is_indirect = parse_indirect(identifier)

    if not is_indirect:
      direct_parsed = parse_direct(identifier)
      if direct_parsed is not None:
        return direct_source(identifier, sort_by_time=self.sort_by_time)

    sr = SegmentRange(parsed)
    mode = self.default_mode if sr.selector is None else ReadMode(sr.selector)
    source = self.default_source if source is None else source

    return source(sr, mode, sort_by_time=self.sort_by_time, only_union_types=self.only_union_types)

  def __init__(self, identifier: str, default_mode=ReadMode.RLOG, default_source=auto_source, sort_by_time=False, only_union_types=False):
    self.default_mode = default_mode
    self.default_source = default_source
    self.identifier = identifier

    self.sort_by_time = sort_by_time
    self.only_union_types = only_union_types

    self.reset()

  def __iter__(self):
    self.reset()
    return self

  def __next__(self):
    return next(self.chain)

  def reset(self):
    self.lrs = self._logreaders_from_identifier(self.identifier)
    self.chain = itertools.chain(*self.lrs)

  @staticmethod
  def from_bytes(dat):
    return _LogFileReader("", dat=dat)


if __name__ == "__main__":
  import codecs
  # capnproto <= 0.8.0 throws errors converting byte data to string
  # below line catches those errors and replaces the bytes with \x__
  codecs.register_error("strict", codecs.backslashreplace_errors)
  log_path = sys.argv[1]
  lr = LogReader(log_path, sort_by_time=True)
  for msg in lr:
    print(msg)
