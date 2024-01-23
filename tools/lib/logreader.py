#!/usr/bin/env python3
import bz2
from functools import partial
import multiprocessing
import capnp
import enum
import numpy as np
import os
import pathlib
import re
import sys
import urllib.parse
import warnings

from typing import Iterable, Iterator, List, Type
from urllib.parse import parse_qs, urlparse

from cereal import log as capnp_log
from openpilot.tools.lib.openpilotci import get_url
from openpilot.tools.lib.filereader import FileReader, file_exists
from openpilot.tools.lib.helpers import RE
from openpilot.tools.lib.route import Route, SegmentRange

LogMessage = Type[capnp._DynamicStructReader]
LogIterable = Iterable[LogMessage]


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

def parse_slice(sr: SegmentRange, route: Route):
  segs = np.arange(route.max_seg_number+1)
  s = create_slice_from_string(sr._slice)
  return segs[s] if isinstance(s, slice) else [segs[s]]

def comma_api_source(sr: SegmentRange, route: Route, mode=ReadMode.RLOG):
  segs = parse_slice(sr, route)

  log_paths = route.log_paths() if mode == ReadMode.RLOG else route.qlog_paths()

  invalid_segs = [seg for seg in segs if log_paths[seg] is None]

  assert not len(invalid_segs), f"Some of the requested segments are not available: {invalid_segs}"

  return [(log_paths[seg]) for seg in segs]

def internal_source(sr: SegmentRange, route: Route, mode=ReadMode.RLOG):
  segs = parse_slice(sr, route)

  return [f"cd:/{sr.dongle_id}/{sr.timestamp}/{seg}/{'rlog' if mode == ReadMode.RLOG else 'qlog'}.bz2" for seg in segs]

def openpilotci_source(sr: SegmentRange, route: Route, mode=ReadMode.RLOG):
  segs = parse_slice(sr, route)

  return [get_url(sr.route_name, seg, 'rlog' if mode == ReadMode.RLOG else 'qlog') for seg in segs]

def direct_source(file_or_url):
  return [file_or_url]

def check_source(source, *args):
  try:
    files = source(*args)
    assert all(file_exists(f) for f in files)
    return True, files
  except Exception:
    return False, None

def auto_source(*args):
  # Automatically determine viable source
  for source in [internal_source, openpilotci_source]:
    valid, ret = check_source(source, *args)
    if valid:
      return ret

  return comma_api_source(*args)

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
  def _parse_identifiers(self, identifier: str | List[str]):
    if isinstance(identifier, list):
      return [i for j in identifier for i in self._parse_identifiers(j)]

    parsed, source, is_indirect = parse_indirect(identifier)

    if not is_indirect:
      direct_parsed = parse_direct(identifier)
      if direct_parsed is not None:
        return direct_source(identifier)

    sr = SegmentRange(parsed)
    route = Route(sr.route_name)
    mode = self.default_mode if sr.selector is None else ReadMode(sr.selector)
    source = self.default_source if source is None else source

    return source(sr, route, mode)

  def __init__(self, identifier: str | List[str], default_mode=ReadMode.RLOG, default_source=auto_source, sort_by_time=False, only_union_types=False):
    self.default_mode = default_mode
    self.default_source = default_source
    self.identifier = identifier

    self.sort_by_time = sort_by_time
    self.only_union_types = only_union_types

    self.reset()

  def __iter__(self):
    for identifier in self.logreader_identifiers:
      yield from _LogFileReader(identifier)

  def _run_on_segment(self, func, identifier):
    lr = _LogFileReader(identifier)
    return func(lr)

  def run_across_segments(self, num_processes, func):
    with multiprocessing.Pool(num_processes) as pool:
      ret = []
      for p in pool.map(partial(self._run_on_segment, func), self.logreader_identifiers):
        ret.extend(p)
      return ret

  def reset(self):
    self.logreader_identifiers = self._parse_identifiers(self.identifier)

  @staticmethod
  def from_bytes(dat):
    return _LogFileReader("", dat=dat)


def get_first_message(lr: LogIterable, msg_type):
  return next(filter(lambda m: m.which() == msg_type, lr), None)


if __name__ == "__main__":
  import codecs
  # capnproto <= 0.8.0 throws errors converting byte data to string
  # below line catches those errors and replaces the bytes with \x__
  codecs.register_error("strict", codecs.backslashreplace_errors)
  log_path = sys.argv[1]
  lr = LogReader(log_path, sort_by_time=True)
  for msg in lr:
    print(msg)
