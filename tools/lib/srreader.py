import enum
import itertools
import numpy as np
import pathlib
import re
from urllib.parse import parse_qs, urlparse

from openpilot.selfdrive.test.openpilotci import get_url
from openpilot.tools.lib.helpers import RE
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.route import Route, SegmentRange

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

def comma_api_source(sr: SegmentRange, mode=ReadMode.RLOG, sort_by_time=False):
  segs = parse_slice(sr)
  route = Route(sr.route_name)

  log_paths = route.log_paths() if mode == ReadMode.RLOG else route.qlog_paths()

  invalid_segs = [seg for seg in segs if log_paths[seg] is None]

  assert not len(invalid_segs), f"Some of the requested segments are not available: {invalid_segs}"

  for seg in segs:
    yield LogReader(log_paths[seg], sort_by_time=sort_by_time)

def internal_source(sr: SegmentRange, mode=ReadMode.RLOG, sort_by_time=False):
  segs = parse_slice(sr)

  for seg in segs:
    yield LogReader(f"cd:/{sr.dongle_id}/{sr.timestamp}/{seg}/{'rlog' if mode == ReadMode.RLOG else 'qlog'}.bz2", sort_by_time=sort_by_time)

def openpilotci_source(sr: SegmentRange, mode=ReadMode.RLOG, sort_by_time=False):
  segs = parse_slice(sr)

  for seg in segs:
    yield LogReader(get_url(sr.route_name, seg, 'rlog' if mode == ReadMode.RLOG else 'qlog'), sort_by_time=sort_by_time)

def direct_source(file_or_url, sort_by_time):
  yield LogReader(file_or_url, sort_by_time=sort_by_time)

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

def parse_cd(identifier):
  if "cd:/" in identifier:
    return identifier.replace("cd:/", "")
  return None

def parse_direct(identifier):
  if "https://" in identifier or "http://" in identifier or pathlib.Path(identifier).exists():
    return identifier
  return None

def parse_indirect(identifier):
  parsed = parse_useradmin(identifier) or parse_cabana(identifier)

  if parsed is not None:
    return parsed, comma_api_source, True

  parsed = parse_cd(identifier)
  if parsed is not None:
    return parsed, internal_source, True

  return identifier, None, False

class SegmentRangeReader:
  def _logreaders_from_identifier(self, identifier):
    parsed, source, is_indirect = parse_indirect(identifier)

    if not is_indirect:
      direct_parsed = parse_direct(identifier)
      if direct_parsed is not None:
        return direct_source(identifier, sort_by_time=self.sort_by_time)

    sr = SegmentRange(parsed)
    mode = self.default_mode if sr.selector is None else ReadMode(sr.selector)
    source = self.default_source if source is None else source

    return source(sr, mode, sort_by_time=self.sort_by_time)

  def __init__(self, identifier: str, default_mode=ReadMode.RLOG, default_source=auto_source, sort_by_time=False):
    self.default_mode = default_mode
    self.default_source = default_source
    self.sort_by_time = sort_by_time
    self.identifier = identifier

    self.reset()

  def __iter__(self):
    return self

  def __next__(self):
    return next(self.chain)

  def reset(self):
    self.lrs = self._logreaders_from_identifier(self.identifier)
    self.chain = itertools.chain(*self.lrs)
