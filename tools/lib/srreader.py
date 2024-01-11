import enum
import re
import numpy as np
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


class SegmentRangeReader:
  def __init__(self, segment_range: str, default_mode=ReadMode.RLOG, source=auto_source, sort_by_time=False):
    sr = SegmentRange(segment_range)

    mode = default_mode if sr.selector is None else ReadMode(sr.selector)

    self.lrs = source(sr, mode, sort_by_time)

  def __iter__(self):
    for lr in self.lrs:
      for m in lr:
        yield m
