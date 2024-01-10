import enum
import re
import numpy as np
from openpilot.selfdrive.test.openpilotci import get_url
from openpilot.tools.lib.helpers import RE
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.route import Route, SegmentRange

class ReadMode(enum.Enum):
  RLOG = 0 # only read rlogs
  QLOG = 1 # only read qlogs
  #AUTO = 2 # default to rlogs, fallback to qlogs, not supported yet


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

def comma_api_source(sr: SegmentRange, mode=ReadMode.RLOG):
  segs = parse_slice(sr)
  route = Route(sr.route_name)

  log_paths = route.log_paths() if mode == ReadMode.RLOG else route.qlog_paths()

  for seg in segs:
    yield LogReader(log_paths[seg])

def internal_source(sr: SegmentRange, mode=ReadMode.RLOG):
  segs = parse_slice(sr)

  for seg in segs:
    yield LogReader(f"cd:/{sr.dongle_id}/{sr.timestamp}/{seg}/{'rlog' if mode == ReadMode.RLOG else 'qlog'}.bz2")

def openpilotci_source(sr: SegmentRange, mode=ReadMode.RLOG):
  segs = parse_slice(sr)

  for seg in segs:
    yield LogReader(get_url(sr.route_name, seg, 'rlog' if mode == ReadMode.RLOG else 'qlog'))

def auto_source(sr: SegmentRange, mode=ReadMode.RLOG):
  # Automatically determine viable source

  try:
    next(internal_source(sr, mode))
    return internal_source(sr, mode)
  except Exception:
    pass

  try:
    next(openpilotci_source(sr, mode))
    return openpilotci_source(sr, mode)
  except Exception:
    pass

  return comma_api_source(sr, mode)


class SegmentRangeReader:
  def __init__(self, segment_range: str, mode=ReadMode.RLOG, source=auto_source):
    sr = SegmentRange(segment_range)
    self.lrs = source(sr, mode)

  def __iter__(self):
    for lr in self.lrs:
      for m in lr:
        yield m
