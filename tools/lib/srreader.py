import numpy as np
from functools import partial

from openpilot.selfdrive.test.openpilotci import get_url
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.route import Route, SegmentRange


def parse_start_end(sr: SegmentRange, route = None):
  start = int(sr.start) if sr.start is not None else None
  end = int(sr.end) if sr.end is not None else None

  if route is None:
    assert start is not None and end is not None, "segment(s) must be provided for non-api sources"
    assert start >= 0 and end >= 0, "relative segment(s) not supported for non-api sources"

    return start, end

  segs = np.arange(route.max_seg_number+1)

  if start is not None:
    if end is None:
      return segs[start:start+1 if start >= 0 else None]
    else:
      return segs[start:end]

  return segs

def comma_api_source(sr: SegmentRange, rlog=True):
  route = Route(sr.route_name)
  segs = parse_start_end(sr, route)

  log_paths = route.log_paths() if rlog else route.qlog_paths()

  for seg in segs:
    yield LogReader(log_paths[seg])

comma_api_source_qlog = partial(comma_api_source, rlog=False)

def internal_source(sr: SegmentRange):
  segs = parse_start_end(sr)

  for seg in segs:
    yield LogReader(f"cd:/{sr.dongle_id}/{sr.timestamp}/{seg}/rlog.bz2")

def openpilotci_source(sr: SegmentRange):
  segs = parse_start_end(sr)

  for seg in segs:
    yield LogReader(get_url(sr.route_name, seg))

class SegmentRangeReader:
  def __init__(self, segment_range: str, source=comma_api_source):
    sr = SegmentRange(segment_range)
    self.lrs = source(sr)

  def __iter__(self):
    for lr in self.lrs:
      for m in lr:
        yield m
