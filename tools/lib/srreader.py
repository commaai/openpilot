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

  def parse_negative_indexing(i):
    return route.max_seg_number - abs(i) + 1

  if start is None:
    start = 0
  else:
    start = parse_negative_indexing(start) if start < 0 else start

  if end is None:
    end = route.max_seg_number if sr.start is None else start
  else:
    end = parse_negative_indexing(end) if end < 0 else end - 1

  return start, end

def comma_api_source(sr: SegmentRange, rlog=True):
  route = Route(sr.route_name)
  start, end = parse_start_end(sr, route)

  log_paths = route.log_paths() if rlog else route.qlog_paths()

  for seg in range(start, end+1):
    yield LogReader(log_paths[seg])

comma_api_source_qlog = partial(comma_api_source, rlog=False)

def internal_source(sr: SegmentRange):
  start, end = parse_start_end(sr)

  for seg in range(start, end+1):
    yield LogReader(f"cd:/{sr.dongle_id}/{sr.timestamp}/{seg}/rlog.bz2")

def openpilotci_source(sr: SegmentRange):
  start, end = parse_start_end(sr)

  for seg in range(start, end+1):
    yield LogReader(get_url(sr.route_name, seg))

class SegmentRangeReader:
  def __init__(self, segment_range: str, source=comma_api_source):
    sr = SegmentRange(segment_range)
    self.lrs = source(sr)

  def __iter__(self):
    for lr in self.lrs:
      for m in lr:
        yield m
