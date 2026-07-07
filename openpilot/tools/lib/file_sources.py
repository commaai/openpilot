import os
import socket

from collections.abc import Callable
from functools import cache
from urllib.parse import urlparse

from openpilot.tools.lib.comma_car_segments import get_url as get_comma_segments_url
from openpilot.tools.lib.openpilotci import get_url
from openpilot.tools.lib.route import Route, SegmentRange, FileName
from openpilot.tools.lib.filereader import FsspecFile
from openpilot.common.utils import retry

# When passed a tuple of file names, each source will return the first that exists (rlog.zst, rlog.bz2)
FileNames = tuple[str, ...]
Source = Callable[[SegmentRange, list[int], FileNames], dict[int, str]]

InternalUnavailableException = Exception("Internal source not available")

DATA_ENDPOINT = os.getenv("DATA_ENDPOINT", "mkv://data-raw.comma.internal/")

@cache
@retry(delay=0.0)
def internal_source_available(url: str) -> bool:
  if os.path.isdir(url):
    return True

  try:
    parsed = urlparse(url)
    hostname = parsed.hostname
    port = parsed.port or 80
    if hostname is None:
      return False

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
      s.settimeout(0.5)
      s.connect((hostname, port))
    return True
  except (socket.gaierror, ConnectionRefusedError):
    pass
  return False


def comma_api_source(sr: SegmentRange, seg_idxs: list[int], fns: FileNames) -> dict[int, str]:
  route = Route(sr.route_name)

  # comma api will have already checked if the file exists
  if fns == FileName.RLOG:
    return {seg: route.log_paths()[seg] for seg in seg_idxs if route.log_paths()[seg] is not None}
  else:
    return {seg: route.qlog_paths()[seg] for seg in seg_idxs if route.qlog_paths()[seg] is not None}


def internal_source(sr: SegmentRange, seg_idxs: list[int], fns: FileNames, endpoint_url: str = DATA_ENDPOINT) -> dict[int, str]:
  if not internal_source_available(endpoint_url):
    raise InternalUnavailableException

  def get_internal_url(sr: SegmentRange, seg, file):
    return f"{endpoint_url.rstrip('/')}/{sr.dongle_id}/{sr.log_id}/{seg}/{file}"

  return eval_source({seg: [get_internal_url(sr, seg, fn) for fn in fns] for seg in seg_idxs})


def openpilotci_source(sr: SegmentRange, seg_idxs: list[int], fns: FileNames) -> dict[int, str]:
  return eval_source({seg: [get_url(sr.route_name, seg, fn) for fn in fns] for seg in seg_idxs})


def comma_car_segments_source(sr: SegmentRange, seg_idxs: list[int], fns: FileNames) -> dict[int, str]:
  return eval_source({seg: get_comma_segments_url(sr.route_name, seg) for seg in seg_idxs})


def eval_source(files: dict[int, list[str] | str]) -> dict[int, str]:
  # Returns valid file URLs given a list of possible file URLs for each segment (e.g. rlog.bz2, rlog.zst)
  valid_files: dict[int, str] = {}

  for seg_idx, urls in files.items():
    if isinstance(urls, str):
      urls = [urls]

    # Add first valid file URL
    for url in urls:
      if FsspecFile.exists(url):
        valid_files[seg_idx] = url
        break

  return valid_files
