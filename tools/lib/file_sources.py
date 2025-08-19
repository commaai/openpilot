from collections.abc import Callable

from openpilot.tools.lib.comma_car_segments import get_url as get_comma_segments_url
from openpilot.tools.lib.openpilotci import get_url
from openpilot.tools.lib.filereader import DATA_ENDPOINT, file_exists, internal_source_available, FilePath
from openpilot.tools.lib.route import Route, SegmentRange, FileName

FileNames = tuple[str, ...]
Source = Callable[[SegmentRange, list[int], FileNames], dict[int, FilePath]]
InternalUnavailableException = Exception("Internal source not available")


def comma_api_source(sr: SegmentRange, seg_idxs: list[int], fns: FileNames) -> dict[int, FilePath]:
  route = Route(sr.route_name)

  # comma api will have already checked if the file exists
  if fns == FileName.RLOG:
    return {seg: route.log_paths()[seg] for seg in seg_idxs}
  else:
    return {seg: route.qlog_paths()[seg] for seg in seg_idxs}


def internal_source(sr: SegmentRange, seg_idxs: list[int], fns: FileNames, endpoint_url: str = DATA_ENDPOINT) -> dict[int, FilePath]:
  if not internal_source_available(endpoint_url):
    raise InternalUnavailableException

  def get_internal_url(sr: SegmentRange, seg, file):
    return f"{endpoint_url.rstrip('/')}/{sr.dongle_id}/{sr.log_id}/{seg}/{file}"

  return eval_source({seg: [get_internal_url(sr, seg, fn) for fn in fns] for seg in seg_idxs})


def openpilotci_source(sr: SegmentRange, seg_idxs: list[int], fns: FileNames) -> dict[int, FilePath]:
  return eval_source({seg: [get_url(sr.route_name, seg, fn) for fn in fns] for seg in seg_idxs})


def comma_car_segments_source(sr: SegmentRange, seg_idxs: list[int], fns: FileNames) -> dict[int, FilePath]:
  return eval_source({seg: get_comma_segments_url(sr.route_name, seg) for seg in seg_idxs})


def direct_source(file_or_url: str) -> list[str]:
  return [file_or_url]


def eval_source(files: dict[int, list[str] | str]) -> dict[int, FilePath]:
  # Returns valid file URLs given a list of possible file URLs for each segment (e.g. rlog.bz2, rlog.zst)
  valid_files: dict[int, FilePath] = {}

  for seg_idx, urls in files.items():
    if isinstance(urls, str):
      urls = [urls]

    # Add first valid file URL or None
    for url in urls:
      if file_exists(url):
        valid_files[seg_idx] = url
        break
    else:
      valid_files[seg_idx] = None

  return valid_files
