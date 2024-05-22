#!/usr/bin/env python3
import bz2
from functools import partial
import multiprocessing
import capnp
import enum
import os
import pathlib
import sys
import tqdm
import urllib.parse
import warnings

from collections.abc import Callable, Iterable, Iterator
from urllib.parse import parse_qs, urlparse

from cereal import log as capnp_log
from openpilot.common.swaglog import cloudlog
from openpilot.tools.lib.comma_car_segments import get_url as get_comma_segments_url
from openpilot.tools.lib.openpilotci import get_url
from openpilot.tools.lib.filereader import FileReader, file_exists, internal_source_available
from openpilot.tools.lib.route import Route, SegmentRange

LogMessage = type[capnp._DynamicStructReader]
LogIterable = Iterable[LogMessage]
RawLogIterable = Iterable[bytes]


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
  RLOG = "r"  # only read rlogs
  QLOG = "q"  # only read qlogs
  SANITIZED = "s"  # read from the commaCarSegments database
  AUTO = "a"  # default to rlogs, fallback to qlogs
  AUTO_INTERACTIVE = "i"  # default to rlogs, fallback to qlogs with a prompt from the user


LogPath = str | None
LogPaths = list[LogPath]
ValidFileCallable = Callable[[LogPath], bool]
Source = Callable[[SegmentRange, ReadMode], LogPaths]

InternalUnavailableException = Exception("Internal source not available")

def default_valid_file(fn: LogPath) -> bool:
  return fn is not None and file_exists(fn)


def auto_strategy(rlog_paths: LogPaths, qlog_paths: LogPaths, interactive: bool, valid_file: ValidFileCallable) -> LogPaths:
  # auto select logs based on availability
  if any(rlog is None or not valid_file(rlog) for rlog in rlog_paths) and all(qlog is not None and valid_file(qlog) for qlog in qlog_paths):
    if interactive:
      if input("Some rlogs were not found, would you like to fallback to qlogs for those segments? (y/n) ").lower() != "y":
        return rlog_paths
    else:
      cloudlog.warning("Some rlogs were not found, falling back to qlogs for those segments...")

    return [rlog if valid_file(rlog) else (qlog if valid_file(qlog) else None)
            for (rlog, qlog) in zip(rlog_paths, qlog_paths, strict=True)]
  return rlog_paths


def apply_strategy(mode: ReadMode, rlog_paths: LogPaths, qlog_paths: LogPaths, valid_file: ValidFileCallable = default_valid_file) -> LogPaths:
  if mode == ReadMode.RLOG:
    return rlog_paths
  elif mode == ReadMode.QLOG:
    return qlog_paths
  elif mode == ReadMode.AUTO:
    return auto_strategy(rlog_paths, qlog_paths, False, valid_file)
  elif mode == ReadMode.AUTO_INTERACTIVE:
    return auto_strategy(rlog_paths, qlog_paths, True, valid_file)
  raise Exception(f"invalid mode: {mode}")


def comma_api_source(sr: SegmentRange, mode: ReadMode) -> LogPaths:
  route = Route(sr.route_name)

  rlog_paths = [route.log_paths()[seg] for seg in sr.seg_idxs]
  qlog_paths = [route.qlog_paths()[seg] for seg in sr.seg_idxs]

  # comma api will have already checked if the file exists
  def valid_file(fn):
    return fn is not None

  return apply_strategy(mode, rlog_paths, qlog_paths, valid_file=valid_file)


def internal_source(sr: SegmentRange, mode: ReadMode) -> LogPaths:
  if not internal_source_available():
    raise InternalUnavailableException

  def get_internal_url(sr: SegmentRange, seg, file):
    return f"cd:/{sr.dongle_id}/{sr.timestamp}/{seg}/{file}.bz2"

  rlog_paths = [get_internal_url(sr, seg, "rlog") for seg in sr.seg_idxs]
  qlog_paths = [get_internal_url(sr, seg, "qlog") for seg in sr.seg_idxs]

  return apply_strategy(mode, rlog_paths, qlog_paths)


def openpilotci_source(sr: SegmentRange, mode: ReadMode) -> LogPaths:
  rlog_paths = [get_url(sr.route_name, seg, "rlog") for seg in sr.seg_idxs]
  qlog_paths = [get_url(sr.route_name, seg, "qlog") for seg in sr.seg_idxs]

  return apply_strategy(mode, rlog_paths, qlog_paths)


def comma_car_segments_source(sr: SegmentRange, mode=ReadMode.RLOG) -> LogPaths:
  return [get_comma_segments_url(sr.route_name, seg) for seg in sr.seg_idxs]


def direct_source(file_or_url: str) -> LogPaths:
  return [file_or_url]


def get_invalid_files(files):
  for f in files:
    if f is None or not file_exists(f):
      yield f


def check_source(source: Source, *args) -> LogPaths:
  files = source(*args)
  assert next(get_invalid_files(files), False) is False
  return files


def auto_source(sr: SegmentRange, mode=ReadMode.RLOG) -> LogPaths:
  if mode == ReadMode.SANITIZED:
    return comma_car_segments_source(sr, mode)

  SOURCES: list[Source] = [internal_source, openpilotci_source, comma_api_source, comma_car_segments_source,]
  exceptions = []

  # for automatic fallback modes, auto_source needs to first check if rlogs exist for any source
  if mode in [ReadMode.AUTO, ReadMode.AUTO_INTERACTIVE]:
    for source in SOURCES:
      try:
        return check_source(source, sr, ReadMode.RLOG)
      except Exception:
        pass

  # Automatically determine viable source
  for source in SOURCES:
    try:
      return check_source(source, sr, mode)
    except Exception as e:
      exceptions.append(e)

  raise Exception(f"auto_source could not find any valid source, exceptions for sources: {exceptions}")


def parse_useradmin(identifier: str):
  if "useradmin.comma.ai" in identifier:
    query = parse_qs(urlparse(identifier).query)
    return query["onebox"][0]
  return None


def parse_cabana(identifier: str):
  if "cabana.comma.ai" in identifier:
    query = parse_qs(urlparse(identifier).query)
    return query["route"][0]
  return None


def parse_direct(identifier: str):
  if identifier.startswith(("http://", "https://", "cd:/")) or pathlib.Path(identifier).exists():
    return identifier
  return None


def parse_indirect(identifier: str):
  parsed = parse_useradmin(identifier) or parse_cabana(identifier)

  if parsed is not None:
    return parsed, comma_api_source, True

  return identifier, None, False


class LogReader:
  def _parse_identifiers(self, identifier: str | list[str]):
    if isinstance(identifier, list):
      return [i for j in identifier for i in self._parse_identifiers(j)]

    parsed, source, is_indirect = parse_indirect(identifier)

    if not is_indirect:
      direct_parsed = parse_direct(identifier)
      if direct_parsed is not None:
        return direct_source(identifier)

    sr = SegmentRange(parsed)
    mode = self.default_mode if sr.selector is None else ReadMode(sr.selector)
    source = self.default_source if source is None else source

    identifiers = source(sr, mode)

    invalid_count = len(list(get_invalid_files(identifiers)))
    assert invalid_count == 0, f"{invalid_count}/{len(identifiers)} invalid log(s) found, please ensure all logs \
are uploaded or auto fallback to qlogs with '/a' selector at the end of the route name."
    return identifiers

  def __init__(self, identifier: str | list[str], default_mode: ReadMode = ReadMode.RLOG,
               default_source=auto_source, sort_by_time=False, only_union_types=False):
    self.default_mode = default_mode
    self.default_source = default_source
    self.identifier = identifier

    self.sort_by_time = sort_by_time
    self.only_union_types = only_union_types

    self.__lrs: dict[int, _LogFileReader] = {}
    self.reset()

  def _get_lr(self, i):
    if i not in self.__lrs:
      self.__lrs[i] = _LogFileReader(self.logreader_identifiers[i], sort_by_time=self.sort_by_time, only_union_types=self.only_union_types)
    return self.__lrs[i]

  def __iter__(self):
    for i in range(len(self.logreader_identifiers)):
      yield from self._get_lr(i)

  def _run_on_segment(self, func, i):
    return func(self._get_lr(i))

  def run_across_segments(self, num_processes, func):
    with multiprocessing.Pool(num_processes) as pool:
      ret = []
      num_segs = len(self.logreader_identifiers)
      for p in tqdm.tqdm(pool.imap(partial(self._run_on_segment, func), range(num_segs)), total=num_segs):
        ret.extend(p)
      return ret

  def reset(self):
    self.logreader_identifiers = self._parse_identifiers(self.identifier)

  @staticmethod
  def from_bytes(dat):
    return _LogFileReader("", dat=dat)

  def filter(self, msg_type: str):
    return (getattr(m, m.which()) for m in filter(lambda m: m.which() == msg_type, self))

  def first(self, msg_type: str):
    return next(self.filter(msg_type), None)


if __name__ == "__main__":
  import codecs

  # capnproto <= 0.8.0 throws errors converting byte data to string
  # below line catches those errors and replaces the bytes with \x__
  codecs.register_error("strict", codecs.backslashreplace_errors)
  log_path = sys.argv[1]
  lr = LogReader(log_path, sort_by_time=True)
  for msg in lr:
    print(msg)
