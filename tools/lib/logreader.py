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
import zstandard as zstd

from collections.abc import Callable, Iterable, Iterator
from typing import cast
from urllib.parse import parse_qs, urlparse

from cereal import log as capnp_log
from openpilot.common.swaglog import cloudlog
from openpilot.tools.lib.comma_car_segments import get_url as get_comma_segments_url
from openpilot.tools.lib.openpilotci import get_url
from openpilot.tools.lib.filereader import DATA_ENDPOINT, FileReader, file_exists, internal_source_available
from openpilot.tools.lib.route import Route, SegmentRange
from openpilot.tools.lib.log_time_series import msgs_to_time_series

LogMessage = type[capnp._DynamicStructReader]
LogIterable = Iterable[LogMessage]
RawLogIterable = Iterable[bytes]


def save_log(dest, log_msgs, compress=True):
  dat = b"".join(msg.as_builder().to_bytes() for msg in log_msgs)

  if compress and dest.endswith(".bz2"):
    dat = bz2.compress(dat)
  elif compress and dest.endswith(".zst"):
    dat = zstd.compress(dat, 10)

  with open(dest, "wb") as f:
    f.write(dat)

def decompress_stream(data: bytes):
  dctx = zstd.ZstdDecompressor()
  decompressed_data = b""

  with dctx.stream_reader(data) as reader:
    decompressed_data = reader.read()

  return decompressed_data

class _LogFileReader:
  def __init__(self, fn, canonicalize=True, only_union_types=False, sort_by_time=False, dat=None):
    self.data_version = None
    self._only_union_types = only_union_types

    ext = None
    if not dat:
      _, ext = os.path.splitext(urllib.parse.urlparse(fn).path)
      if ext not in ('', '.bz2', '.zst'):
        # old rlogs weren't compressed
        raise ValueError(f"unknown extension {ext}")

      with FileReader(fn) as f:
        dat = f.read()

    if ext == ".bz2" or dat.startswith(b'BZh9'):
      dat = bz2.decompress(dat)
    elif ext == ".zst" or dat.startswith(b'\x28\xB5\x2F\xFD'):
      # https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md#zstandard-frames
      dat = decompress_stream(dat)

    ents = capnp_log.Event.read_multiple_bytes(dat)

    self._ents = []
    try:
      for e in ents:
        self._ents.append(e)
    except capnp.KjException:
      warnings.warn("Corrupted events detected", RuntimeWarning, stacklevel=1)

    if sort_by_time:
      self._ents.sort(key=lambda x: x.logMonoTime)

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
  AUTO = "a"  # default to rlogs, fallback to qlogs
  AUTO_INTERACTIVE = "i"  # default to rlogs, fallback to qlogs with a prompt from the user


class FileName(enum.Enum):
  RLOG = ("rlog.zst", "rlog.bz2")
  QLOG = ("qlog.zst", "qlog.bz2")


LogPath = str | None
Source = Callable[[SegmentRange, FileName], list[LogPath]]

InternalUnavailableException = Exception("Internal source not available")


class LogsUnavailable(Exception):
  pass


def comma_api_source(sr: SegmentRange, fns: FileName) -> list[LogPath]:
  route = Route(sr.route_name)

  # comma api will have already checked if the file exists
  if fns == FileName.RLOG:
    return [route.log_paths()[seg] for seg in sr.seg_idxs]
  else:
    return [route.qlog_paths()[seg] for seg in sr.seg_idxs]


def internal_source(sr: SegmentRange, fns: FileName, endpoint_url: str = DATA_ENDPOINT) -> list[LogPath]:
  if not internal_source_available(endpoint_url):
    raise InternalUnavailableException

  def get_internal_url(sr: SegmentRange, seg, file):
    return f"{endpoint_url.rstrip('/')}/{sr.dongle_id}/{sr.log_id}/{seg}/{file}"

  return eval_source([[get_internal_url(sr, seg, fn) for fn in fns.value] for seg in sr.seg_idxs])


def openpilotci_source(sr: SegmentRange, fns: FileName) -> list[LogPath]:
  return eval_source([[get_url(sr.route_name, seg, fn) for fn in fns.value] for seg in sr.seg_idxs])


def comma_car_segments_source(sr: SegmentRange, fns: FileName) -> list[LogPath]:
  return eval_source([get_comma_segments_url(sr.route_name, seg) for seg in sr.seg_idxs])


def testing_closet_source(sr: SegmentRange, fns: FileName) -> list[LogPath]:
  if not internal_source_available('http://testing.comma.life'):
    raise InternalUnavailableException
  return eval_source([f"http://testing.comma.life/download/{sr.route_name.replace('|', '/')}/{seg}/rlog" for seg in sr.seg_idxs])


def direct_source(file_or_url: str) -> list[str]:
  return [file_or_url]


def eval_source(files: list[list[str] | str]) -> list[LogPath]:
  # Returns valid file URLs given a list of possible file URLs for each segment (e.g. rlog.bz2, rlog.zst)
  valid_files: list[LogPath] = []

  for urls in files:
    if isinstance(urls, str):
      urls = [urls]

    for url in urls:
      if file_exists(url):
        valid_files.append(url)
        break
    else:
      valid_files.append(None)

  return valid_files


def auto_source(identifier: str, sources: list[Source], default_mode: ReadMode) -> list[str]:
  exceptions = {}

  sr = SegmentRange(identifier)
  mode = default_mode if sr.selector is None else ReadMode(sr.selector)

  if mode == ReadMode.QLOG:
    try_fns = [FileName.QLOG]
  else:
    try_fns = [FileName.RLOG]

  # If selector allows it, fallback to qlogs
  if mode in (ReadMode.AUTO, ReadMode.AUTO_INTERACTIVE):
    try_fns.append(FileName.QLOG)

  # Build a dict of valid files as we evaluate each source. May contain mix of rlogs, qlogs, and None.
  # This function only returns when we've sourced all files, or throws an exception
  valid_files: dict[int, LogPath] = {}
  for fn in try_fns:
    for source in sources:
      try:
        files = source(sr, fn)

        # Check every source returns an expected number of files
        assert len(files) == len(valid_files) or len(valid_files) == 0, f"Source {source.__name__} returned unexpected number of files"

        # Build a dict of valid files
        for idx, f in enumerate(files):
          if valid_files.get(idx) is None:
            valid_files[idx] = f

        # We've found all files, return them
        if all(f is not None for f in valid_files.values()):
          return cast(list[str], list(valid_files.values()))

      except Exception as e:
        exceptions[source.__name__] = e

    if fn == try_fns[0]:
      missing_logs = list(valid_files.values()).count(None)
      if mode == ReadMode.AUTO:
        cloudlog.warning(f"{missing_logs}/{len(valid_files)} rlogs were not found, falling back to qlogs for those segments...")
      elif mode == ReadMode.AUTO_INTERACTIVE:
        if input(f"{missing_logs}/{len(valid_files)} rlogs were not found, would you like to fallback to qlogs for those segments? (y/N) ").lower() != "y":
          break

  missing_logs = list(valid_files.values()).count(None)
  raise LogsUnavailable(f"{missing_logs}/{len(valid_files)} logs were not found, please ensure all logs " +
                        "are uploaded. You can fall back to qlogs with '/a' selector at the end of the route name.\n\n" +
                        "Exceptions for sources:\n  - " + "\n  - ".join([f"{k}: {repr(v)}" for k, v in exceptions.items()]))


def parse_indirect(identifier: str) -> str:
  if "useradmin.comma.ai" in identifier:
    query = parse_qs(urlparse(identifier).query)
    return query["onebox"][0]
  return identifier


def parse_direct(identifier: str):
  if identifier.startswith(("http://", "https://", "cd:/")) or pathlib.Path(identifier).exists():
    return identifier
  return None


class LogReader:
  def _parse_identifier(self, identifier: str) -> list[str]:
    # useradmin, etc.
    identifier = parse_indirect(identifier)

    # direct url or file
    direct_parsed = parse_direct(identifier)
    if direct_parsed is not None:
      return direct_source(identifier)

    identifiers = auto_source(identifier, self.sources, self.default_mode)
    return identifiers

  def __init__(self, identifier: str | list[str], default_mode: ReadMode = ReadMode.RLOG,
               sources: list[Source] = None, sort_by_time=False, only_union_types=False):
    if sources is None:
      sources = [internal_source, openpilotci_source, comma_api_source,
                 comma_car_segments_source, testing_closet_source]

    self.default_mode = default_mode
    self.sources = sources
    self.identifier = identifier
    if isinstance(identifier, str):
      self.identifier = [identifier]

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

  def run_across_segments(self, num_processes, func, disable_tqdm=False, desc=None):
    with multiprocessing.Pool(num_processes) as pool:
      ret = []
      num_segs = len(self.logreader_identifiers)
      for p in tqdm.tqdm(pool.imap(partial(self._run_on_segment, func), range(num_segs)), total=num_segs, disable=disable_tqdm, desc=desc):
        ret.extend(p)
      return ret

  def reset(self):
    self.logreader_identifiers = []
    for identifier in self.identifier:
      self.logreader_identifiers.extend(self._parse_identifier(identifier))

  @staticmethod
  def from_bytes(dat):
    return _LogFileReader("", dat=dat)

  def filter(self, msg_type: str):
    return (getattr(m, m.which()) for m in filter(lambda m: m.which() == msg_type, self))

  def first(self, msg_type: str):
    return next(self.filter(msg_type), None)

  @property
  def time_series(self):
    return msgs_to_time_series(self)

if __name__ == "__main__":
  import codecs

  # capnproto <= 0.8.0 throws errors converting byte data to string
  # below line catches those errors and replaces the bytes with \x__
  codecs.register_error("strict", codecs.backslashreplace_errors)
  log_path = sys.argv[1]
  lr = LogReader(log_path, sort_by_time=True)
  for msg in lr:
    print(msg)
