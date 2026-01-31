#!/usr/bin/env python3
import os
import capnp
import urllib.parse
import warnings
from urllib.request import urlopen
import zstandard as zstd

from opendbc.car.common.basedir import BASEDIR

capnp_log = capnp.load(os.path.join(BASEDIR, "rlog.capnp"))


def decompress_stream(data: bytes):
  dctx = zstd.ZstdDecompressor()
  decompressed_data = b""

  with dctx.stream_reader(data) as reader:
    decompressed_data = reader.read()

  return decompressed_data


class LogReader:
  def __init__(self, fn, only_union_types=False, sort_by_time=False):
    self._only_union_types = only_union_types
    _, ext = os.path.splitext(urllib.parse.urlparse(fn).path)

    if fn.startswith("http"):
      with urlopen(fn) as f:
        dat = f.read()
    else:
      with open(fn, "rb") as f:
        dat = f.read()

    if ext == ".zst" or dat.startswith(b'\x28\xB5\x2F\xFD'):
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

  def __iter__(self):
    for ent in self._ents:
      if self._only_union_types:
        try:
          ent.which()
          yield ent
        except capnp.lib.capnp.KjException:
          pass
      else:
        yield ent

  def filter(self, msg_type: str):
    return (getattr(m, m.which()) for m in filter(lambda m: m.which() == msg_type, self))

  def first(self, msg_type: str):
    return next(self.filter(msg_type), None)
