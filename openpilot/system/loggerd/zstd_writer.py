from __future__ import annotations

from pathlib import Path
from types import TracebackType

import zstandard as zstd


LOG_COMPRESSION_LEVEL = 10


class ZstdFileWriter:
  def __init__(self, filename: str | Path, compression_level: int = LOG_COMPRESSION_LEVEL):
    self._file = open(filename, "wb")
    self._writer = zstd.ZstdCompressor(level=compression_level).stream_writer(self._file, closefd=False)
    self._closed = False

  def write(self, data: bytes | bytearray | memoryview) -> None:
    if self._closed:
      raise ValueError("write to closed ZstdFileWriter")
    self._writer.write(data)

  def close(self) -> None:
    if self._closed:
      return

    self._writer.close()
    self._file.flush()
    self._file.close()
    self._closed = True

  def __enter__(self) -> ZstdFileWriter:
    return self

  def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None) -> None:
    self.close()
