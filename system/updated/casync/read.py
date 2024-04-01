import abc
import io
import os
import pathlib
import shutil
from typing import Iterator
import requests

from dataclasses import dataclass

from openpilot.system.updated.casync.chunk_reader import ChunkBuffer, ChunkReader, DirectoryChunkReader, RemoteChunkReader
from openpilot.system.updated.casync.format import CAArchive, CAEntry, CAFilename, CAGoodbye, CAIndex, CAPayload, CASymlink, parse_archive


@dataclass
class Node(abc.ABC):
  entry: CAEntry
  paths: list[CAFilename]

  def _path(self) -> str:
    return os.path.join(*[f.filename for f in self.paths if f.filename is not None])

  @abc.abstractmethod
  def save(self, directory: pathlib.Path):
    pass


@dataclass
class SymlinkNode(Node):
  symlink: CASymlink

  def save(self, directory: pathlib.Path):
    path = directory / self._path()
    path.parent.mkdir(exist_ok=True, parents=True)
    path.symlink_to(self.symlink.target)


@dataclass
class FileNode(Node):
  payload: CAPayload

  def save(self, directory: pathlib.Path):
    path = directory / self._path()
    path.parent.mkdir(exist_ok=True, parents=True)

    path.write_bytes(self.payload.data)
    path.chmod(self.entry.mode)



def parse_archive_tree(archive: CAArchive) -> Iterator[Node]:
  current_paths = [CAFilename(None)]
  current_entry = None

  for item in archive:
    if isinstance(item, CAFilename):
      current_paths.append(item)
    if isinstance(item, CAEntry):
      current_entry = item
    if isinstance(item, CAPayload):
      assert current_entry is not None
      yield FileNode(current_entry, current_paths[1:], item)
      current_paths.pop()
    if isinstance(item, CASymlink):
      assert current_entry is not None
      yield SymlinkNode(current_entry, current_paths[1:], item)
      current_paths.pop()
    if isinstance(item, CAGoodbye):
      current_paths.pop()


def parse_caidx(caindex: CAIndex, chunk_reader: ChunkReader) -> CAArchive:
  buffer = ChunkBuffer(caindex.chunks, chunk_reader)
  archive = parse_archive(buffer)

  return archive


def extract_archive(archive: CAArchive, directory: pathlib.Path):
  shutil.rmtree(directory, ignore_errors=True)
  os.mkdir(directory)

  for entry in parse_archive_tree(archive):
    entry.save(directory)


def extract_local(local_caidx, output_path: pathlib.Path):
  caidx = CAIndex.from_file(local_caidx)
  casync_store = os.path.join(os.path.dirname(local_caidx), "default.castr")

  chunk_reader = DirectoryChunkReader(casync_store)

  archive = parse_caidx(caidx, chunk_reader)
  extract_archive(archive, output_path)


def extract_remote(remote_caidx, output_path: pathlib.Path):
  resp = requests.get(remote_caidx, timeout=30)
  resp.raise_for_status()
  caidx = CAIndex.from_buffer(io.BytesIO(resp.content))
  casync_store = os.path.join(os.path.dirname(remote_caidx), "default.castr")

  chunk_reader = RemoteChunkReader(casync_store)

  archive = parse_caidx(caidx, chunk_reader)
  extract_archive(archive, output_path)


if __name__ == "__main__":
  print("extracting a local caidx...")
  extract_local("/tmp/ceec/nightly_rebuild.caidx", pathlib.Path("/tmp/local"))
  print("extracting a remote caidx...")
  extract_remote( \
    "https://commadist.blob.core.windows.net/openpilot-releases/0.9.7-a51b164cbd5cdb557a0ae392300b9b91e16c9053-release.caidx", \
     pathlib.Path("/tmp/remote"))
