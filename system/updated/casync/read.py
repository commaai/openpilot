import abc
import io
import os
import pathlib
import shutil
import requests

from dataclasses import dataclass

from openpilot.system.updated.casync.chunk_reader import ChunkReader, DirectoryChunkReader, RemoteChunkReader
from openpilot.system.updated.casync.format import CAArchive, CAEntry, CAFilename, CAGoodbye, CAIndex, CAPayload, CASymlink


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

@dataclass
class Tree:
  nodes: list[Node]

  def __init__(self, archive: CAArchive):
    self.nodes = self.parse(archive, [CAFilename(None)])

  def parse(self, archive: CAArchive, current_paths: list[CAFilename]):
    current_entry = None

    ret: list[Node] = []

    for item in archive.items:
      if isinstance(item, CAFilename):
        current_paths.append(item)
      if isinstance(item, CAEntry):
        current_entry = item
      if isinstance(item, CAPayload):
        assert current_entry is not None
        ret.append(FileNode(current_entry, current_paths[1:], item))
        current_paths.pop()
      if isinstance(item, CASymlink):
        assert current_entry is not None
        ret.append(SymlinkNode(current_entry, current_paths[1:], item))
        current_paths.pop()
      if isinstance(item, CAGoodbye):
        current_paths.pop()
      if isinstance(item, CAArchive):
        ret.extend(self.parse(item, list(current_paths)))
        current_paths.pop()

    return ret

  def save(self, directory: pathlib.Path):
    for node in self.nodes:
      node.save(directory)


def parse_caidx(caindex: CAIndex, chunk_reader: ChunkReader) -> CAArchive:
  data = b"".join(chunk_reader.read(chunk) for chunk in caindex.chunks)

  archive = CAArchive.from_buffer(io.BytesIO(data))

  return archive


def extract_archive(archive: CAArchive, directory: pathlib.Path):
  shutil.rmtree(directory, ignore_errors=True)
  os.mkdir(directory)

  tree = Tree(archive)
  tree.save(directory)


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
  extract_local("/tmp/ceec/nightly_rebuild.caidx", pathlib.Path("/tmp/local"))
  extract_remote( \
    "https://commadist.blob.core.windows.net/openpilot-releases/0.9.7-48e61b19ad34fb63769f42d88c311054f30d0f5d-release.caidx", \
     pathlib.Path("/tmp/remote"))
