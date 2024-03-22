import abc
from dataclasses import dataclass
import io
import os
import pathlib
import shutil
import struct

import requests
from openpilot.system.updated.casync.reader import DirectoryChunkReader, AutoChunkReader


# from https://github.com/systemd/casync/blob/e6817a79d89b48e1c6083fb1868a28f1afb32505/src/caformat.h#L49

FLAG_PERMISSIONS = 0xb000000000000100
FLAG_SYMLINKS = 0xb000000000000200

REQUIRED_FLAGS = FLAG_SYMLINKS | FLAG_PERMISSIONS # correspond to [--with=symlinks --with=permissions]

CA_HEADER_LEN = 32
CA_TABLE_HEADER_LEN = 16
CA_TABLE_ENTRY_LEN = 40
CA_TABLE_MIN_LEN = CA_TABLE_HEADER_LEN + CA_TABLE_ENTRY_LEN

CA_FORMAT_INDEX = 0x96824d9c7b129ff9
CA_FORMAT_ENTRY = 0x1396fabcea5bbb51
CA_FORMAT_GOODBYE = 0xdfd35c5e8327c403
CA_FORMAT_FILENAME = 0x6dbb6ebcb3161f0b
CA_FORMAT_PAYLOAD = 0x8b9e1d93d6dcffc9
CA_FORMAT_SYMLINK = 0x664a6fb6830e0d6c

CA_FORMAT_TABLE = 0xe75b9e112f17417d
CA_FORMAT_TABLE_TAIL_MARKER = 0x4b4f050e5549ecd1

CA_MAX_FILENAME_SIZE = 256


@dataclass
class CAFormatHeader(abc.ABC):
  size: int
  type: int

  @staticmethod
  def from_buffer(b: io.BytesIO) -> 'CAFormatHeader':
    size, type = struct.unpack("<QQ", b.read(CA_TABLE_HEADER_LEN))

    return CAFormatHeader(size, type)


@dataclass
class CAChunk:
  sha: bytes
  offset: int
  length: int

  @staticmethod
  def from_buffer(b: io.BytesIO, last_offset: int):
    new_offset = struct.unpack("<Q", b.read(8))[0]

    sha = b.read(32)
    length = new_offset - last_offset

    return CAChunk(sha, last_offset, length)


@dataclass
class CaFormatIndex:
  header: CAFormatHeader
  flags: int
  chunk_size_min: int
  chunk_size_avg: int
  chunk_size_max: int

  @staticmethod
  def from_buffer(b: io.BytesIO):
    header = CAFormatHeader.from_buffer(b)
    assert header.type == CA_FORMAT_INDEX
    return CaFormatIndex(header, *struct.unpack("<QQQQ", b.read(CA_HEADER_LEN)))


@dataclass
class CATableItem:
  offset: int
  chunk_id: bytes

  marker: int

  @staticmethod
  def from_buffer(b: io.BytesIO):
    offset = struct.unpack("<Q", b.read(8))[0]
    chunk_id = b.read(32)

    marker = struct.unpack("<Q", chunk_id[24:])[0]

    return CATableItem(offset, chunk_id, marker)


@dataclass
class CATable:
  items: list[CATableItem]

  @staticmethod
  def from_buffer(b: io.BytesIO):
    header = CAFormatHeader.from_buffer(b)
    assert header.type == CA_FORMAT_TABLE

    items = []
    while True:
      item = CATableItem.from_buffer(b)

      if item.marker == CA_FORMAT_TABLE_TAIL_MARKER:
        break
      items.append(item)

    return CATable(items)

  def to_chunks(self) -> list[CAChunk]:
    offset = 0
    ret = []
    for item in self.items:
      length = item.offset - offset
      ret.append(CAChunk(item.chunk_id, item.offset, length))
      offset += length

    return ret


@dataclass
class CAIndex:
  chunks: list[CAChunk]

  @staticmethod
  def from_buffer(b: io.BytesIO):
    b.seek(0, os.SEEK_END)
    length = b.tell()
    b.seek(0, os.SEEK_SET)

    index = CaFormatIndex.from_buffer(b)
    table = CATable.from_buffer(b)

    chunks = table.to_chunks()

    return CAIndex(chunks)

  @staticmethod
  def from_file(filepath):
    with open(filepath, "rb") as f:
      return CAIndex.from_buffer(f)

  def chunks(self):
    return self.chunks


@dataclass
class CAFilename:
  filename: str

  @staticmethod
  def from_buffer(b: io.BytesIO):
    header = CAFormatHeader.from_buffer(b)

    filename = b""

    while len(filename) < CA_MAX_FILENAME_SIZE:
      c = b.read(1)
      if c == b'\x00':
        break
      filename += c

    return CAFilename(filename.decode("utf-8"))


@dataclass
class CAEntry:
  feature_flags: int
  mode: int
  flags: int
  uid: int
  gid: int
  mtime: int

  @staticmethod
  def from_buffer(b: io.BytesIO):
    entry = CAFormatHeader.from_buffer(b)
    return CAEntry(*struct.unpack("<QQQQQQ", b.read(8*6)))


@dataclass
class CAPayload:
  data: bytes

  @staticmethod
  def from_buffer(b: io.BytesIO):
    header = CAFormatHeader.from_buffer(b)

    data = b.read(header.size - 16)
    return CAPayload(data)

  def __repr__(self):
    return f"CAPayload(data={len(self.data)} bytes)"


@dataclass
class CASymlink:
  target: str

  @staticmethod
  def from_buffer(b: io.BytesIO):
    header = CAFormatHeader.from_buffer(b)

    target = b""

    while len(target) < CA_MAX_FILENAME_SIZE:
      c = b.read(1)
      if c == b'\x00':
        break
      target += c

    return CASymlink(target.decode("utf-8"))


@dataclass
class CAGoodbye:
  @staticmethod
  def from_buffer(b: io.BytesIO):
    header = CAFormatHeader.from_buffer(b)
    b.read(header.size - 16)

    return CAGoodbye()


@dataclass
class CAArchive:
  items: list

  @staticmethod
  def from_buffer(b: io.BytesIO):
    entry = CAEntry.from_buffer(b)

    assert entry.feature_flags == REQUIRED_FLAGS
    items = [entry]

    cur = b.tell()

    size = b.seek(0, 2)
    b.seek(cur, 0)

    while b.tell() < size:
      header = CAFormatHeader.from_buffer(b)
      b.seek(-16, 1) # reset back to header

      if header.type == CA_FORMAT_GOODBYE:
        goodbye = CAGoodbye.from_buffer(b)
        items.append(goodbye)
      elif header.type == CA_FORMAT_FILENAME:
        filename = CAFilename.from_buffer(b)
        archive = CAArchive.from_buffer(b)
        items.append(filename)
        items.append(archive)
      elif header.type == CA_FORMAT_PAYLOAD:
        payload = CAPayload.from_buffer(b)
        items.append(payload)
      elif header.type == CA_FORMAT_SYMLINK:
        symlink = CASymlink.from_buffer(b)
        items.append(symlink)
      else:
        raise Exception(f"unsupported type: {header.type:02x}")

    return CAArchive(items)


@dataclass
class Node:
  entry: CAEntry
  paths: list[CAFilename]

  def _path(self):
    return os.path.join(*[f.filename for f in self.paths])


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
    self.nodes = self.parse(archive, [None])

  def parse(self, archive: CAArchive, current_paths: list[CAFilename | None]):
    current_entry = None

    ret = []

    for item in archive.items:
      if isinstance(item, CAFilename):
        current_paths.append(item)
      if isinstance(item, CAEntry):
        current_entry = item
      if isinstance(item, CAPayload):
        ret.append(FileNode(current_entry, current_paths[1:], item))
        current_paths.pop()
      if isinstance(item, CASymlink):
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


def parse_caidx(caidx_path) -> CAArchive:
  if os.path.isfile(caidx_path):
    caidx = CAIndex.from_file(caidx_path)
  else:
    resp = requests.get(caidx_path, timeout=30)
    resp.raise_for_status()
    caidx = CAIndex.from_buffer(io.BytesIO(resp.content))

  casync_directory = os.path.dirname(caidx_path)

  chunk_reader = DirectoryChunkReader(os.path.join(casync_directory, "default.castr"))

  data = b"".join(chunk_reader.read(chunk) for chunk in caidx.chunks)

  archive = CAArchive.from_buffer(io.BytesIO(data))

  return archive


def extract_archive(archive: CAArchive, directory: pathlib.Path):
  shutil.rmtree(directory, ignore_errors=True)
  os.mkdir(directory)

  tree = Tree(archive)
  tree.save(directory)


if __name__ == "__main__":
  archive = parse_caidx("/tmp/casync/nightly_rebuild.caidx")
  extract_archive(archive, pathlib.Path("/tmp/nightly-test"))
