from dataclasses import dataclass
import io
import struct
from typing import Self

from openpilot.system.updated.casync.chunk_reader import CAChunk


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


class CAFormatBase:
  HEADER_CA_FORMAT: int | None = None
  FORMAT_STR: str

  @classmethod
  def from_buffer(cls, b: io.BytesIO) -> Self:
    if cls.HEADER_CA_FORMAT is not None:
      header = CAFormatHeader.from_buffer(b)
    else:
      header = None

    unpacked = struct.unpack(cls.FORMAT_STR, b.read(struct.calcsize(cls.FORMAT_STR)))

    if header is not None:
      unpacked = (header, *unpacked)

    return cls(*unpacked)


@dataclass
class CAFormatHeader(CAFormatBase):
  FORMAT_STR = "<QQ"

  size: int
  format_type: int


@dataclass
class CaFormatIndex(CAFormatBase):
  HEADER_CA_FORMAT = CA_FORMAT_INDEX
  FORMAT_STR = "<QQQQ"

  header: CAFormatHeader
  flags: int
  chunk_size_min: int
  chunk_size_avg: int
  chunk_size_max: int


@dataclass
class CATableItem(CAFormatBase):
  FORMAT_STR = "<Q32s"

  offset: int
  chunk_id: bytes

  @property
  def marker(self):
    return struct.unpack("<Q", self.chunk_id[24:])[0]


@dataclass
class CATable:
  index: CaFormatIndex

  items: list[CATableItem]

  @staticmethod
  def from_buffer(b: io.BytesIO):
    index = CaFormatIndex.from_buffer(b)
    header = CAFormatHeader.from_buffer(b)
    assert header.format_type == CA_FORMAT_TABLE

    items = []
    while True:
      item = CATableItem.from_buffer(b)

      if item.marker == CA_FORMAT_TABLE_TAIL_MARKER:
        break
      items.append(item)

    return CATable(index, items)

  def to_chunks(self) -> list[CAChunk]:
    offset = 0
    ret = []
    for i, item in enumerate(self.items):
      length = item.offset - offset

      assert length <= self.index.chunk_size_max

      # Last chunk can be smaller
      if i < len(self.items) - 1:
        assert length >= self.index.chunk_size_min

      ret.append(CAChunk(item.chunk_id, offset, length))
      offset += length

    return ret


@dataclass
class CAIndex:
  chunks: list[CAChunk]

  @staticmethod
  def from_buffer(b: io.BytesIO):
    table = CATable.from_buffer(b)

    chunks = table.to_chunks()

    return CAIndex(chunks)

  @staticmethod
  def from_file(filepath):
    with open(filepath, "rb") as f:
      return CAIndex.from_buffer(f)



def string_from_buffer(b: io.BytesIO):
  _ = CAFormatHeader.from_buffer(b)

  string = b""

  while len(string) < CA_MAX_FILENAME_SIZE:
    c = b.read(1)
    if c == b'\x00':
      break
    string += c

  return string.decode("utf-8")


@dataclass
class CAFilename:
  filename: str | None

  @staticmethod
  def from_buffer(b: io.BytesIO):
    return CAFilename(string_from_buffer(b))


@dataclass
class CASymlink:
  target: str

  @staticmethod
  def from_buffer(b: io.BytesIO):
    return CAFilename(string_from_buffer(b))


@dataclass
class CAEntry(CAFormatBase):
  HEADER_CA_FORMAT = CA_FORMAT_ENTRY
  FORMAT_STR = "<QQQQQQ"

  header: CAFormatHeader
  feature_flags: int
  mode: int
  flags: int
  uid: int
  gid: int
  mtime: int


@dataclass
class CAPayload:
  data: bytes

  @staticmethod
  def from_buffer(b: io.BytesIO):
    header = CAFormatHeader.from_buffer(b)

    data = b.read(header.size - 16)
    return CAPayload(data)


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
  def from_buffer(b: io.BytesIO) -> 'CAArchive':
    entry = CAEntry.from_buffer(b)

    assert entry.feature_flags == REQUIRED_FLAGS
    items: list[CAEntry | CAGoodbye | CAFilename | CAArchive | CASymlink | CAPayload] = [entry]

    cur = b.tell()

    size = b.seek(0, 2)
    b.seek(cur, 0)

    while b.tell() < size:
      header = CAFormatHeader.from_buffer(b)
      b.seek(-16, 1) # reset back to header

      if header.format_type == CA_FORMAT_FILENAME:
        filename = CAFilename.from_buffer(b)
        archive = CAArchive.from_buffer(b)
        items.append(filename)
        items.append(archive)
      elif header.format_type == CA_FORMAT_GOODBYE:
        goodbye = CAGoodbye.from_buffer(b)
        items.append(goodbye)
        break
      elif header.format_type == CA_FORMAT_PAYLOAD:
        payload = CAPayload.from_buffer(b)
        items.append(payload)
        break
      elif header.format_type == CA_FORMAT_SYMLINK:
        symlink = CASymlink.from_buffer(b)
        items.append(symlink)
        break
      else:
        raise Exception(f"unsupported type: {header.format_type:02x}")

    return CAArchive(items)
