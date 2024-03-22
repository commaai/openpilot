import abc
from dataclasses import dataclass
import io
import os
import struct
from openpilot.system.updated.casync.reader import DirectoryChunkReader


CA_FORMAT_TABLE_TAIL_MARKER = 0xe75b9e112f17417
FLAGS = 0xb000000000000000

CA_HEADER_LEN = 32
CA_TABLE_HEADER_LEN = 16
CA_TABLE_ENTRY_LEN = 40
CA_TABLE_MIN_LEN = CA_TABLE_HEADER_LEN + CA_TABLE_ENTRY_LEN

CA_FORMAT_INDEX = 0x96824d9c7b129ff9
CA_FORMAT_TABLE = 0xe75b9e112f17417d
CA_FORMAT_ENTRY = 0x1396fabcea5bbb51


@dataclass
class CAFormatHeader(abc.ABC):
  size: int
  type: int

  @staticmethod
  @abc.abstractmethod
  def from_buffer(b: io.BytesIO) -> 'CAFormatHeader':
    pass


def create_header_with_type(MAGIC_TYPE) -> type[CAFormatHeader]:

  class MagicCAFormatHeader(CAFormatHeader):
    @staticmethod
    def from_buffer(b: io.BytesIO):
      # Parse table header
      length, magic = struct.unpack("<QQ", b.read(CA_TABLE_HEADER_LEN))
      assert magic == MAGIC_TYPE
      return MagicCAFormatHeader(length, magic)

  return MagicCAFormatHeader


CAIndexHeader = create_header_with_type(CA_FORMAT_INDEX)
CATableHeader = create_header_with_type(CA_FORMAT_TABLE)
CAEntryHeader = create_header_with_type(CA_FORMAT_ENTRY)


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
  header: CAIndexHeader
  flags: int
  chunk_size_min: int
  chunk_size_avg: int
  chunk_size_max: int

  @staticmethod
  def from_buffer(b: io.BytesIO):
    header = CAIndexHeader.from_buffer(b)

    return CaFormatIndex(header, *struct.unpack("<QQQQ", b.read(CA_HEADER_LEN)))


@dataclass
class CAIndex:
  chunks: list[CAChunk]

  @staticmethod
  def from_buffer(b: io.BytesIO):
    b.seek(0, os.SEEK_END)
    length = b.tell()
    b.seek(0, os.SEEK_SET)

    #format_index = CaFormatIndex.from_buffer(b)
    #table_header = CATableHeader.from_buffer(b)

    num_chunks = (length - CA_HEADER_LEN - CA_TABLE_MIN_LEN) // CA_TABLE_ENTRY_LEN

    chunks = []

    offset = 0
    for _ in range(num_chunks):
      chunk = CAChunk.from_buffer(b, offset)
      offset += chunk.length
      chunks.append(chunk)

    return CAIndex(chunks)

  @staticmethod
  def from_file(filepath):
    with open(filepath, "rb") as f:
      return CAIndex.from_buffer(f)

  def chunks(self):
    return self.chunks


@dataclass
class CAEntry:
  header: CAEntryHeader
  feature_flags: int
  mode: int
  flags: int
  uid: int
  gid: int
  mtime: int

  @staticmethod
  def from_buffer(b: io.BytesIO):
    entry = CAEntryHeader.from_buffer(b)
    return CAEntry(entry, *struct.unpack("<QQQQQQ", b.read(8*6)))


@dataclass
class CAArchive:
  entry: CAEntryHeader

  @staticmethod
  def from_buffer(b: io.BytesIO):
    entry = CAEntry.from_buffer(b)

    return CAArchive(entry)


@dataclass
class CAFile:
  filename: str
  data: bytes

  @staticmethod
  def from_bytes(b: io.BytesIO):
    filename = ""
    while True:
      c = b.read(1)
      if c == 0:
        break

    archive = CAArchive.from_buffer(b)

    return CAFile(filename, b"1234")



@dataclass
class CATar:
  archive: CAArchive

  @staticmethod
  def from_bytes(b: io.BytesIO):
    archive = CAArchive.from_buffer(b)

    return CATar(archive)

  def files(self) -> list[CAFile]:
    print(self.archive)
    #files = []
    return CAFile()


def parse_caidx(caidx_path) -> CATar:
  caidx = CAIndex.from_file(caidx_path)

  chunk_reader = DirectoryChunkReader("/tmp/test_casync/default.castr")

  data = b"".join(chunk_reader.read(chunk) for chunk in caidx.chunks[:1])

  tar = CATar.from_bytes(io.BytesIO(data))

  return tar


def extract_tar(tar: CATar, directory: str):
  for file in tar.files():
    with open(f"{directory}") as f:
      f.write(file.data)


if __name__ == "__main__":
  tar = parse_caidx("/tmp/test_casync/test.caidx")

  extract_tar(tar, "/tmp/test")
