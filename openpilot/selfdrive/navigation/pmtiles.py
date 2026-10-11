"""PMTiles v3 reader for the gzip-compressed Protomaps MVT basemap, over any byte-range source."""
import gzip
import struct
from bisect import bisect_right
from functools import lru_cache


def read_varint(data: bytes, offset: int) -> tuple[int, int]:
  value = 0
  for shift in range(0, 70, 7):
    if offset >= len(data):
      raise ValueError("Truncated varint")
    byte = data[offset]
    offset += 1
    value |= (byte & 127) << shift
    if byte < 128:
      return value, offset
  raise ValueError("Varint exceeds 64 bits")


def tile_id(z: int, x: int, y: int) -> int:
  """PMTiles orders each zoom level along a Hilbert curve."""
  if not 0 <= z <= 26 or not 0 <= x < 1 << z or not 0 <= y < 1 << z:
    raise ValueError("Invalid tile coordinates")
  distance = 0
  size = (1 << z) // 2
  while size:
    rx, ry = int(bool(x & size)), int(bool(y & size))
    distance += size * size * ((3 * rx) ^ ry)
    if not ry:
      if rx:
        x, y = size - 1 - x, size - 1 - y
      x, y = y, x
    size //= 2
  return ((1 << (2 * z)) - 1) // 3 + distance


def decode_directory(data: bytes) -> tuple[list[int], list[tuple[int, int, int]]]:
  count, offset = read_varint(data, 0)
  if count > len(data):
    raise ValueError("Invalid directory entry count")
  columns = []
  for _ in range(4):
    column = []
    for _ in range(count):
      value, offset = read_varint(data, offset)
      column.append(value)
    columns.append(column)
  ids, runs, lengths, offsets = columns
  for i in range(count):
    if i:
      ids[i] += ids[i - 1]
    if offsets[i] == 0:
      if i == 0:
        raise ValueError("First directory offset cannot be implicit")
      offsets[i] = offsets[i - 1] + lengths[i - 1]
    else:
      offsets[i] -= 1
    if lengths[i] == 0 or (i and ids[i] <= ids[i - 1]):
      raise ValueError("Invalid directory entry")
  if offset != len(data):
    raise ValueError("Trailing directory data")
  return ids, list(zip(runs, lengths, offsets, strict=True))


class PMTiles:
  def __init__(self, source):
    """source has size, read(offset, length), prefetch(ranges) and close()."""
    self.source = source
    self.size = source.size
    try:
      header = self.read(0, 127)
      if header[:8] != b'PMTiles\x03':
        raise ValueError("Expected a PMTiles v3 archive")
      self.root_offset, self.root_length, _, _, self.leaf_offset, self.leaf_length, self.data_offset, self.data_length = struct.unpack_from('<8Q', header, 8)
      compression, tile_type, self.min_zoom, self.max_zoom = header[97:99], header[99], header[100], header[101]
      if compression != b'\x02\x02' or tile_type != 1:
        raise ValueError("Expected gzip-compressed MVT vector tiles")
      if not 0 <= self.min_zoom <= self.max_zoom <= 26:
        raise ValueError("Unsupported PMTiles zoom range")
      for offset, length in ((self.root_offset, self.root_length), (self.leaf_offset, self.leaf_length), (self.data_offset, self.data_length)):
        if offset + length > self.size:
          raise ValueError("PMTiles section outside archive")
      self.directory = lru_cache(maxsize=32)(self._read_directory)
      self.directory(self.root_offset, self.root_length)
    except Exception:
      self.source.close()
      raise

  def read(self, offset: int, length: int) -> bytes:
    if offset < 0 or length < 0 or offset + length > self.size:
      raise ValueError("PMTiles read outside archive")
    data = self.source.read(offset, length)
    if len(data) != length:
      raise ValueError("Truncated PMTiles archive")
    return data

  def _read_directory(self, offset: int, length: int):
    return decode_directory(gzip.decompress(self.read(offset, length)))

  def locate(self, z: int, x: int, y: int) -> tuple[int, int] | None:
    """The byte range of a tile in the archive, or None when there is no such tile."""
    if not self.min_zoom <= z <= self.max_zoom:
      return None
    target = tile_id(z, x, y)
    offset, length = self.root_offset, self.root_length
    for _ in range(4):
      ids, entries = self.directory(offset, length)
      index = bisect_right(ids, target) - 1
      if index < 0:
        return None
      run, length, offset = entries[index]
      if run:
        if target >= ids[index] + run:
          return None
        if offset + length > self.data_length:
          raise ValueError("Tile outside PMTiles data section")
        return self.data_offset + offset, length
      if offset + length > self.leaf_length:
        raise ValueError("Directory outside PMTiles leaf section")
      offset += self.leaf_offset
    raise ValueError("PMTiles directory nesting exceeds four levels")

  def get(self, z: int, x: int, y: int) -> bytes | None:
    located = self.locate(z, x, y)
    return gzip.decompress(self.read(*located)) if located else None

  def prefetch(self, tiles):
    """Cache the (z, x, y) tiles ahead of reads."""
    self.source.prefetch([located for located in (self.locate(*tile) for tile in tiles) if located])

  def close(self):
    self.directory.cache_clear()
    self.source.close()
