"""Small QR encoder for the UI's byte-mode, error-correction-level-L codes."""

import numpy as np
import pyray as rl


# (ec codewords per block, block count) for levels L, M, Q, H, versions 1-40
_EC = [
  ((7, 1), (10, 1), (13, 1), (17, 1)), ((10, 1), (16, 1), (22, 1), (28, 1)), ((15, 1), (26, 1), (18, 2), (22, 2)),
  ((20, 1), (18, 2), (26, 2), (16, 4)), ((26, 1), (24, 2), (18, 4), (22, 4)), ((18, 2), (16, 4), (24, 4), (28, 4)),
  ((20, 2), (18, 4), (18, 6), (26, 5)), ((24, 2), (22, 4), (22, 6), (26, 6)), ((30, 2), (22, 5), (20, 8), (24, 8)),
  ((18, 4), (26, 5), (24, 8), (28, 8)), ((20, 4), (30, 5), (28, 8), (24, 11)), ((24, 4), (22, 8), (26, 10), (28, 11)),
  ((26, 4), (22, 9), (24, 12), (22, 16)), ((30, 4), (24, 9), (20, 16), (24, 16)), ((22, 6), (24, 10), (30, 12), (24, 18)),
  ((24, 6), (28, 10), (24, 17), (30, 16)), ((28, 6), (28, 11), (28, 16), (28, 19)), ((30, 6), (26, 13), (28, 18), (28, 21)),
  ((28, 7), (26, 14), (26, 21), (26, 25)), ((28, 8), (26, 16), (30, 20), (28, 25)), ((28, 8), (26, 17), (28, 23), (30, 25)),
  ((28, 9), (28, 17), (30, 23), (24, 34)), ((30, 9), (28, 18), (30, 25), (30, 30)), ((30, 10), (28, 20), (30, 27), (30, 32)),
  ((26, 12), (28, 21), (30, 29), (30, 35)), ((28, 12), (28, 23), (28, 34), (30, 37)), ((30, 12), (28, 25), (30, 34), (30, 40)),
  ((30, 13), (28, 26), (30, 35), (30, 42)), ((30, 14), (28, 28), (30, 38), (30, 45)), ((30, 15), (28, 29), (30, 40), (30, 48)),
  ((30, 16), (28, 31), (30, 43), (30, 51)), ((30, 17), (28, 33), (30, 45), (30, 54)), ((30, 18), (28, 35), (30, 48), (30, 57)),
  ((30, 19), (28, 37), (30, 51), (30, 60)), ((30, 19), (28, 38), (30, 53), (30, 63)), ((30, 20), (28, 40), (30, 56), (30, 66)),
  ((30, 21), (28, 43), (30, 59), (30, 70)), ((30, 22), (28, 45), (30, 62), (30, 74)), ((30, 24), (28, 47), (30, 65), (30, 77)),
  ((30, 25), (28, 49), (30, 68), (30, 81)),
]

# GF(256) with the QR polynomial x^8 + x^4 + x^3 + x^2 + 1: powers of alpha and their logs
_EXP = [1]
for _ in range(254):
  _EXP.append(_EXP[-1] << 1 ^ (0x11D if _EXP[-1] & 0x80 else 0))
_LOG = {v: i for i, v in enumerate(_EXP)}


def _bch_format(data: int) -> int:
  v = data << 10
  for shift in range(14, 9, -1):
    if v >> shift & 1:
      v ^= 0x537 << (shift - 10)
  return (data << 10 | v) ^ 0x5412


# 15-bit format info indexed by (level bits << 3 | mask). Level bits: L=01, M=00, Q=11, H=10.
_FORMATS = [_bch_format(d) for d in range(32)]


def _raw_modules(version: int) -> int:
  result = (16 * version + 128) * version + 64
  if version >= 2:
    align = version // 7 + 2
    result -= (25 * align - 10) * align - 55
  return result - (36 if version >= 7 else 0)


def _block_lengths(version: int, level: int) -> list[int]:
  """Data codewords per Reed-Solomon block. The last blocks may be one longer."""
  ec, nblocks = _EC[version - 1][level]
  total = _raw_modules(version) // 8 - ec * nblocks
  return [total // nblocks + (i >= nblocks - total % nblocks) for i in range(nblocks)]


def _interleaved(version: int, level: int) -> list[tuple[int, int]]:
  """(block, index within block) of each transmitted codeword: data column-major, then ECC column-major."""
  ec, nblocks = _EC[version - 1][level]
  lens = _block_lengths(version, level)
  data = [(b, i) for i in range(max(lens)) for b in range(nblocks) if i < lens[b]]
  ecc = [(b, lens[b] + i) for i in range(ec) for b in range(nblocks)]
  return data + ecc


def _capacity(version: int) -> int:
  return sum(_block_lengths(version, 0))


def _append_bits(bits: list[int], value: int, length: int) -> None:
  bits.extend((value >> i) & 1 for i in range(length - 1, -1, -1))


def _data_codewords(data: bytes, version: int) -> bytes:
  """Byte-mode-encode the payload, terminated and padded to the version's capacity."""
  capacity = _capacity(version)
  bits: list[int] = []
  _append_bits(bits, 4, 4)  # byte mode
  _append_bits(bits, len(data), 8 if version <= 9 else 16)
  for value in data:
    _append_bits(bits, value, 8)
  bits.extend([0] * min(4, capacity * 8 - len(bits)))  # terminator
  bits.extend([0] * (-len(bits) % 8))  # byte alignment
  result = bytearray(sum(bits[i + j] << (7 - j) for j in range(8)) for i in range(0, len(bits), 8))
  pad = (0xEC, 0x11)
  while len(result) < capacity:
    result.append(pad[(len(result) - (len(bits) // 8)) & 1])
  return bytes(result)


def _codewords(data: bytes, version: int) -> bytes:
  """Split data codewords into Reed-Solomon blocks and interleave data + ECC."""
  data = _data_codewords(data, version)
  divisor = _divisor(_EC[version - 1][0][0])
  blocks = []
  offset = 0
  for length in _block_lengths(version, 0):
    block = data[offset:offset + length]
    blocks.append(block + _remainder(block, divisor))
    offset += length
  return bytes(blocks[b][i] for b, i in _interleaved(version, 0))


def _multiply(x: int, y: int) -> int:
  return _EXP[(_LOG[x] + _LOG[y]) % 255] if x and y else 0


def _divisor(degree: int) -> bytes:
  result = bytearray([0] * (degree - 1) + [1])
  root = 1
  for _ in range(degree):
    for j in range(degree):
      result[j] = _multiply(result[j], root)
      if j + 1 < degree:
        result[j] ^= result[j + 1]
    root = _multiply(root, 2)
  return bytes(result)


def _remainder(data: bytes, divisor: bytes) -> bytes:
  result = bytearray(len(divisor))
  for value in data:
    factor = value ^ result.pop(0)
    result.append(0)
    for i, coefficient in enumerate(divisor):
      result[i] ^= _multiply(coefficient, factor)
  return bytes(result)


def _alignment_positions(version: int) -> list[int]:
  if version == 1:
    return []
  count = version // 7 + 2
  step = (version * 8 + count * 3 + 5) // (count * 4 - 4) * 2
  return [6] + [version * 4 + 10 - step * i for i in range(count - 1)][::-1]


class _Qr:
  def __init__(self, version: int, data: bytes):
    self.version = version
    self.size = version * 4 + 17
    self.modules = [[False] * self.size for _ in range(self.size)]
    self.function = [[False] * self.size for _ in range(self.size)]
    self._draw_functions()
    self._draw_data(_codewords(data, version))
    for y in range(self.size):
      for x in range(self.size):
        if not self.function[y][x]:
          self.modules[y][x] ^= (x + y) % 2 == 0
    self._format()

  def _set_function(self, x: int, y: int, dark: bool) -> None:
    if 0 <= x < self.size and 0 <= y < self.size:
      self.modules[y][x] = dark
      self.function[y][x] = True

  def _finder(self, x: int, y: int) -> None:
    for dy in range(-4, 5):
      for dx in range(-4, 5):
        distance = max(abs(dx), abs(dy))
        self._set_function(x + dx, y + dy, distance != 2 and distance != 4)

  def _alignment(self, x: int, y: int) -> None:
    for dy in range(-2, 3):
      for dx in range(-2, 3):
        self._set_function(x + dx, y + dy, max(abs(dx), abs(dy)) != 1)

  def _draw_functions(self) -> None:
    for i in range(self.size):
      self._set_function(6, i, i % 2 == 0)
      self._set_function(i, 6, i % 2 == 0)
    self._finder(3, 3)
    self._finder(self.size - 4, 3)
    self._finder(3, self.size - 4)
    positions = _alignment_positions(self.version)
    for y in positions:
      for x in positions:
        if not ((x == 6 and y in (6, self.size - 7)) or (x == self.size - 7 and y == 6)):
          self._alignment(x, y)
    # reserve the format-info modules before the data is placed; the real
    # values are written by the second _format call after masking
    self._format()
    if self.version >= 7:
      value = self.version
      for _ in range(12):
        value = (value << 1) ^ ((value >> 11) * 0x1F25)
      value = self.version << 12 | value
      for i in range(18):
        bit = ((value >> i) & 1) != 0
        a = self.size - 11 + i % 3
        b = i // 3
        self._set_function(a, b, bit)
        self._set_function(b, a, bit)

  def _format(self) -> None:
    for i in range(15):
      bit = ((_FORMATS[1 << 3 | 0] >> i) & 1) != 0  # level L, mask 0
      y_pos = i if i < 6 else i + 1 if i < 8 else self.size - 15 + i
      self._set_function(8, y_pos, bit)
      x_pos = self.size - 1 - i if i < 8 else 15 - i if i < 9 else 14 - i
      self._set_function(x_pos, 8, bit)
    self._set_function(8, self.size - 8, True)

  def _draw_data(self, data: bytes) -> None:
    bits = ((byte >> s) & 1 for byte in data for s in reversed(range(8)))
    upward = True
    right = self.size - 1
    while right >= 1:
      if right == 6:  # skip the vertical timing column
        right = 5
      for vert in range(self.size):
        y = self.size - 1 - vert if upward else vert
        for x in (right, right - 1):
          if not self.function[y][x]:
            self.modules[y][x] = bool(next(bits, 0))
      upward = not upward
      right -= 2


def make_texture(data: str, inverted: bool = False) -> rl.Texture:
  """Render a URL as the RGBA QR texture used by the UI. The texture upload
  copies the pixels, so the intermediate image/array don't need to outlive it."""
  raw = data.encode()
  for version in range(1, 21):
    count_bits = 8 if version <= 9 else 16
    if 4 + count_bits + len(raw) * 8 <= _capacity(version) * 8:
      break
  else:
    raise ValueError("QR URL is too long")
  modules = np.pad(_Qr(version, raw).modules, 0 if inverted else 4)
  modules = np.repeat(np.repeat(modules, 10, axis=0), 10, axis=1)
  gray = ((modules == inverted) * 255).astype(np.uint8)
  img_array = np.dstack((gray, gray, gray, np.full_like(gray, 255)))

  rl_image = rl.Image()
  rl_image.data = rl.ffi.cast("void *", img_array.ctypes.data)
  rl_image.width = img_array.shape[1]
  rl_image.height = img_array.shape[0]
  rl_image.mipmaps = 1
  rl_image.format = rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
  return rl.load_texture_from_image(rl_image)
