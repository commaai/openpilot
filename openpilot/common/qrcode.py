"""QR code encoding, decoding, and UI textures."""

import functools
import itertools

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


# ---- Symbol structure for decoding ----


class QRError(Exception):
  pass


_LEVELS = (1, 0, 3, 2)  # format info level bits -> column in _EC

_MASKS = [
  lambda i, j: (i + j) % 2 == 0,
  lambda i, j: i % 2 == 0,
  lambda i, j: j % 3 == 0,
  lambda i, j: (i + j) % 3 == 0,
  lambda i, j: (i // 2 + j // 3) % 2 == 0,
  lambda i, j: (i * j) % 2 + (i * j) % 3 == 0,
  lambda i, j: ((i * j) % 2 + (i * j) % 3) % 2 == 0,
  lambda i, j: ((i + j) % 2 + (i * j) % 3) % 2 == 0,
]

_ALIGNMENT = np.ones((5, 5), dtype=bool)
_ALIGNMENT[1:4, 1:4] = False
_ALIGNMENT[2, 2] = True


def _gf_inv(a: int) -> int:
  return _EXP[-_LOG[a] % 255]


@functools.lru_cache
def _data_coords(version: int) -> tuple[np.ndarray, np.ndarray]:
  """(rows, cols) of the data and error correction modules in placement order: two-column zigzag from the right."""
  dim = version * 4 + 17
  func = np.zeros((dim, dim), dtype=bool)  # finder, timing, alignment, format, and version modules
  func[:9, :9] = func[:9, dim - 8:] = func[dim - 8:, :9] = True
  func[6, :] = func[:, 6] = True
  positions = _alignment_positions(version)
  for r, c in itertools.product(positions, positions):
    if (r, c) not in ((6, 6), (6, dim - 7), (dim - 7, 6)):
      func[r - 2:r + 3, c - 2:c + 3] = True
  if version >= 7:
    func[:6, dim - 11:dim - 8] = func[dim - 11:dim - 8, :6] = True
  ys = np.arange(dim)
  rows, cols = [], []
  # the vertical timing column is skipped, so the pairs left of it start at odd columns
  for i, right in enumerate(col if col > 6 else col - 1 for col in range(dim - 1, 0, -2)):
    r = np.repeat(ys[::-1] if i % 2 == 0 else ys, 2)
    c = np.tile((right, right - 1), dim)
    keep = ~func[r, c]
    rows.append(r[keep])
    cols.append(c[keep])
  return np.concatenate(rows), np.concatenate(cols)


# ---- Matrix decoding ----


def _poly_eval(p: list[int], x: int) -> int:
  # p is highest degree first
  y = 0
  for c in p:
    y = _multiply(y, x) ^ c
  return y


_EXP_TABLE = np.array(_EXP)
_LOG_TABLE = np.array([_LOG.get(v, 0) for v in range(256)])


def _syndromes(msg: list[int], nsym: int) -> list[int]:
  """syn[i] = msg(alpha^i), msg highest degree first."""
  m = np.array(msg)
  exponents = np.arange(nsym)[:, None] * (len(msg) - 1 - np.arange(len(msg)))
  return np.bitwise_xor.reduce(_EXP_TABLE[(_LOG_TABLE[m] + exponents) % 255] * (m != 0), axis=1).tolist()


def _rs_correct(msg: list[int], nsym: int) -> list[int]:
  """Corrects up to nsym // 2 errors in a Reed-Solomon codeword, in place."""
  n = len(msg)
  syn = _syndromes(msg, nsym)
  if not any(syn):
    return msg

  # Berlekamp-Massey, sigma is lowest degree first
  sigma, prev, L, m, b = [1], [1], 0, 1, 1
  for r in range(nsym):
    d = syn[r]
    for i in range(1, L + 1):
      d ^= _multiply(sigma[i], syn[r - i])
    if d == 0:
      m += 1
      continue
    coef = _multiply(d, _gf_inv(b))
    shifted = [0] * m + prev
    saved = sigma[:]
    sigma = sigma + [0] * max(0, len(shifted) - len(sigma))
    for i, c in enumerate(shifted):
      sigma[i] ^= _multiply(coef, c)
    if 2 * L <= r:
      L, prev, b, m = r + 1 - L, saved, d, 1
    else:
      m += 1
  sigma = sigma[:L + 1]
  if 2 * L > nsym:
    raise QRError("too many errors")

  # Chien search: codeword position p has locator alpha^(n-1-p)
  positions = [p for p in range(n) if _poly_eval(sigma[::-1], _EXP[(p - n + 1) % 255]) == 0]
  if len(positions) != L:
    raise QRError("error locator mismatch")

  # solve syn[i] = sum_k e_k * X_k^i for the magnitudes e_k
  xlog = [(n - 1 - p) % 255 for p in positions]
  A = [[_EXP[(xlog[k] * i) % 255] for k in range(L)] + [syn[i]] for i in range(L)]
  for col in range(L):
    piv = next((r for r in range(col, L) if A[r][col]), None)
    if piv is None:
      raise QRError("singular")
    A[col], A[piv] = A[piv], A[col]
    inv = _gf_inv(A[col][col])
    A[col] = [_multiply(inv, v) for v in A[col]]
    for r in range(L):
      if r != col and A[r][col]:
        f = A[r][col]
        A[r] = [a ^ _multiply(f, c) for a, c in zip(A[r], A[col], strict=True)]
  for k, p in enumerate(positions):
    msg[p] ^= A[k][L]

  if any(_syndromes(msg, nsym)):
    raise QRError("uncorrectable")
  return msg


def _read_format(m: np.ndarray) -> int:
  """Returns the closest format info (level bits << 3 | mask) from either copy."""
  dim = m.shape[0]
  copies = ([(8, i) for i in range(6)] + [(8, 7), (8, 8), (7, 8)] + [(5 - i, 8) for i in range(6)],
            [(dim - 1 - i, 8) for i in range(7)] + [(8, dim - 8 + i) for i in range(8)])  # (row, col), msb first
  candidates = []
  for coords in copies:
    bits = int("".join(str(int(m[r, c])) for r, c in coords), 2)
    candidates += [((bits ^ f).bit_count(), i) for i, f in enumerate(_FORMATS)]
  distance, fmt = min(candidates)
  if distance > 3:
    raise QRError("bad format info")
  return fmt


_ALNUM = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"

_ECI_ENCODINGS = {
  0: "cp437", 2: "cp437", 1: "iso8859-1", 3: "iso8859-1",
  **{i + 2: f"iso8859-{i}" for i in range(2, 17) if i != 12},
  20: "shift_jis", 21: "cp1250", 22: "cp1251", 23: "cp1252", 24: "cp1256",
  25: "utf-16-be", 26: "utf-8", 27: "ascii", 170: "ascii", 28: "big5", 29: "gb18030", 30: "euc_kr",
}


class _Bits:
  def __init__(self, data: list[int]):
    self._value = int.from_bytes(bytes(data), "big")
    self.remaining = len(data) * 8

  def read(self, n: int) -> int:
    if n > self.remaining:
      raise QRError("bitstream underflow")
    self.remaining -= n
    return self._value >> self.remaining & (1 << n) - 1

  def read_below(self, n: int, limit: int) -> int:
    v = self.read(n)
    if v >= limit:
      raise QRError("value out of range")
    return v


def _parse_data(data: list[int], version: int) -> str:
  bits = _Bits(data)
  out: list[str] = []
  encoding = None
  band = 0 if version <= 9 else 1 if version <= 26 else 2
  while bits.remaining >= 4:
    mode = bits.read(4)
    if mode == 0:
      break
    if mode == 7:  # ECI character set assignment
      first = bits.read(8)
      extra = 0 if first < 0x80 else 8 if first < 0xC0 else 16 if first < 0xE0 else -1  # 1, 2, or 3 byte assignment
      if extra < 0:
        raise QRError("bad ECI assignment")
      assignment = (first & 0x7F >> extra // 8) << extra | bits.read(extra)
      encoding = _ECI_ENCODINGS.get(assignment)
      if encoding is None:
        raise QRError(f"unsupported ECI assignment {assignment}")
    elif mode == 1:
      n = bits.read((10, 12, 14)[band])
      while n > 0:
        k = min(n, 3)  # 3 digits in 10 bits, the last 2 or 1 in 7 or 4
        out.append(f"{bits.read_below((4, 7, 10)[k - 1], 10 ** k):0{k}d}")
        n -= k
    elif mode == 2:
      n = bits.read((9, 11, 13)[band])
      while n > 0:
        k = min(n, 2)  # 2 characters in 11 bits, a last one in 6
        v = bits.read_below((6, 11)[k - 1], 45 ** k)
        out.append(_ALNUM[v // 45] * (k - 1) + _ALNUM[v % 45])
        n -= k
    elif mode == 4:
      n = bits.read((8, 16, 16)[band])
      segment = bytes(bits.read(8) for _ in range(n))
      try:
        out.append(segment.decode(encoding or "utf-8"))
      except UnicodeDecodeError as e:
        if encoding is not None:
          raise QRError("invalid ECI byte segment") from e
        out.append(segment.decode("latin-1"))
    elif mode == 8:
      n = bits.read((8, 10, 12)[band])
      for _ in range(n):
        v = bits.read(13)
        c = (v // 0xC0) << 8 | v % 0xC0
        c += 0x8140 if c < 0x1F00 else 0xC140
        try:
          out.append(c.to_bytes(2, "big").decode("shift_jis"))
        except UnicodeDecodeError as e:
          raise QRError("invalid Kanji character") from e
    else:
      raise QRError(f"unsupported mode {mode}")
  return "".join(out)


def decode_matrix(m: np.ndarray) -> str:
  """Decodes a square boolean module matrix (True = dark) without a quiet zone."""
  dim = m.shape[0]
  if m.shape != (dim, dim) or dim % 4 != 1 or not 21 <= dim <= 177:
    raise QRError("bad matrix size")
  version = (dim - 17) // 4

  fmt = _read_format(m)
  level = _LEVELS[fmt >> 3]
  rows, cols = _data_coords(version)
  bits = m[rows, cols] ^ _MASKS[fmt & 7](rows, cols)
  codewords = np.packbits(bits[:len(bits) // 8 * 8]).tolist()

  ec, _ = _EC[version - 1][level]
  lens = _block_lengths(version, level)
  blocks = [[0] * (n + ec) for n in lens]
  for (b, i), codeword in zip(_interleaved(version, level), codewords, strict=True):
    blocks[b][i] = codeword

  data: list[int] = []
  for block, n in zip(blocks, lens, strict=True):
    data += _rs_correct(block, ec)[:n]
  return _parse_data(data, version)


# ---- Image decoding ----


def _box_sums(a: np.ndarray, radii: tuple[int, ...]) -> list[np.ndarray]:
  """Sums over (2r + 1)^2 neighborhoods of the last two axes, edge padded, from one integral image."""
  P = max(radii)
  lead = [(0, 0)] * (a.ndim - 2)
  cs = np.pad(np.cumsum(np.cumsum(np.pad(a, lead + [(P, P), (P, P)], mode="edge"), -2), -1), lead + [(1, 0), (1, 0)])
  H, W = a.shape[-2:]
  out = []
  for r in radii:
    lo, hi = P - r, P + r + 1
    out.append(cs[..., hi:hi + H, hi:hi + W] - cs[..., lo:lo + H, hi:hi + W] - cs[..., hi:hi + H, lo:lo + W] + cs[..., lo:lo + H, lo:lo + W])
  return out


def _binarize(gray: np.ndarray) -> np.ndarray:
  """Adaptive threshold: each pixel against the mean of the surrounding tiles that have contrast."""
  h, w = gray.shape
  if h < 21 or w < 21:
    raise QRError("image too small")
  B = max(8, min(h, w) // 128 * 2)
  H, W = -(-h // B), -(-w // B)
  padded = np.pad(gray, ((0, H * B - h), (0, W * B - w)), mode="edge")
  # block statistics from a subsample are plenty
  sub = np.ascontiguousarray(padded[::2, ::2].reshape(H, B // 2, W, B // 2).transpose(0, 2, 1, 3)).reshape(H, W, -1)
  blocks = sub.sum(axis=2, dtype=np.uint32) / sub.shape[2]
  known = sub.max(axis=2) - sub.min(axis=2) >= 32
  # Flat tiles cannot estimate their own threshold: use the tiles with contrast nearby, then
  # further out, then the global midrange. A flat tile is then all dark or all light.
  est = np.full((H, W), (blocks.min() + blocks.max()) / 2)
  filled = np.zeros((H, W), dtype=bool)
  for total, count in _box_sums(np.stack((known * blocks, known.astype(float))), (2, 6)):
    fill = ~filled & (count > 0)
    est[fill] = total[fill] / count[fill]
    filled |= fill
  thr = np.where(known, np.minimum(est, 254) + 1, np.where(blocks <= est, 255, 0)).astype(np.uint8)
  return (padded.reshape(H, B, W, B) < thr[:, None, :, None]).reshape(H * B, W * B)[:h, :w]


class _Runs:
  """Run-length table of a padded, flattened binary image with a per-pixel run index."""

  def __init__(self, padded: np.ndarray):
    self.flat = padded.ravel()
    self.lines, self.stride = padded.shape
    change = self.flat[1:] != self.flat[:-1]
    self.starts = np.concatenate(([0], np.flatnonzero(change) + 1))
    self.lengths = np.diff(np.append(self.starts, self.flat.size)).astype(np.int32)

  def run_at(self, line: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Index of the run containing the pixel at `pos` along `line`."""
    return np.searchsorted(self.starts, line * self.stride + pos + 1, side="right") - 1

  @staticmethod
  def _match(lengths: list[np.ndarray], ratios: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Checks windows of runs against the ratios, given the length of each run. Returns (ok, module size)."""
    S = sum(ratios)
    total = sum(lengths[1:], start=lengths[0])
    ok = total >= 2 * S  # modules need to be at least 2 px
    for L, r in zip(lengths, ratios, strict=True):
      ok &= np.abs(2 * S * L - 2 * r * total) <= r * total  # integer form of |L - r * total / S| <= r * total / (2 * S)
    return ok, total / S

  def scan(self, ratios: tuple[int, ...]) -> np.ndarray:
    """Returns the indices of all dark runs starting a window of runs matching the ratios."""
    n = len(ratios)
    N = len(self.lengths) - n + 1
    if N <= 0:
      return np.zeros(0, dtype=int)
    ok, _ = self._match([self.lengths[k:N + k] for k in range(n)], ratios)
    ok &= self.flat[self.starts[:N]]
    first = np.flatnonzero(ok)
    return first[self.starts[first] // self.stride == self.starts[first + n - 1] // self.stride]

  def check(self, first: np.ndarray, ratios: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Checks the run windows starting at run index `first`. Returns (ok, center position along the line, module size)."""
    n, half = len(ratios), len(ratios) // 2
    ok = (first >= 0) & (first + n <= len(self.starts))
    idx = np.clip(first[:, None] + np.arange(n), 0, len(self.starts) - 1)
    matched, module = self._match([self.lengths[idx[:, k]] for k in range(n)], ratios)
    ok &= matched & self.flat[self.starts[idx[:, 0]]]
    ok &= self.starts[idx[:, 0]] // self.stride == self.starts[idx[:, -1]] // self.stride
    center = self.starts[idx[:, half]] % self.stride - 1 + self.lengths[idx[:, half]] / 2
    return ok, center, module


def _find_patterns(binary: np.ndarray, ratios: tuple[int, ...]) -> list[tuple[float, float, float]]:
  """Finds dark/light run patterns with the given module ratios. Returns (x, y, module size)."""
  half = len(ratios) // 2
  step = 2  # the center rows of a 2 px finder pattern still get scanned twice
  rows_t = _Runs(np.pad(binary[::step], ((0, 0), (1, 1))))
  first = rows_t.scan(ratios)
  if len(first) == 0:
    return []
  _, cx, hmod = rows_t.check(first, ratios)
  row = rows_t.starts[first] // rows_t.stride * step

  xi = cx.astype(int)
  xs, col = np.unique(xi, return_inverse=True)
  cols_t = _Runs(np.pad(binary[:, xs].T, ((0, 0), (1, 1))))
  ok, cy, vmod = cols_t.check(cols_t.run_at(col, row) - half, ratios)
  ok &= (0.5 <= vmod / hmod) & (vmod / hmod <= 2)
  line = np.clip(np.rint(cy / step), 0, rows_t.lines - 1).astype(int)
  ok2, cx2, hmod2 = rows_t.check(rows_t.run_at(line, xi) - half, ratios)
  ok &= ok2 & (0.5 <= hmod2 / vmod) & (hmod2 / vmod <= 2)

  found: list[list[float]] = []  # [x, y, module, count]
  for x, y, module in zip(cx2[ok], cy[ok], (hmod2[ok] + vmod[ok]) / 2, strict=True):
    for f in found:
      if abs(f[0] - x) <= f[2] and abs(f[1] - y) <= f[2] and 0.5 <= f[2] / module <= 2:
        c = f[3]
        f[0], f[1], f[2], f[3] = (f[0] * c + x) / (c + 1), (f[1] * c + y) / (c + 1), (f[2] * c + module) / (c + 1), c + 1
        break
    else:
      found.append([x, y, module, 1])
  found.sort(key=lambda f: -f[3])
  return [(f[0], f[1], f[2]) for f in found if f[3] >= 2]


def _pick_finders(patterns: list[tuple[float, float, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
  """Returns (top-left, top-right, bottom-left) centers and the module size of the most square-looking triple."""
  best = None
  for a, b, c in itertools.combinations(patterns[:10], 3):
    mods = sorted((a[2], b[2], c[2]))
    if mods[2] / mods[0] > 1.5:
      continue
    pts = [np.array(p[:2]) for p in (a, b, c)]
    d = [np.linalg.norm(pts[(i + 1) % 3] - pts[(i + 2) % 3]) for i in range(3)]
    tl = int(np.argmax(d))  # opposite the hypotenuse
    p1, p2 = pts[(tl + 1) % 3], pts[(tl + 2) % 3]
    v1, v2 = p1 - pts[tl], p2 - pts[tl]
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
      continue
    cos = abs(np.dot(v1, v2)) / (n1 * n2)
    if cos > 0.35 or not 0.6 <= n1 / n2 <= 1.6:
      continue
    score = cos + abs(np.log(n1 / n2)) + np.log(mods[2] / mods[0])
    if best is not None and score >= best[0]:
      continue
    if v1[0] * v2[1] - v1[1] * v2[0] < 0:
      p1, p2 = p2, p1
    best = (score, pts[tl], p1, p2, float(sum(mods) / 3))
  if best is None:
    raise QRError("no finder patterns")
  return best[1:]


def _perspective(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
  """Homography mapping the four src points onto the four dst points."""
  A = [row for (x, y), (u, v) in zip(src, dst, strict=True)
       for row in ([x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y])]
  try:
    h = np.linalg.solve(np.array(A, dtype=float), np.asarray(dst, dtype=float).ravel())
  except np.linalg.LinAlgError as e:
    raise QRError("degenerate geometry") from e
  return np.append(h, 1).reshape(3, 3)


def _transform(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
  p = np.column_stack((pts, np.ones(len(pts)))) @ H.T
  return p[:, :2] / p[:, 2:3]


def _match_alignment(binary: np.ndarray, est: np.ndarray, offs: np.ndarray, r: int, module: float) -> np.ndarray | None:
  h, w = binary.shape
  dy = np.arange(max(0, int(est[1]) - r), min(h, int(est[1]) + r)) - est[1]
  dx = np.arange(max(0, int(est[0]) - r), min(w, int(est[0]) + r)) - est[0]
  if len(dy) == 0 or len(dx) == 0:
    return None
  y = np.rint(est[1] + dy[:, None, None] + offs[None, None, :, 1]).astype(int)
  x = np.rint(est[0] + dx[None, :, None] + offs[None, None, :, 0]).astype(int)
  valid = ((y >= 0) & (y < h) & (x >= 0) & (x < w)).all(axis=2)
  samples = binary[np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)]
  score = np.where(valid, (samples == _ALIGNMENT.ravel()).sum(axis=2), 0)
  if score.max() < 23:
    return None
  hits = np.argwhere(score == score.max())
  centers = np.column_stack((est[0] + dx[hits[:, 1]], est[1] + dy[hits[:, 0]]))
  closest = centers[np.argmin(np.linalg.norm(centers - est, axis=1))]
  return centers[np.linalg.norm(centers - closest, axis=1) <= module / 2].mean(axis=0)


def _locate_alignment(binary: np.ndarray, H: np.ndarray, center: float, module: float) -> np.ndarray | None:
  """Template matches the 5x5 alignment pattern around its position estimated from H."""
  grid = np.mgrid[-2:3, -2:3].reshape(2, -1).T[:, ::-1] + center  # (25, 2) module coords (x, y)
  pts = _transform(H, grid)
  # the affine estimate can be off in both position and local scale under perspective
  for radius in (2, 4, 8, 16):
    for scale in (1.0, 0.8, 1.25, 0.65, 1.5):
      found = _match_alignment(binary, pts[12], (pts - pts[12]) * scale, int(module * radius), module)
      if found is not None:
        return found
  return None


def _sample(binary: np.ndarray, tl: np.ndarray, tr: np.ndarray, bl: np.ndarray, module: float, dim: int, use_alignment: bool) -> np.ndarray:
  src = np.array([(3.5, 3.5), (dim - 3.5, 3.5), (3.5, dim - 3.5), (dim - 3.5, dim - 3.5)])
  dst = np.array([tl, tr, bl, tr + bl - tl])
  H = _perspective(src, dst)
  if use_alignment and dim > 21:
    align = _locate_alignment(binary, H, dim - 6.5, module)
    if align is not None:
      src[3], dst[3] = (dim - 6.5, dim - 6.5), align
      H = _perspective(src, dst)

  rows, cols = np.mgrid[0:dim, 0:dim]
  pts = _transform(H, np.column_stack((cols.ravel() + 0.5, rows.ravel() + 0.5)))
  xy = np.rint(pts).astype(int)
  h, w = binary.shape
  if (xy < 0).any() or (xy[:, 0] >= w).any() or (xy[:, 1] >= h).any():
    raise QRError("code extends outside image")
  return binary[xy[:, 1], xy[:, 0]].reshape(dim, dim)


def decode(gray: np.ndarray) -> str | None:
  """Decodes the QR code in a 2D uint8 grayscale image. Modules need to be at least 2 px.
  Returns None if nothing could be decoded."""
  try:
    binary = _binarize(gray)
    tl, tr, bl, module = _pick_finders(_find_patterns(binary, (1, 1, 3, 1, 1)))
  except QRError:
    return None

  d = (np.linalg.norm(tr - tl) + np.linalg.norm(bl - tl)) / 2
  dim = int(round((d / module + 7 - 17) / 4)) * 4 + 17
  dims = [cand for cand in (dim, dim - 4, dim + 4) if 21 <= cand <= 177]
  for cand, use_alignment, transpose in itertools.product(dims, (True, False), (False, True)):
    try:
      m = _sample(binary, tl, tr, bl, module, cand, use_alignment)
      return decode_matrix(m.T if transpose else m)
    except QRError:
      pass
  return None
