import hashlib
import math
from pathlib import Path

import numpy as np

from openpilot.common import qrcode as qr
from openpilot.common.test import OpenpilotTestCase

LPA = "LPA:1$rsp.truphone.com$QRF-BETTERROAMING-PMRDGIR2EARDEIT5"


# Matrices generated with python-qrcode 8.2, covering all versions and EC levels.
# Packed fixtures keep the decoder tests independent of our encoder.
FIXTURES = {}
for path in Path(__file__).with_name("fixtures").glob("qrcode_*.npz"):
  with np.load(path) as fixtures:
    FIXTURES.update({key: fixtures[key] for key in fixtures.files})


def fixture(key: str) -> np.ndarray:
  bits = np.unpackbits(FIXTURES[key])
  size = math.isqrt(len(bits))
  return bits[:size * size].reshape(size, size).astype(bool)


def render(matrix: np.ndarray, box: int = 6, border: int = 4) -> np.ndarray:
  img = np.repeat(np.repeat(np.pad(matrix, border), box, axis=0), box, axis=1)
  return np.where(img, 0, 255).astype(np.uint8)


def make(data: str, version: int | None = None, level: int = 0, box: int = 6, border: int = 4):
  matrix = fixture(hashlib.sha256(f"{version}:{level}:{data}".encode()).hexdigest())
  return matrix, render(matrix, box, border)


def warp(img: np.ndarray, H: np.ndarray) -> np.ndarray:
  """Bilinear resampling through the output -> input homography H, white outside the image."""
  h, w = img.shape
  rows, cols = np.mgrid[0:h, 0:w]
  pts = qr._transform(H, np.column_stack((cols.ravel() + 0.5, rows.ravel() + 0.5))) - 0.5
  x0, y0 = np.floor(pts[:, 0]).astype(int), np.floor(pts[:, 1]).astype(int)
  fx, fy = pts[:, 0] - x0, pts[:, 1] - y0
  padded = np.pad(img.astype(float), 1, constant_values=255)

  def at(y, x):
    return padded[np.clip(y + 1, 0, h + 1), np.clip(x + 1, 0, w + 1)]

  out = at(y0, x0) * (1 - fx) * (1 - fy) + at(y0, x0 + 1) * fx * (1 - fy) + at(y0 + 1, x0) * (1 - fx) * fy + at(y0 + 1, x0 + 1) * fx * fy
  return np.clip(out, 0, 255).astype(np.uint8).reshape(h, w)


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
  h, w = img.shape
  t = np.radians(angle)
  R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
  center = np.array([w / 2, h / 2])
  corners = np.array([(0, 0), (w, 0), (w, h), (0, h)], dtype=float)
  return warp(img, qr._perspective((corners - center) @ R.T + center, corners))


class TestQRCode(OpenpilotTestCase):
  def test_alignment_positions(self):
    assert qr._alignment_positions(7) == [6, 22, 38]
    assert qr._alignment_positions(32) == [6, 34, 60, 86, 112, 138]
    assert qr._alignment_positions(40) == [6, 30, 58, 86, 114, 142, 170]

  def test_all_versions(self):
    for version in range(1, 41):
      for level in range(4):
        with self.subTest(version=version, level=level):
          data = "".join(chr(ord("a") + i % 26) for i in range(version))
          matrix, img = make(data, version, level, box=3)
          assert qr.decode_matrix(matrix) == data
          assert qr.decode(img) == data

  def test_modes(self):
    for data in ["0123456789012345", "HELLO WORLD $1.50", LPA, "こんにちは", "ünïcødé", "mixed 123 ABC xyz"]:
      with self.subTest(data=data):
        matrix, img = make(data)
        assert qr.decode_matrix(matrix) == data
        assert qr.decode(img) == data

  def test_error_correction(self):
    matrix, _ = make(LPA, level=2)
    rng = np.random.default_rng(0)
    flipped = matrix.copy()
    for r, c in rng.integers(9, matrix.shape[0] - 9, size=(40, 2)):
      flipped[r, c] ^= True
    assert qr.decode_matrix(flipped) == LPA

  def test_large_modules(self):
    for data in ["0123456789012345", "HELLO WORLD $1.50", LPA, "mixed 123 ABC xyz"]:
      for box in [16, 20, 24, 32]:
        for dark, light in [(0, 255), (60, 200), (140, 250)]:
          with self.subTest(data=data, box=box, dark=dark):
            _, img = make(data, box=box)
            img = np.where(img == 0, dark, light).astype(np.uint8)
            assert qr.decode(img) == data

  def test_image_edges(self):
    # a code touching the image edge must not lose the rows and columns left over from tiling
    matrix, _ = make(LPA)
    for size in (200, 203):
      with self.subTest(size=size):
        img = np.full((size, size), 255, dtype=np.uint8)
        code = render(matrix, box=5, border=0)
        img[size - code.shape[0]:, size - code.shape[1]:] = code
        assert qr.decode(img) == LPA

  def test_eci(self):
    # qrcode_eci.npz: packed Segno 1.6.6 matrices, generated with mode='byte',
    # eci=True, micro=False and the named encoding. Mixed also includes numeric,
    # alphanumeric, and Kanji segments after changing the byte encoding twice.
    cases = {
      "iso8859-5": "Привет", "utf-16-be": "héllo", "utf-8": "こんにちは",
      "shift_jis": "日本語", "cp1251": "Привет", "iso8859-1": "héllo",
      "mixed": "hélloПривет日本語123ABC漢字",
    }
    for encoding, expected in cases.items():
      with self.subTest(encoding=encoding):
        matrix = fixture(encoding)
        assert qr.decode_matrix(matrix) == expected
        assert qr.decode(render(matrix)) == expected

  def test_parse_data(self):
    def parse(stream: str) -> str:
      stream += '0' * (-len(stream) % 8)
      return qr._parse_data([int(stream[i:i + 8], 2) for i in range(0, len(stream), 8)], 1)

    def eci(assignment: str, payload: bytes = b'A') -> str:
      return parse('0111' + assignment + '0100' + f'{len(payload):08b}' + ''.join(f'{b:08b}' for b in payload) + '0000')

    # ASCII assignment 170 uses the two-byte ECI representation.
    assert eci('1000000010101010') == 'A'
    for assignment in ['00001110', '1000001111100111', '110000010000000000000000', '11100000']:
      with self.subTest(assignment=assignment), self.assertRaises(qr.QRError):
        eci(assignment)
    with self.assertRaises(qr.QRError):
      eci('00011010', b'\xff')  # Invalid UTF-8 must not fall back to Latin-1.

    # out-of-range numeric, alphanumeric, and Kanji values are format errors, not crashes
    for stream in ['0001' + '0000000011' + '1111111111', '0001' + '0000000010' + '1111111',
                   '0010' + '000000010' + '11111111111', '0010' + '000000001' + '111111',
                   '1000' + '00000001' + '0000000111111']:
      with self.subTest(stream=stream), self.assertRaises(qr.QRError):
        parse(stream)

  def test_rotation(self):
    for angle in [0, 90, 180, 270, 25, 110]:
      with self.subTest(angle=angle):
        _, img = make(LPA, box=8, border=12)
        assert qr.decode(rotate(img, angle)) == LPA

  def test_mirrored(self):
    _, img = make(LPA)
    assert qr.decode(img[:, ::-1]) == LPA

  def test_perspective_and_noise(self):
    _, img = make(LPA, box=10, border=8)
    h, w = img.shape
    corners = np.array([(40, 60), (w - 20, 30), (w - 60, h - 40), (30, h - 90)])
    arr = warp(img, qr._perspective(corners, np.array([(0, 0), (w, 0), (w, h), (0, h)]))).astype(float)
    rng = np.random.default_rng(1)
    arr = arr * 0.6 + 60 + rng.normal(0, 12, arr.shape)  # low contrast + noise
    # uneven lighting
    arr += np.linspace(-40, 40, w)[None, :]
    assert qr.decode(np.clip(arr, 0, 255).astype(np.uint8)) == LPA

  def test_no_code(self):
    rng = np.random.default_rng(2)
    assert qr.decode(rng.integers(0, 256, size=(240, 320), dtype=np.uint8)) is None
    assert qr.decode(np.full((240, 320), 200, dtype=np.uint8)) is None

  def test_encoder_roundtrip(self):
    for version in range(1, 21):
      with self.subTest(version=version):
        assert qr.decode_matrix(np.array(qr._Qr(version, b"hello").modules)) == "hello"
