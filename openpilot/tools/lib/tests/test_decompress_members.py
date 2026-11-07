import bz2
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import zstandard as zstd

from openpilot.tools.lib.file_downloader import decompress_chunk, decompress_file, make_decompressor


class TestDecompressMembers(unittest.TestCase):
  def test_bz2_members(self):
    self.check_members("bz2", bz2.compress)

  def test_zstd_members(self):
    self.check_members("zst", zstd.ZstdCompressor().compress)

  def check_members(self, kind, compress):
    first, second = compress(b"one"), compress(b"two")
    with TemporaryDirectory() as directory:
      source, output = Path(directory) / "input", Path(directory) / "output"
      source.write_bytes(first + second)
      decompress_file(source, output)
      self.assertEqual(output.read_bytes(), b"onetwo")

      for cut in (len(first) - 1, len(first), len(first) + 1):
        buffer = io.BytesIO()
        decoder = make_decompressor(kind)
        data = first + second
        for chunk in (data[:cut], data[cut:]):
          decoder = decompress_chunk(decoder, kind, chunk, buffer)
        self.assertTrue(decoder.eof)
        self.assertEqual(buffer.getvalue(), b"onetwo")

      source.write_bytes(first + second[:-1])
      with self.assertRaises(EOFError):
        decompress_file(source, output)
