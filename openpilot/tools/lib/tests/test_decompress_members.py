import bz2
import io

import pytest
import zstandard as zstd

from openpilot.tools.lib.file_downloader import decompress_chunk, decompress_file, make_decompressor


@pytest.mark.parametrize(("kind", "compress"), [("bz2", bz2.compress), ("zst", zstd.ZstdCompressor().compress)])
def test_concatenated_members(tmp_path, kind, compress):
  first, second = compress(b"one"), compress(b"two")
  source, output = tmp_path / "input", tmp_path / "output"
  source.write_bytes(first + second)
  decompress_file(source, output)
  assert output.read_bytes() == b"onetwo"

  for cut in (len(first) - 1, len(first), len(first) + 1):
    buffer = io.BytesIO()
    decoder = make_decompressor(kind)
    data = first + second
    for chunk in (data[:cut], data[cut:]):
      decoder = decompress_chunk(decoder, kind, chunk, buffer)
    assert decoder.eof and buffer.getvalue() == b"onetwo"

  source.write_bytes(first + second[:-1])
  with pytest.raises(EOFError):
    decompress_file(source, output)
