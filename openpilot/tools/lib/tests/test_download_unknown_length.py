import contextlib
import io
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock, patch

from openpilot.tools.lib.file_downloader import cmd_download
from openpilot.tools.lib.url_file import URLFile


class TestUnknownDownloadLength(unittest.TestCase):
  def test_reads_stream_to_decide_if_empty(self):
    args = Namespace(url="https://example.com/route", no_cache=True)
    for chunks in ([b"abc", b"def"], []):
      with self.subTest(chunks=chunks), tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, COMMA_CACHE=tmp):
        response = Mock(status=200, headers={"Transfer-Encoding": "chunked"})
        response.stream.return_value = iter(chunks)
        with patch.object(URLFile, "pool_manager") as pool, contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
          pool.return_value.request.return_value = response
          if chunks:
            cmd_download(args)
            self.assertEqual(Path(out.getvalue().strip()).read_bytes(), b"abcdef")
          else:
            with self.assertRaises(SystemExit):
              cmd_download(args)
            self.assertEqual(list(Path(tmp).iterdir()), [])
        response.release_conn.assert_called_once()
