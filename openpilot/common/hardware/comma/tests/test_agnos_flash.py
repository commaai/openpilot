import hashlib
import http.server
import struct
import subprocess
import threading
from unittest.mock import patch

import pytest
import requests

from openpilot.common.hardware.comma import agnos

BLOCK_SIZE = 4096


def sparse_image(chunks: list[tuple[int, bytes, int]]) -> tuple[bytes, bytes]:
  """builds an android sparse image from (type, payload, blocks) chunks, returns (sparse, raw)"""
  header = struct.pack("<IHHHHIIII", 0xed26ff3a, 1, 0, 28, 12, BLOCK_SIZE, sum(c[2] for c in chunks), len(chunks), 0)
  sparse, raw = [header], []
  for chunk_type, payload, blocks in chunks:
    sparse.append(struct.pack("<H2xI4x", chunk_type, blocks) + payload)
    raw.append(payload if chunk_type == 0xcac1 else payload * (blocks * BLOCK_SIZE // 4))
  return b"".join(sparse), b"".join(raw)


class Server:
  def __init__(self, body: bytes, truncate: bool = False):
    class Handler(http.server.BaseHTTPRequestHandler):
      def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body[:len(body) // 2] if truncate else body)

      def log_message(self, *args):
        pass

    self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    self.url = f"http://127.0.0.1:{self.httpd.server_port}/image.xz"
    threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

  def close(self):
    self.httpd.shutdown()
    self.httpd.server_close()


def flash(tmp_path, image: bytes, raw: bytes, sparse: bool, truncate: bool = False) -> list[int]:
  # multiple xz blocks, like the real images, so xz decompresses with threads
  compressed = subprocess.check_output(["xz", "--threads=4", "--block-size=1MiB", "--stdout"], input=image)
  server = Server(compressed, truncate)
  partition = {"name": "system", "url": server.url, "size": len(raw), "sparse": sparse,
               "hash": hashlib.sha256(image).hexdigest(), "hash_raw": hashlib.sha256(raw).hexdigest()}
  progress: list[int] = []
  try:
    with patch.object(agnos, "get_partition_path", return_value=str(tmp_path / "system")), patch.object(agnos.os, "sync"):
      agnos.extract_compressed_image(1, partition, None, progress.append)
  finally:
    server.close()
  return progress


def test_sparse_image(tmp_path):
  big_raw = bytes(range(256)) * (BLOCK_SIZE // 256) * 700  # bigger than one read
  image, raw = sparse_image([
    (0xcac1, big_raw, len(big_raw) // BLOCK_SIZE),
    (0xcac2, b"\xab\xcd\xef\x01", 600),  # fill
    (0xcac1, b"\x5a" * BLOCK_SIZE, 1),
  ])
  progress = flash(tmp_path, image, raw, sparse=True)
  assert (tmp_path / "system").read_bytes() == raw
  assert progress == sorted(progress) and progress[-1] == len(raw)
  assert all(b - a <= agnos.CHUNK_SIZE for a, b in zip([0, *progress], progress, strict=False))


def test_raw_image(tmp_path):
  raw = hashlib.sha256(b"seed").digest() * 100_000
  flash(tmp_path, raw, raw, sparse=False)
  assert (tmp_path / "system").read_bytes() == raw


def test_interrupted_download_is_retryable(tmp_path):
  raw = hashlib.sha256(b"seed").digest() * 100_000
  with pytest.raises(requests.exceptions.RequestException):
    flash(tmp_path, raw, raw, sparse=False, truncate=True)
