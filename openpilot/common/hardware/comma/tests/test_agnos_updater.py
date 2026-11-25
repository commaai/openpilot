import json
import os
import requests

TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(TEST_DIR, "../agnos.json")


from openpilot.common.test import OpenpilotTestCase
class TestAgnosUpdater(OpenpilotTestCase):

  def test_manifest(self):
    with open(MANIFEST) as f:
      m = json.load(f)

    for img in m:
      r = requests.head(img['url'], timeout=10)
      r.raise_for_status()
      assert r.headers['Content-Type'] == "application/x-xz"
      if not img['sparse']:
        assert img['hash'] == img['hash_raw']

  def test_resume_after_dropped_connection(self):
    import lzma
    import threading
    from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
    from unittest import mock
    from openpilot.common.hardware.comma.agnos import StreamingDecompressor

    raw = os.urandom(3 * 1024 * 1024)
    blob = lzma.compress(raw)
    ranges = []

    class Handler(BaseHTTPRequestHandler):
      def log_message(self, *args):
        pass

      def do_GET(self):
        start = int(self.headers.get('Range', 'bytes=0-')[6:].rstrip('-'))
        ranges.append(start)
        self.send_response(206 if start else 200)
        self.send_header('Content-Length', str(len(blob) - start))
        self.end_headers()
        # drop the connection partway through the first two requests
        end = len(blob) if len(ranges) > 2 else start + len(blob) // 3
        self.wfile.write(blob[start:end])

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
      with mock.patch('openpilot.common.hardware.comma.agnos.time.sleep'):
        d = StreamingDecompressor(f"http://127.0.0.1:{server.server_address[1]}/img.xz")
        out = b''.join(iter(lambda: d.read(1024 * 1024), b''))
    finally:
      server.shutdown()

    assert out == raw
    assert len(ranges) == 3 and 0 == ranges[0] < ranges[1] < ranges[2], ranges
