#!/usr/bin/env python3
"""The /live/local signaling forward, against a stand-in webrtcd."""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from openpilot.tools.ycpe.daemon import webrtc
from openpilot.tools.ycpe.daemon.core import config
from openpilot.tools.ycpe.daemon.tests.common import DaemonCase

OFFER = {"sdp": "v=0\r\n", "cameras": ["road"], "enabled": True, "bridge_services_out": ["carState"]}
STREAM = "/local/webrtc/stream"


class Webrtcd(BaseHTTPRequestHandler):
  seen: list = []

  def do_POST(self):
    body = self.rfile.read(int(self.headers["Content-Length"]))
    Webrtcd.seen.append((self.path, self.headers.get_content_type(), json.loads(body), self.headers.get("X-Ycpe-Token")))
    out = json.dumps({"sdp": "answer", "type": "answer"}).encode()
    self.send_response(200)
    self.send_header("Content-Length", str(len(out)))
    self.end_headers()
    self.wfile.write(out)

  def log_message(self, *args):
    pass


class TestWebrtc(DaemonCase):
  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.webrtcd = ThreadingHTTPServer(("127.0.0.1", 0), Webrtcd)
    threading.Thread(target=cls.webrtcd.serve_forever, daemon=True).start()

  @classmethod
  def tearDownClass(cls):
    cls.webrtcd.shutdown()
    cls.webrtcd.server_close()
    super().tearDownClass()

  def setUp(self):
    config.webrtcd_port = self.webrtcd.server_address[1]
    Webrtcd.seen.clear()

  def test_forwards(self):
    r, data = self.post(STREAM, OFFER)
    self.assertEqual((r.status, json.loads(data)), (200, {"sdp": "answer", "type": "answer"}))
    self.assertEqual(Webrtcd.seen, [("/stream", "application/json", OFFER, None)])  # the session token stays here

  def test_needs_an_offer(self):
    for body in ([OFFER], {"sdp": 1}, {}, "x"):
      r, _ = self.post(STREAM, body)
      self.assertEqual(r.status, 400, body)
    r, _ = self.post(STREAM, b"{not json")
    self.assertEqual(r.status, 400)
    r, _ = self.post(STREAM, {"sdp": "x" * webrtc.MAX_BODY})
    self.assertEqual(r.status, 413)
    self.assertEqual(Webrtcd.seen, [])

  def test_no_webrtcd(self):
    config.webrtcd_port = 1  # nothing listens there
    r, data = self.post(STREAM, OFFER)
    self.assertEqual(r.status, 502)
    self.assertIn("webrtcd on 127.0.0.1:1", json.loads(data)["error"])

  def test_post_only(self):
    r, _ = self.req(STREAM)
    self.assertEqual(r.status, 405)
    self.assertEqual(Webrtcd.seen, [])


if __name__ == "__main__":
  unittest.main()
