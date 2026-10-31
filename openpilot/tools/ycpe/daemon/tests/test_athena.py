#!/usr/bin/env python3
"""The athena and billing relays, against a stand-in upstream: nothing here reaches either."""
import json
import unittest
from unittest import mock

from openpilot.tools.ycpe.daemon import athena
from openpilot.tools.ycpe.daemon.core import config
from openpilot.tools.ycpe.daemon.tests.common import DaemonCase

DONGLE = "0123456789abcdef"
AUTH = {"Authorization": "JWT a.b.c"}


class FakeResponse:
  def __init__(self, data, status=200):
    self.data, self.status = data, status

  def read(self, n):
    return self.data[:n]

  def release_conn(self):
    pass


class RelayCase(DaemonCase):
  """a stand-in upstream for the relays"""
  def setUp(self):
    self.pool = mock.Mock()
    self.pool.request.return_value = FakeResponse(b'{"id": 0, "jsonrpc": "2.0", "result": []}')
    patcher = mock.patch.object(athena.URLFile, "pool_manager", return_value=self.pool)
    patcher.start()
    self.addCleanup(patcher.stop)


class TestAthena(RelayCase):

  def call(self, body, dongle=DONGLE, headers=AUTH, **kw):
    return self.post(f"/local/athena/{dongle}", body, headers, **kw)

  def test_relays(self):
    r, data = self.call({"id": 0, "jsonrpc": "2.0", "method": "cancelUpload", "params": {"upload_id": ["x"]}, "extra": 1}, headers={
      **AUTH, "Origin": f"http://localhost:{config.port}", "Cookie": "c=1"})
    self.assertEqual((r.status, json.loads(data)["result"]), (200, []))
    (method, url), kw = self.pool.request.call_args
    self.assertEqual((method, url), ("POST", f"https://athena.comma.ai/{DONGLE}"))
    self.assertEqual(kw["headers"], {**AUTH, "Content-Type": "application/json"})  # nothing else from the page
    self.assertEqual(json.loads(kw["body"]), {"id": 0, "jsonrpc": "2.0", "method": "cancelUpload", "params": {"upload_id": ["x"]}})
    self.assertFalse(kw["redirect"])

  def test_clips(self):
    # the clip menu's calls; a chunk (512 KB on the device) comes back as base64 in the answer
    chunk = json.dumps({"id": 0, "jsonrpc": "2.0", "result": {"data": "A" * (700 << 10), "offset": 0, "size": 1 << 20}}).encode()
    self.pool.request.return_value = FakeResponse(chunk)
    for method in ("getVersion", "getClipState", "createClip", "deleteClip", "getClipChunk"):
      r, data = self.call({"method": method, "params": {}})
      self.assertEqual((r.status, data), (200, chunk), method)
      self.assertEqual(json.loads(self.pool.request.call_args[1]["body"])["method"], method)

  def test_upstream_status(self):
    self.pool.request.return_value = FakeResponse(b'{"error": "unauthorized"}', 401)
    r, _ = self.call({"method": "listUploadQueue"})
    self.assertEqual(r.status, 401)
    self.pool.request.return_value = FakeResponse(b"x" * (athena.MAX_RESPONSE + 1))
    r, _ = self.call({"method": "listUploadQueue"})
    self.assertEqual(r.status, 502)

  def test_allowlist(self):
    for body in ({"method": "reboot"}, {"method": "setNavDestination"}, {"method": "startLocalProxy", "params": {}}, {"method": None}, {},
                 [{"method": "listUploadQueue"}], "listUploadQueue"):
      r, _ = self.call(body)
      self.assertEqual(r.status, 403, body)
    r, _ = self.call(b"{not json")
    self.assertEqual(r.status, 400)
    r, _ = self.call(json.dumps({"method": "uploadFilesToUrls", "params": {"files_data": ["x" * athena.MAX_BODY]}}).encode())
    self.assertEqual(r.status, 413)
    self.pool.request.assert_not_called()

  def test_body_limit_before_reading(self):
    # a body that never comes: refused on its Content-Length, not after waiting to read it
    r, _ = self.call(b"{}", headers={**AUTH, "Content-Length": str(athena.MAX_BODY + 1)})
    self.assertEqual(r.status, 413)
    self.pool.request.assert_not_called()

  def test_bad_dongle(self):
    for dongle in ("0123456789ABCDEF", "0123456789abcde", "0123456789abcdef0", "..%2f..%2fv1", "0123456789abcdeg", "%2e%2e"):
      r, _ = self.call({"method": "listUploadQueue"}, dongle=dongle)
      self.assertIn(r.status, (400, 404), dongle)
    r, _ = self.call({"method": "listUploadQueue"}, dongle=f"{DONGLE}/x")
    self.assertEqual(r.status, 404)
    self.pool.request.assert_not_called()

  def test_needs_jwt(self):
    for headers in ({}, {"Authorization": "Bearer x"}, {"Authorization": "JWT a\tb"}):
      r, _ = self.call({"method": "listUploadQueue"}, headers=headers)
      self.assertEqual(r.status, 401, headers)
    self.pool.request.assert_not_called()

  def test_post_only(self):
    r, _ = self.req(f"/local/athena/{DONGLE}", headers=AUTH)
    self.assertEqual(r.status, 405)


class TestBilling(RelayCase):
  """the referrals read, on the same forwarding"""
  def test_relays(self):
    self.pool.request.return_value = FakeResponse(b'{"code": "ABC"}')
    r, data = self.req("/local/billing/referrals", headers={**AUTH, "Origin": f"http://localhost:{config.port}", "Cookie": "c=1"})
    self.assertEqual((r.status, data), (200, b'{"code": "ABC"}'))
    (method, url), kw = self.pool.request.call_args
    self.assertEqual((method, url, kw["headers"], kw["body"]), ("GET", "https://billing.comma.ai/v1/referrals", AUTH, None))

  def test_only_referrals(self):
    for path, method in (("/local/billing/referrals", "POST"), ("/local/billing/prime/stripe_checkout", "POST"),
                         ("/local/billing/prime/stripe_portal", "GET"), ("/local/billing/referrals/x", "GET")):
      r, _ = self.req(path, method=method, body=b"{}" if method == "POST" else None, headers=AUTH)
      self.assertIn(r.status, (404, 405), path)
    self.pool.request.assert_not_called()

  def test_needs_jwt(self):
    for headers in ({}, {"Authorization": "Bearer x"}):
      r, _ = self.req("/local/billing/referrals", headers=headers)
      self.assertEqual(r.status, 401, headers)
    self.pool.request.assert_not_called()


if __name__ == "__main__":
  unittest.main()
