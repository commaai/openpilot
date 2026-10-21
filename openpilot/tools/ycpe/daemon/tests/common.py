import http.client
import json
import threading
import time
import unittest

from openpilot.tools.ycpe.daemon import core, server
from openpilot.tools.ycpe.daemon.core import config


def until(cond, timeout=10.0):
  """Waits for cond() to hold, up to timeout s."""
  end = time.monotonic() + timeout
  while not cond() and time.monotonic() < end:
    time.sleep(0.01)


class DaemonCase(unittest.TestCase):
  """A daemon on a free port for the test class, and requests to it as the page makes them."""
  @classmethod
  def setUpClass(cls):
    server.Handler.log_message = lambda *a: None
    cls.srv = server.make_server(0)
    config.port = cls.srv.server_address[1]
    threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

  @classmethod
  def tearDownClass(cls):
    cls.srv.shutdown()
    cls.srv.server_close()

  def connect(self, path, method="GET", host=None, headers=None, token=True, body=None):
    c = http.client.HTTPConnection("127.0.0.1", config.port, timeout=60)  # transcodes stream for a while
    self.addCleanup(c.close)
    h = {"Host": host or f"localhost:{config.port}", **(headers or {})}
    if token:
      h[core.TOKEN_HEADER] = config.token
    c.request(method, path, body=body, headers=h)
    return c.getresponse()

  def req(self, path, **kw):
    r = self.connect(path, **kw)
    return r, r.read()

  def json(self, path, body=None, method=None, status=200):
    """GET, or POST of body as JSON (bytes are sent as is), expecting status."""
    r, data = self.req(path, method=method or ("GET" if body is None else "POST"),
                       body=body if body is None or isinstance(body, bytes) else json.dumps(body))
    self.assertEqual(r.status, status, data)
    return json.loads(data)

  def read_until(self, r, marker: bytes) -> bytes:
    """An SSE response's bytes up to and including marker."""
    seen = b""
    while marker not in seen:
      seen += r.read1(4096)
    return seen
