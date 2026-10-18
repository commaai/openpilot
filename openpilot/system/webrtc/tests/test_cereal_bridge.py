import asyncio
import json
import random

from openpilot.common.test import OpenpilotTestCase
from openpilot.system.webrtc.cereal_bridge import Unpacker, pack
from openpilot.system.webrtc.webrtcd import ServerState, handle_bridge


class TestCerealBridge(OpenpilotTestCase):
  def test_unpack_any_chunking(self):
    events = [random.randbytes(random.randint(0, 5000)) for _ in range(100)]
    stream = b"".join(pack(e) for e in events)

    for _ in range(10):
      unpacker = Unpacker()
      out, pos = [], 0
      while pos < len(stream):
        n = random.randint(1, 3000)
        out += unpacker.feed(stream[pos:pos + n])
        pos += n
      assert out == events
      assert len(unpacker.buf) == 0

  def test_bridge_request_policy(self):
    def request(can_send, services_out=(), services_in=()):
      body = json.dumps({"sdp": "", "services_out": list(services_out), "services_in": list(services_in)}).encode()
      status, resp, _ = asyncio.run(handle_bridge(ServerState(), body, "application/json", can_send))
      return status, json.loads(resp)

    assert request(True, services_out=["notAService"])[0] == 400
    assert request(False, services_in=["testJoystick"])[0] == 403
    assert request(True, services_in=["carControl"])[0] == 403
