import asyncio
import json
import time
from unittest import mock

import capnp
from openpilot.cereal import messaging, log
from teleoprtc.tracks import VIDEO_CLOCK_RATE

from openpilot.selfdrive.test.helpers import OpenpilotTestCase
from openpilot.system.webrtc.webrtcd import CerealOutgoingMessageProxy, CerealIncomingMessageProxy
from openpilot.system.webrtc.device.video import LiveStreamVideoStreamTrack


class TestStreamSession(OpenpilotTestCase):
  def setUp(self):
    super().setUp()
    self.loop = asyncio.new_event_loop()

  def tearDown(self):
    self.loop.stop()
    self.loop.close()

  def test_outgoing_proxy(self):
    test_msg = log.Event.new_message()
    test_msg.logMonoTime = 123
    test_msg.valid = True
    test_msg.customReservedRawData0 = b"test"
    expected_dict = {"type": "customReservedRawData0", "logMonoTime": 123, "valid": True, "data": "test"}
    expected_json = json.dumps(expected_dict).encode()

    channel = mock.Mock()
    channel.is_open.return_value = True
    proxy = CerealOutgoingMessageProxy(["customReservedRawData0"])
    def mocked_update(t):
      proxy.sm.update_msgs(0, [test_msg])

    self.enterContext(mock.patch.object(messaging.SubMaster, "update", side_effect=mocked_update))
    proxy.add_channel(channel)

    proxy.update()

    channel.send.assert_called_once_with(expected_json)

  def test_incoming_proxy(self):
    tested_msgs = [
      {"type": "customReservedRawData0", "data": "test"}, # primitive
      {"type": "can", "data": [{"address": 0, "dat": "", "src": 0}]}, # list
      {"type": "testJoystick", "data": {"axes": [0, 0], "buttons": [False]}}, # dict
    ]

    mocked_pubmaster = mock.MagicMock(spec=messaging.PubMaster)

    proxy = CerealIncomingMessageProxy(mocked_pubmaster)

    for msg in tested_msgs:
      proxy.send(json.dumps(msg).encode())

      mocked_pubmaster.send.assert_called_once()
      mt, md = mocked_pubmaster.send.call_args.args
      msg_type = msg["type"]
      assert isinstance(msg_type, str)
      assert mt == msg_type
      assert isinstance(md, capnp._DynamicStructBuilder)
      assert hasattr(md, msg_type)

      mocked_pubmaster.reset_mock()

  def test_livestream_track(self):
    fake_msg = messaging.new_message("livestreamDriverEncodeData")

    config = {"receive.return_value": fake_msg.to_bytes()}
    self.enterContext(mock.patch("msgq.SubSocket", spec=True, **config))
    track = LiveStreamVideoStreamTrack("driver")

    assert track.id.startswith("driver")

    for i in range(5):
      packet = self.loop.run_until_complete(track.recv())
      if i == 0:
        start_ns = time.monotonic_ns()
        start_pts = packet.pts
      assert abs(i + packet.pts - (start_pts + (((time.monotonic_ns() - start_ns) * VIDEO_CLOCK_RATE) // 1_000_000_000))) < 450 #5ms
      assert bytes(packet) == b""
