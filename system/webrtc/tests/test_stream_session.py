# 1.a test if messages are proxied out (CerealOutgoingProxy)
# 1.b test out-proxy message failures (...)
# 2.a test if messages are proxied in (stream session message handle )
# 2.b test in-proxy message failures (message format invalid, no type and data fields)
# 3. test cleanup
# 4. test video and audio device tracks

import asyncio
import unittest
from unittest.mock import Mock, MagicMock, patch
import json

from aiortc import RTCDataChannel
from aiortc.mediastreams import VIDEO_CLOCK_RATE, VIDEO_TIME_BASE
import capnp

from cereal import messaging, log

from openpilot.system.webrtc.webrtcd import CerealOutgoingMessageProxy, CerealIncomingMessageProxy
from openpilot.system.webrtc.device.video import LiveStreamVideoStreamTrack
from openpilot.common.realtime import DT_DMON


class TestCerealProxy(unittest.TestCase):
  def test_outgoing_proxy(self):
    test_msg = log.Event.new_message()
    test_msg.logMonoTime = 123
    test_msg.valid = True
    test_msg.customReservedRawData0 = b"test"
    expected_dict = {"type": "customReservedRawData0", "logMonoTime": 123, "valid": True, "data": "test"}
    expected_json = json.dumps(expected_dict).encode()

    channel = Mock(spec=RTCDataChannel)
    mocked_submaster = messaging.SubMaster(["customReservedRawData0"])
    def mocked_update(t):
      mocked_submaster.update_msgs(0, [test_msg])

    with patch.object(messaging.SubMaster, "update", side_effect=mocked_update):
      proxy = CerealOutgoingMessageProxy(mocked_submaster)
      proxy.add_channel(channel)

      proxy.update()

      channel.send.assert_called_once_with(expected_json)

  def test_incoming_proxy(self):
    tested_msgs = [
      {"type": "customReservedRawData0", "data": "test"}, # primitive
      {"type": "can", "data": [{"address": 0, "busTime": 0, "dat": "", "src": 0}]}, # list
      {"type": "testJoystick", "data": {"axes": [0, 0], "buttons": [False]}}, # dict
    ]

    mocked_pubmaster = MagicMock(spec=messaging.PubMaster)

    proxy = CerealIncomingMessageProxy(mocked_pubmaster)

    for msg in tested_msgs:
      proxy.send(json.dumps(msg).encode())

      mocked_pubmaster.send.assert_called_once()
      mt, md = mocked_pubmaster.send.call_args.args
      self.assertEqual(mt, msg["type"])
      self.assertIsInstance(md, capnp._DynamicStructBuilder)
      self.assertTrue(hasattr(md, msg["type"]))

      mocked_pubmaster.reset_mock()

  def test_livestream_track(self):
    fake_msg = messaging.new_message("livestreamDriverEncodeData")

    config = {"receive.return_value": fake_msg.to_bytes()}
    with patch("cereal.messaging.SubSocket", spec=True, **config):
      track = LiveStreamVideoStreamTrack("driver")

      self.assertTrue(track.id.startswith("driver"))
      self.assertEqual(track.codec_preference(), "H264")

      for i in range(5):
        packet = asyncio.get_event_loop().run_until_complete(track.recv())
        self.assertEqual(packet.time_base, VIDEO_TIME_BASE)
        self.assertEqual(packet.pts, int(i * DT_DMON * VIDEO_CLOCK_RATE))
        self.assertEqual(packet.size, 0)


if __name__ == "__main__":
  unittest.main()