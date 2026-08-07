import asyncio
import json
import time

import capnp
from openpilot.common.test import OpenpilotTestCase
from openpilot.cereal import messaging, log
from teleoprtc.tracks import VIDEO_CLOCK_RATE

from openpilot.system.webrtc.webrtcd import CerealOutgoingMessageProxy, CerealIncomingMessageProxy, StreamSession
from openpilot.system.webrtc.device.video import LiveStreamVideoStreamTrack


class TestStreamSession(OpenpilotTestCase):
  def setup_method(self):
    self.loop = asyncio.new_event_loop()

  def teardown_method(self):
    self.loop.stop()
    self.loop.close()

  def test_outgoing_proxy(self, mocker):
    test_msg = log.Event.new_message()
    test_msg.logMonoTime = 123
    test_msg.valid = True
    test_msg.customReservedRawData0 = b"test"
    expected_dict = {"type": "customReservedRawData0", "logMonoTime": 123, "valid": True, "data": "test"}
    expected_json = json.dumps(expected_dict).encode()

    channel = mocker.Mock()
    channel.is_open.return_value = True
    proxy = CerealOutgoingMessageProxy(["customReservedRawData0"])
    def mocked_update(t):
      proxy.sm.update_msgs(0, [test_msg])

    mocker.patch.object(messaging.SubMaster, "update", side_effect=mocked_update)
    proxy.add_channel(channel)

    proxy.update()

    channel.send.assert_called_once_with(expected_json)

  def test_incoming_proxy(self, mocker):
    tested_msgs = [
      {"type": "customReservedRawData0", "data": "test"}, # primitive
      {"type": "can", "data": [{"address": 0, "dat": "", "src": 0}]}, # list
      {"type": "testJoystick", "data": {"axes": [0, 0], "buttons": [False]}}, # dict
    ]

    mocked_pubmaster = mocker.MagicMock(spec=messaging.PubMaster)

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

  def test_speaker_volume(self, mocker):
    session = StreamSession.__new__(StreamSession)
    session.logger = mocker.Mock()
    session.incoming_bridge_services = []
    session.incoming_bridge = None
    session.speaker_volume_pm = mocker.Mock()

    for volume in (0, 42, 100):
      session.message_handler(json.dumps({"type": "speakerVolume", "data": {"volume": volume}}).encode())

    assert session.speaker_volume_pm.send.call_count == 3
    for expected, call in zip((0, 42, 100), session.speaker_volume_pm.send.call_args_list, strict=True):
      service, msg = call.args
      assert service == "speakerVolume"
      assert msg.speakerVolume.volume == expected

  def test_speaker_volume_validation(self, mocker):
    session = StreamSession.__new__(StreamSession)
    session.logger = mocker.Mock()
    session.incoming_bridge_services = []
    session.incoming_bridge = None
    session.speaker_volume_pm = mocker.Mock()

    for volume in (-1, 101, 50.0, True):
      session.message_handler(json.dumps({"type": "speakerVolume", "data": {"volume": volume}}).encode())

    session.message_handler(json.dumps({"type": "speakerVolume", "data": 50}).encode())

    session.speaker_volume_pm.send.assert_not_called()
    assert session.logger.exception.call_count == 5

  def test_livestream_track(self, mocker):
    fake_msg = messaging.new_message("livestreamDriverEncodeData")

    config = {"receive.return_value": fake_msg.to_bytes()}
    mocker.patch("msgq.SubSocket", spec=True, **config)
    track = LiveStreamVideoStreamTrack("driver")

    assert track.id.startswith("driver")

    for i in range(5):
      packet = self.loop.run_until_complete(track.recv())
      if i == 0:
        start_ns = time.monotonic_ns()
        start_pts = packet.pts
      assert abs(i + packet.pts - (start_pts + (((time.monotonic_ns() - start_ns) * VIDEO_CLOCK_RATE) // 1_000_000_000))) < 450 #5ms
      assert bytes(packet) == b""
