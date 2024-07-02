import asyncio
import json
# for aiortc and its dependencies
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning) # TODO: remove this when google-crc32c publish a python3.12 wheel

from aiortc import RTCDataChannel
from aiortc.mediastreams import VIDEO_CLOCK_RATE, VIDEO_TIME_BASE
import capnp
import pyaudio
from cereal import messaging, log

from openpilot.system.webrtc.webrtcd import CerealOutgoingMessageProxy, CerealIncomingMessageProxy
from openpilot.system.webrtc.device.video import LiveStreamVideoStreamTrack
from openpilot.system.webrtc.device.audio import AudioInputStreamTrack
from openpilot.common.realtime import DT_DMON


class TestStreamSession:
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

    channel = mocker.Mock(spec=RTCDataChannel)
    mocked_submaster = messaging.SubMaster(["customReservedRawData0"])
    def mocked_update(t):
      mocked_submaster.update_msgs(0, [test_msg])

    mocker.patch.object(messaging.SubMaster, "update", side_effect=mocked_update)
    proxy = CerealOutgoingMessageProxy(mocked_submaster)
    proxy.add_channel(channel)

    proxy.update()

    channel.send.assert_called_once_with(expected_json)

  def test_incoming_proxy(self, mocker):
    tested_msgs = [
      {"type": "customReservedRawData0", "data": "test"}, # primitive
      {"type": "can", "data": [{"address": 0, "busTime": 0, "dat": "", "src": 0}]}, # list
      {"type": "testJoystick", "data": {"axes": [0, 0], "buttons": [False]}}, # dict
    ]

    mocked_pubmaster = mocker.MagicMock(spec=messaging.PubMaster)

    proxy = CerealIncomingMessageProxy(mocked_pubmaster)

    for msg in tested_msgs:
      proxy.send(json.dumps(msg).encode())

      mocked_pubmaster.send.assert_called_once()
      mt, md = mocked_pubmaster.send.call_args.args
      assert mt == msg["type"]
      assert isinstance(md, capnp._DynamicStructBuilder)
      assert hasattr(md, msg["type"])

      mocked_pubmaster.reset_mock()

  def test_livestream_track(self, mocker):
    fake_msg = messaging.new_message("livestreamDriverEncodeData")

    config = {"receive.return_value": fake_msg.to_bytes()}
    mocker.patch("msgq.SubSocket", spec=True, **config)
    track = LiveStreamVideoStreamTrack("driver")

    assert track.id.startswith("driver")
    assert track.codec_preference() == "H264"

    for i in range(5):
      packet = self.loop.run_until_complete(track.recv())
      assert packet.time_base == VIDEO_TIME_BASE
      assert packet.pts == int(i * DT_DMON * VIDEO_CLOCK_RATE)
      assert packet.size == 0

  def test_input_audio_track(self, mocker):
    packet_time, rate = 0.02, 16000
    sample_count = int(packet_time * rate)
    mocked_stream = mocker.MagicMock(spec=pyaudio.Stream)
    mocked_stream.read.return_value = b"\x00" * 2 * sample_count

    config = {"open.side_effect": lambda *args, **kwargs: mocked_stream}
    mocker.patch("pyaudio.PyAudio", spec=True, **config)
    track = AudioInputStreamTrack(audio_format=pyaudio.paInt16, packet_time=packet_time, rate=rate)

    for i in range(5):
      frame = self.loop.run_until_complete(track.recv())
      assert frame.rate == rate
      assert frame.samples == sample_count
      assert frame.pts == i * sample_count
