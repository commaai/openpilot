import asyncio
import json
import time
# for aiortc and its dependencies
import warnings
from types import SimpleNamespace
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning) # TODO: remove this when google-crc32c publish a python3.12 wheel

from aiortc import RTCDataChannel
from aiortc.mediastreams import VIDEO_CLOCK_RATE, VIDEO_TIME_BASE
import capnp
from cereal import messaging, log

from openpilot.system.webrtc.webrtcd import CerealOutgoingMessageProxy, CerealIncomingMessageProxy, \
  LivestreamBitrateController, LivestreamStatsSample, LIVESTREAM_ENCODER_CONTROL
from openpilot.system.webrtc.device.video import LiveStreamVideoStreamTrack


class FakePubMaster:
  def __init__(self):
    self.services = []
    self.sent = []

  async def add_services_if_needed(self, services):
    self.services.extend(services)

  def send(self, service, msg):
    control = msg.livestreamEncoderControl
    self.sent.append((service, control.bitrate, control.sequence))


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
      {"type": "can", "data": [{"address": 0, "dat": "", "src": 0}]}, # list
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
      if i == 0:
        start_ns = time.monotonic_ns()
        start_pts = packet.pts
      assert abs(i + packet.pts - (start_pts + (((time.monotonic_ns() - start_ns) * VIDEO_CLOCK_RATE) // 1_000_000_000))) < 450 #5ms
      assert packet.size == 0

  def test_livestream_bitrate_controller_policy(self, mocker):
    pub_master = FakePubMaster()
    controller = LivestreamBitrateController(mocker.Mock(), pub_master, max_bitrate=4_000_000, min_bitrate=500_000)

    controller._set_target(controller.target_bitrate, force=True)
    assert pub_master.sent[-1][:2] == (LIVESTREAM_ENCODER_CONTROL, 4_000_000)

    congested = LivestreamStatsSample(packet_loss_fraction=0.06)
    assert not controller.update_target(congested)
    assert controller.target_bitrate == 4_000_000

    assert controller.update_target(congested)
    assert controller.target_bitrate == 2_800_000
    assert pub_master.sent[-1][:2] == (LIVESTREAM_ENCODER_CONTROL, 2_800_000)

    clean = LivestreamStatsSample(packet_loss_fraction=0.0, rtt=0.1, available_outgoing_bitrate=4_000_000)
    for _ in range(7):
      assert not controller.update_target(clean)
    assert controller.update_target(clean)
    assert controller.target_bitrate == 3_200_000
    assert pub_master.sent[-1][:2] == (LIVESTREAM_ENCODER_CONTROL, 3_200_000)

  def test_livestream_bitrate_controller_stats_sample(self, mocker):
    pub_master = FakePubMaster()
    controller = LivestreamBitrateController(mocker.Mock(), pub_master, max_bitrate=4_000_000, min_bitrate=500_000)
    mocker.patch("openpilot.system.webrtc.webrtcd.time.monotonic", side_effect=[100.0, 101.0])

    first = {
      "remote": SimpleNamespace(type="remote-inbound-rtp", fractionLost=13, packetsLost=1, roundTripTime=0.2),
      "pair": SimpleNamespace(type="candidate-pair", state="succeeded", availableOutgoingBitrate=3_000_000),
      "out": SimpleNamespace(type="outbound-rtp", kind="video", bytesSent=1000),
    }
    sample = controller.sample_from_stats(first)
    assert round(sample.packet_loss_fraction, 4) == round(13 / 256, 4)
    assert sample.packets_lost_delta is None
    assert sample.rtt == 0.2
    assert sample.available_outgoing_bitrate == 3_000_000
    assert sample.send_bitrate is None

    second = {
      "remote": SimpleNamespace(type="remote-inbound-rtp", fractionLost=0, packetsLost=3, roundTripTime=0.3),
      "out": SimpleNamespace(type="outbound-rtp", kind="video", bytesSent=2000),
    }
    sample = controller.sample_from_stats(second)
    assert sample.packets_lost_delta == 2
    assert sample.rtt == 0.3
    assert sample.send_bitrate == 8_000

