import json
import numpy as np
import os
import re
import random
import string
import subprocess
import time
from collections.abc import Collection
from collections import defaultdict
from pathlib import Path

from openpilot.common.parameterized import parameterized
from openpilot.common.test import OpenpilotTestCase
import openpilot.cereal.messaging as messaging
from openpilot.cereal import log
from openpilot.cereal.services import SERVICE_LIST
from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.common.timeout import Timeout
from openpilot.common.hardware.hw import Paths
from openpilot.common.hardware import COMMA_HARDWARE
from openpilot.system.micd import SAMPLE_BUFFER, SAMPLE_RATE
from openpilot.system.loggerd.xattr_cache import getxattr
from openpilot.system.loggerd.deleter import PRESERVE_ATTR_NAME, PRESERVE_ATTR_VALUE
from openpilot.system.manager.process_config import managed_processes
from openpilot.common.version import get_version
from openpilot.tools.lib.helpers import RE
from openpilot.tools.lib.logreader import LogReader
from openpilot.cereal.visionipc import VisionStreamType
from msgq.visionipc import VisionIpcServer

SentinelType = log.Sentinel.SentinelType

CEREAL_SERVICES = [f for f in log.Event.schema.union_fields if f in SERVICE_LIST
                   and SERVICE_LIST[f].should_log and "encode" not in f.lower()]


class TestLoggerd(OpenpilotTestCase):
  def _get_latest_log_dir(self):
    log_dirs = sorted(Path(Paths.log_root()).iterdir(), key=lambda f: f.stat().st_mtime)
    return log_dirs[-1]

  def _get_log_dir(self, x):
    for l in x.splitlines():
      for p in l.split(' '):
        path = Path(p.strip())
        if path.is_dir():
          return path
    return None

  def _get_log_fn(self, x):
    for l in x.splitlines():
      for p in l.split(' '):
        path = Path(p.strip())
        if path.is_file():
          return path
    return None

  def _gen_bootlog(self):
    with Timeout(5):
      out = subprocess.check_output("./bootlog", cwd=os.path.join(BASEDIR, "openpilot/system/loggerd"), encoding='utf-8')

    log_fn = self._get_log_fn(out)

    # check existence
    assert log_fn is not None

    return log_fn

  def _check_init_data(self, msgs):
    msg = msgs[0]
    assert msg.which() == 'initData'

  def _check_sentinel(self, msgs, route):
    start_type = SentinelType.startOfRoute if route else SentinelType.startOfSegment
    assert msgs[1].sentinel.type == start_type

    end_type = SentinelType.endOfRoute if route else SentinelType.endOfSegment
    assert msgs[-1].sentinel.type == end_type

  def _publish_random_messages(self, services: Collection[str]) -> dict[str, list]:
    pm = messaging.PubMaster(list(services))

    managed_processes["loggerd"].start()
    for s in services:
      assert pm.wait_for_readers_to_update(s, timeout=5)

    sent_msgs = defaultdict(list)
    for i in range(random.randint(2, 10) * 100):
      for s in services:
        try:
          m = messaging.new_message(s)
        except Exception:
          m = messaging.new_message(s, random.randint(2, 10))
        pm.send(s, m)
        sent_msgs[s].append(m)

      # Keep msgq's finite per-service queues from wrapping; this test asserts
      # that loggerd logged every message we sent.
      if (i + 1) % 100 == 0:
        for s in services:
          assert pm.wait_for_readers_to_update(s, timeout=5)

    for s in services:
      assert pm.wait_for_readers_to_update(s, timeout=5)
    managed_processes["loggerd"].stop()

    return sent_msgs

  def _publish_camera_and_audio_messages(self, num_segs=1, segment_length=5):
    # Use small frame sizes for testing (width, height, size, stride, uv_offset)
    # NV12 format: size = stride * height * 1.5, uv_offset = stride * height
    w, h = 320, 240
    frame_spec = (w, h, w * h * 3 // 2, w, w * h)
    streams = [
      (VisionStreamType.VISION_STREAM_NARROW_ROAD, frame_spec, "narrowRoadCameraState"),
      (VisionStreamType.VISION_STREAM_CABIN, frame_spec, "cabinCameraState"),
      (VisionStreamType.VISION_STREAM_WIDE_ROAD, frame_spec, "wideRoadCameraState"),
    ]

    sm = messaging.SubMaster(["narrowRoadEncodeData"])
    pm = messaging.PubMaster([s for _, _, s in streams] + ["rawAudioData"])
    vipc_server = VisionIpcServer("camerad")
    for stream_type, frame_spec, _ in streams:
      vipc_server.create_buffers_with_sizes(stream_type, 40, *(frame_spec))
    vipc_server.start_listener()

    os.environ["LOGGERD_TEST"] = "1"
    os.environ["LOGGERD_SEGMENT_LENGTH"] = str(segment_length)
    managed_processes["loggerd"].start()
    managed_processes["encoderd"].start()
    assert pm.wait_for_readers_to_update("narrowRoadCameraState", timeout=5)

    record_audio = Params().get_bool("RecordAudio")
    if record_audio:
      assert pm.wait_for_readers_to_update("rawAudioData", timeout=5)
    fps = 20
    start_time = time.monotonic_ns()
    for n in range(1, int(num_segs * segment_length * fps) + 1):
      timestamp = start_time + n * 1_000_000_000 // fps
      # send video
      for stream_type, frame_spec, state in streams:
        dat = np.empty(frame_spec[2], dtype=np.uint8)
        vipc_server.send(stream_type, dat[:].flatten().tobytes(), n, timestamp, timestamp)

        camera_state = messaging.new_message(state)
        frame = getattr(camera_state, state)
        frame.frameId = n
        pm.send(state, camera_state)

      # send audio
      msg = messaging.new_message('rawAudioData')
      msg.logMonoTime = timestamp
      msg.rawAudioData.data = bytes(SAMPLE_BUFFER * 2)  # 50ms of mono int16 audio
      msg.rawAudioData.sampleRate = SAMPLE_RATE
      pm.send('rawAudioData', msg)
      if record_audio:
        assert pm.wait_for_readers_to_update('rawAudioData', timeout=5)

      for _, _, state in streams:
        assert pm.wait_for_readers_to_update(state, timeout=5, dt=0.001)

      sm.update(100)  # wait for encode data publish

    managed_processes["loggerd"].stop()
    managed_processes["encoderd"].stop()

  def test_init_data_values(self):
    os.environ["CLEAN"] = random.choice(["0", "1"])

    dongle  = ''.join(random.choice(string.printable) for n in range(random.randint(1, 100)))
    fake_params = [
      # param, initData field, value
      ("DongleId", "dongleId", dongle),
      ("GitCommit", "gitCommit", "commit"),
      ("GitCommitDate", "gitCommitDate", "date"),
      ("GitBranch", "gitBranch", "branch"),
      ("GitRemote", "gitRemote", "remote"),
    ]
    params = Params()
    for k, _, v in fake_params:
      params.put(k, v, block=True)
    params.put("AccessToken", "abc", block=True)

    lr = list(LogReader(str(self._gen_bootlog())))
    initData = lr[0].initData

    assert initData.dirty != bool(os.environ["CLEAN"])
    assert initData.version == get_version()

    if os.path.isfile("/proc/cmdline"):
      with open("/proc/cmdline") as f:
        assert list(initData.kernelArgs) == f.read().strip().split(" ")

      with open("/proc/version") as f:
        assert initData.kernelVersion == f.read()

    # check params
    logged_params = {entry.key: entry.value for entry in initData.params.entries}
    expected_params = {k for k, _, __ in fake_params} | {'AccessToken', 'BootCount'}
    assert set(logged_params.keys()) == expected_params, set(logged_params.keys()) ^ expected_params
    assert logged_params['AccessToken'] == b'', f"DONT_LOG param value was logged: {repr(logged_params['AccessToken'])}"
    for param_key, initData_key, v in fake_params:
      assert getattr(initData, initData_key) == v
      assert logged_params[param_key].decode() == v

  def test_rotation(self):
    Params().put("RecordFront", True, block=True)

    expected_files = {"rlog.zst", "qlog.zst", "qcamera.ts", "fcamera.hevc", "dcamera.hevc", "ecamera.hevc"}

    num_segs = random.randint(2, 3)
    length = random.randint(4, 5) # H264 encoder uses 40 lookahead frames and does B-frame reordering, so minimum 3 seconds before qcam output

    self._publish_camera_and_audio_messages(num_segs=num_segs, segment_length=length)

    route_path = str(self._get_latest_log_dir()).rsplit("--", 1)[0]
    for n in range(num_segs):
      p = Path(f"{route_path}--{n}")
      logged = {f.name for f in p.iterdir() if f.is_file()}
      diff = logged ^ expected_files
      assert len(diff) == 0, f"didn't get all expected files. seg={n} {route_path=}, {diff=}\n{logged=} {expected_files=}"

  def test_bootlog(self):
    # generate bootlog with fake launch log
    launch_log = ''.join(str(random.choice(string.printable)) for _ in range(100))
    with open("/tmp/launch_log", "w") as f:
      f.write(launch_log)

    bootlog_path = self._gen_bootlog()
    lr = list(LogReader(str(bootlog_path)))

    # check length
    assert len(lr) == 2  # boot + initData

    self._check_init_data(lr)

    # check msgs
    bootlog_msgs = [m for m in lr if m.which() == 'boot']
    assert len(bootlog_msgs) == 1

    # sanity check values
    boot = bootlog_msgs.pop().boot
    assert abs(boot.wallTimeNanos - time.time_ns()) < 5*1e9 # within 5s
    assert boot.launchLog == launch_log

    if COMMA_HARDWARE:
      for fn in ["console-ramoops", "pmsg-ramoops-0"]:
        path = Path(os.path.join("/sys/fs/pstore/", fn))
        if path.is_file():
          with open(path, "rb") as f:
            expected_val = f.read()
          bootlog_val = [e.value for e in boot.pstore.entries if e.key == fn][0]
          assert expected_val == bootlog_val
    else:
      assert len(boot.pstore.entries) == 0

    # next one should increment by one
    bl1 = re.match(RE.LOG_ID_V2, bootlog_path.name)
    bl2 = re.match(RE.LOG_ID_V2, self._gen_bootlog().name)
    assert bl1.group('uid') != bl2.group('uid')
    assert int(bl1.group('count')) == 0 and int(bl2.group('count')) == 1

  def test_qlog(self):
    qlog_services = [s for s in CEREAL_SERVICES if SERVICE_LIST[s].decimation is not None]
    no_qlog_services = [s for s in CEREAL_SERVICES if SERVICE_LIST[s].decimation is None]

    services = random.sample(qlog_services, random.randint(2, min(10, len(qlog_services)))) + \
               random.sample(no_qlog_services, random.randint(2, min(10, len(no_qlog_services))))
    sent_msgs = self._publish_random_messages(services)

    qlog_path = os.path.join(self._get_latest_log_dir(), "qlog.zst")
    lr = list(LogReader(qlog_path))

    # check initData and sentinel
    self._check_init_data(lr)
    self._check_sentinel(lr, True)

    recv_msgs = defaultdict(list)
    for m in lr:
      recv_msgs[m.which()].append(m)

    for s, msgs in sent_msgs.items():
      recv_cnt = len(recv_msgs[s])

      if s in no_qlog_services:
        # check services with no specific decimation aren't in qlog
        assert recv_cnt == 0, f"got {recv_cnt} {s} msgs in qlog"
      else:
        # check logged message count matches decimation
        decimation = SERVICE_LIST[s].decimation
        assert decimation is not None
        expected_cnt = (len(msgs) - 1) // decimation + 1
        assert recv_cnt == expected_cnt, f"expected {expected_cnt} msgs for {s}, got {recv_cnt}"

  def test_rlog(self):
    services = random.sample(CEREAL_SERVICES, random.randint(5, 10))
    sent_msgs = self._publish_random_messages(services)

    lr = list(LogReader(os.path.join(self._get_latest_log_dir(), "rlog.zst")))

    # check initData and sentinel
    self._check_init_data(lr)
    self._check_sentinel(lr, True)

    # check all messages were logged and in order
    lr = lr[2:-1] # slice off initData and both sentinels
    for m in lr:
      sent = sent_msgs[m.which()].pop(0)
      sent.clear_write_flag()
      assert sent.to_bytes() == m.as_builder().to_bytes()

  def test_preserving_bookmarked_segments(self):
    services = set(random.sample(CEREAL_SERVICES, random.randint(5, 10))) | {"userBookmark"}
    self._publish_random_messages(services)

    segment_dir = self._get_latest_log_dir()
    assert getxattr(segment_dir, PRESERVE_ATTR_NAME) == PRESERVE_ATTR_VALUE

  def test_not_preserving_nonbookmarked_segments(self):
    services = set(random.sample(CEREAL_SERVICES, random.randint(5, 10))) - {"userBookmark"}
    self._publish_random_messages(services)

    segment_dir = self._get_latest_log_dir()
    assert getxattr(segment_dir, PRESERVE_ATTR_NAME) is None

  @parameterized.expand([True, False])
  def test_record_front(self, record_front):
    params = Params()
    params.put_bool("RecordFront", record_front, block=True)

    self._publish_camera_and_audio_messages()

    cabin_hevc_exists = os.path.exists(os.path.join(self._get_latest_log_dir(), 'dcamera.hevc'))
    assert cabin_hevc_exists == record_front

  def _assert_audio_packets_match(self, qcamera, audio):
    encoded = subprocess.check_output([
      'ffmpeg', '-v', 'error', '-i', str(qcamera), '-map', '0:a:0', '-c:a', 'copy', '-f', 'adts', 'pipe:1',
    ])
    packets = []
    offset = 0
    while offset < len(encoded):
      header = encoded[offset:offset + 7]
      assert header[0] == 0xff and header[1] & 0xf6 == 0xf0
      size = ((header[3] & 3) << 11) | (header[4] << 3) | (header[5] >> 5)
      header_size = 7 if header[1] & 1 else 9
      assert size > header_size
      packets.append(encoded[offset + header_size:offset + size])
      offset += size
    assert offset == len(encoded)
    assert packets == [bytes(packet.data) for packet in audio]

  @parameterized.expand([True, False])
  def test_record_audio(self, record_audio):
    params = Params()
    params.put_bool("RecordAudio", record_audio, block=True)

    self._publish_camera_and_audio_messages()

    qcamera_ts_path = os.path.join(self._get_latest_log_dir(), 'qcamera.ts')
    streams = json.loads(subprocess.check_output([
      "ffprobe", "-i", qcamera_ts_path, "-show_streams", "-select_streams", "a", "-loglevel", "error", "-of", "json",
    ]))["streams"]
    assert bool(streams) == record_audio
    if record_audio:
      assert len(streams) == 1
      assert streams[0]["codec_name"] == "aac"
      assert int(streams[0]["sample_rate"]) == SAMPLE_RATE
      assert streams[0]["channels"] == 1

    messages = list(LogReader(os.path.join(self._get_latest_log_dir(), 'rlog.zst')))
    assert not any(message.which() == 'rawAudioData' for message in messages)
    audio = [message.audioEncodeData for message in messages if message.which() == 'audioEncodeData']
    assert bool(audio) == record_audio
    if record_audio:
      assert all(packet.sampleRate == SAMPLE_RATE and packet.header and packet.data for packet in audio)
      assert sum(packet.samples - packet.discardStart - packet.discardEnd for packet in audio) == 5 * SAMPLE_RATE
      self._assert_audio_packets_match(qcamera_ts_path, audio)
    assert not any(message.which() in ('rawAudioData', 'audioEncodeData')
                   for message in LogReader(os.path.join(self._get_latest_log_dir(), 'qlog.zst')))

  def test_audio_without_video(self):
    Params().put_bool("RecordAudio", True, block=True)
    pm = messaging.PubMaster(["rawAudioData"])
    managed_processes["loggerd"].start()
    assert pm.wait_for_readers_to_update("rawAudioData", timeout=5)
    sample_count = 23 * SAMPLE_BUFFER  # exercise a final partial AAC frame
    samples = (3000 * np.sin(2 * np.pi * 12000 * np.arange(sample_count) / SAMPLE_RATE)).astype(np.int16)
    start_time = time.monotonic_ns()
    for start in range(0, sample_count, SAMPLE_BUFFER):
      message = messaging.new_message('rawAudioData', valid=True)
      message.logMonoTime = start_time + start * 1_000_000_000 // SAMPLE_RATE
      message.rawAudioData.sampleRate = SAMPLE_RATE
      message.rawAudioData.data = samples[start:start + SAMPLE_BUFFER].tobytes()
      pm.send('rawAudioData', message)
      assert pm.wait_for_readers_to_update('rawAudioData', timeout=5)
    managed_processes["loggerd"].stop()

    messages = list(LogReader(os.path.join(self._get_latest_log_dir(), 'rlog.zst')))
    audio = [message.audioEncodeData for message in messages if message.which() == 'audioEncodeData']
    assert not any(message.which() == 'rawAudioData' for message in messages)
    assert sum(packet.samples - packet.discardStart - packet.discardEnd for packet in audio) == sample_count
    assert audio[0].discardStart > 0 and audio[-1].discardEnd > 0
    assert all(packet.segmentNum == 0 and packet.header == audio[0].header for packet in audio)

    # Reconstruct ADTS from AudioSpecificConfig to prove the logged packets decode independently.
    encoded = bytearray()
    for packet in audio:
      config = int.from_bytes(packet.header[:2], 'big')
      profile = (config >> 11) - 1
      frequency_index = (config >> 7) & 15
      channels = (config >> 3) & 15
      size = len(packet.data) + 7
      encoded.extend(bytes([0xff, 0xf1, (profile << 6) | (frequency_index << 2) | (channels >> 2),
                            ((channels & 3) << 6) | (size >> 11), (size >> 3) & 255, ((size & 7) << 5) | 31, 0xfc]))
      encoded.extend(packet.data)
    frames = json.loads(subprocess.check_output([
      'ffprobe', '-v', 'error', '-f', 'aac', '-i', 'pipe:0', '-show_frames', '-show_entries',
      'frame=nb_samples,sample_rate,channels', '-of', 'json',
    ], input=encoded))["frames"]
    assert sum(frame['nb_samples'] for frame in frames) == sum(packet.samples for packet in audio)
    assert all(frame['channels'] == 1 for frame in frames)

  def test_audio_rotation(self):
    Params().put_bool("RecordAudio", True, block=True)
    self._publish_camera_and_audio_messages(num_segs=3, segment_length=5)
    route_path = str(self._get_latest_log_dir()).rsplit("--", 1)[0]
    total_samples = 0
    for segment in range(3):
      path = Path(f"{route_path}--{segment}")
      audio = [message.audioEncodeData for message in LogReader(str(path / 'rlog.zst')) if message.which() == 'audioEncodeData']
      assert audio and all(packet.segmentNum == segment for packet in audio)
      assert audio[0].discardStart > 0
      total_samples += sum(packet.samples - packet.discardStart - packet.discardEnd for packet in audio)
      streams = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'a', '-show_streams', '-of', 'json', str(path / 'qcamera.ts'),
      ]))["streams"]
      assert len(streams) == 1 and int(streams[0]['sample_rate']) == SAMPLE_RATE
      self._assert_audio_packets_match(path / 'qcamera.ts', audio)
    assert total_samples == 15 * SAMPLE_RATE
