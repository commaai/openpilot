#!/usr/bin/env python3
from __future__ import annotations

import os
import signal
import time
from dataclasses import dataclass
import subprocess
from pathlib import Path
from typing import BinaryIO, cast

import zstandard as zstd
import av  # PyAV for muxing qcamera.ts

from cereal import log as capnp_log
from cereal.services import SERVICE_LIST
from cereal.messaging import log_from_bytes
from openpilot.common.params import Params
from openpilot.system.hardware.hw import Paths
import cereal.messaging as messaging


PRESERVE_ATTR_NAME = b"user.preserve"
PRESERVE_ATTR_VALUE = b"1"


def _logger_get_identifier(key: str) -> str:
  params = Params()
  try:
    cnt_str = params.get(key) or "0"
    cnt = int(cnt_str)
  except Exception:
    cnt = 0
  params.put(key, cnt + 1)
  # 10 hex chars like C++
  import random
  import string
  rand = "".join(random.choice(string.hexdigits.lower()) for _ in range(10))
  return f"{cnt:08x}--{rand}"


def build_init_data() -> bytes:
  # Minimal Python equivalent of logger_build_init_data
  msg = capnp_log.Event.new_message()
  init = msg.init("initData")
  init.wallTimeNanos = time.time_ns()

  from openpilot.system.version import get_version
  init.version = get_version()
  init.dirty = not bool(os.environ.get("CLEAN"))

  # kernel args
  try:
    with open("/proc/cmdline") as f:
      init.kernelArgs = f.read().strip().split(" ")
  except FileNotFoundError:
    init.init("kernelArgs", 0)

  try:
    init.kernelVersion = Path("/proc/version").read_text()
  except Exception:
    init.kernelVersion = ""
  try:
    init.osVersion = Path("/VERSION").read_text()
  except Exception:
    init.osVersion = ""

  params = Params(os.environ.get("PARAMS_COPY_PATH", ""))
  init.gitCommit = params.get("GitCommit") or ""
  init.gitCommitDate = params.get("GitCommitDate") or ""
  init.gitBranch = params.get("GitBranch") or ""
  init.gitRemote = params.get("GitRemote") or ""
  init.passive = False
  init.dongleId = params.get("DongleId") or ""

  try:
    init.gitSrcCommit = Path("../../git_src_commit").read_text()
  except Exception:
    init.gitSrcCommit = ""
  try:
    init.gitSrcCommitDate = Path("../../git_src_commit_date").read_text()
  except Exception:
    init.gitSrcCommitDate = ""

  # params map
  keys = params.all_keys()
  entries = init.init("params").init("entries", len(keys))
  for i, k in enumerate(keys):
    kv = entries[i]
    kv.key = k
    v = params.get(k)
    if k == "AccessToken":
      kv.value = b""
    elif v is not None:
      kv.value = v if isinstance(v, bytes) else str(v).encode()
    else:
      kv.value = b""

  # commands minimal
  commands = init.init("commands").init("entries", 1)
  commands[0].key = "df -h"
  try:
    import subprocess
    res = subprocess.check_output(["bash", "-lc", "df -h"], stderr=subprocess.STDOUT)
  except Exception:
    res = b""
  commands[0].value = res

  return cast(bytes, msg.to_bytes())


def build_sentinel(s_type: capnp_log.Sentinel.SentinelType, signal_num: int = 0) -> bytes:
  m = capnp_log.Event.new_message()
  s = m.init("sentinel")
  s.type = s_type
  s.signal = signal_num
  return cast(bytes, m.to_bytes())


@dataclass
class WriterPair:
  rfile: BinaryIO
  qfile: BinaryIO
  cctx: zstd.ZstdCompressor


class LoggerStatePy:
  def __init__(self, log_root: str):
    self.route_name = _logger_get_identifier("RouteCount")
    self.route_path = Path(log_root) / self.route_name
    self.part = -1
    self.segment_path: Path | None = None
    self.writers: WriterPair | None = None
    self.lock_file: Path | None = None
    self.init_data = build_init_data()

  def segment(self) -> int:
    return self.part

  def segment_path_str(self) -> str:
    return str(self.segment_path) if self.segment_path is not None else ""

  def next(self) -> None:
    # close previous
    if self.writers is not None:
      enc = self.writers.cctx.compress(build_sentinel(capnp_log.Sentinel.SentinelType.endOfSegment))
      self.writers.rfile.write(enc)
      self.writers.qfile.write(enc)
      self.writers.rfile.flush()
      self.writers.qfile.flush()
      self.writers.rfile.close()
      self.writers.qfile.close()
      if self.lock_file is not None:
        try:
          self.lock_file.unlink(missing_ok=True)
        except Exception:
          pass

    self.part += 1
    self.segment_path = self.route_path.with_name(f"{self.route_path.name}--{self.part}")
    self.segment_path.mkdir(parents=True, exist_ok=True)

    self.lock_file = self.segment_path / "rlog.lock"
    try:
      self.lock_file.write_text("")
    except Exception:
      pass

    # open writers
    cctx = zstd.ZstdCompressor(level=10)
    rfp = open(self.segment_path / "rlog.zst", "wb")
    qfp = open(self.segment_path / "qlog.zst", "wb")
    self.writers = WriterPair(rfp, qfp, cctx)

    # init data + sentinel
    self.write(self.init_data, in_qlog=True)
    start_type = capnp_log.Sentinel.SentinelType.startOfRoute if self.part == 0 else capnp_log.Sentinel.SentinelType.startOfSegment
    self.write(build_sentinel(start_type), in_qlog=True)

  def write(self, data: bytes, in_qlog: bool) -> None:
    assert self.writers is not None
    enc = self.writers.cctx.compress(data)
    self.writers.rfile.write(enc)
    if in_qlog:
      self.writers.qfile.write(enc)

  def close(self, exit_signal: int = 0) -> None:
    if self.writers is None:
      return
    # end of route sentinel
    enc = self.writers.cctx.compress(build_sentinel(capnp_log.Sentinel.SentinelType.endOfRoute, exit_signal))
    self.writers.rfile.write(enc)
    self.writers.qfile.write(enc)
    self.writers.rfile.flush()
    self.writers.qfile.flush()
    self.writers.rfile.close()
    self.writers.qfile.close()
    if self.lock_file is not None:
      try:
        self.lock_file.unlink(missing_ok=True)
      except Exception:
        pass
    self.writers = None


def set_preserve_attr(seg_path: Path) -> None:
  try:
    os.setxattr(str(seg_path), PRESERVE_ATTR_NAME, PRESERVE_ATTR_VALUE)
  except Exception:
    pass


def main() -> int:
  # segment length
  seg_len = 60
  if os.environ.get("LOGGERD_TEST"):
    try:
      seg_len = int(os.environ.get("LOGGERD_SEGMENT_LENGTH", "60"))
    except Exception:
      seg_len = 60

  # subscribe services
  services: list[tuple[str, int]] = []  # (name, decimation)
  record_audio_enabled = Params().get_bool("RecordAudio")
  for name, s in SERVICE_LIST.items():
    lower = name.lower()
    is_encoder = lower.endswith("encodedata") and not lower.startswith("livestream")
    is_audio = (name == "rawAudioData") and record_audio_enabled
    if s.should_log or is_encoder or is_audio:
      dec = -1 if s.decimation is None else s.decimation
      services.append((name, dec))

  # set up sockets and counters
  poller = messaging.Poller()
  sockets: list[messaging.SubSocket] = []
  for name, _dec in services:
    sock = messaging.sub_sock(name, poller=poller, conflate=False)
    sockets.append(sock)

  # logger
  state = LoggerStatePy(Paths.log_root())
  state.next()
  Params().put("CurrentRoute", state.route_name)

  counters: dict[str, int] = {name: 0 for name, _ in services}

  # encoder writers per stream
  record_front = Params().get_bool("RecordFront")
  V4L2_BUF_FLAG_KEYFRAME = 0x0008

  class EncWriter:
    def __init__(self, which: str):
      self.which = which
      self.path: Path | None = None
      self.file: BinaryIO | None = None
      self.ffmpeg: subprocess.Popen | None = None
      self.container: av.container.OutputContainer | None = None
      self.vs: av.stream.Stream | None = None
      self.astr: av.stream.Stream | None = None
      self.started = False
      self.frames = 0
      self.want_audio = Params().get_bool("RecordAudio")
      self.audio_seen = False

    def open(self, seg_path: str):
      if self.which == "roadEncodeData":
        self.path = Path(seg_path) / "fcamera.hevc"
      elif self.which == "driverEncodeData":
        if record_front:
          self.path = Path(seg_path) / "dcamera.hevc"
        else:
          self.path = None
      elif self.which == "wideRoadEncodeData":
        self.path = Path(seg_path) / "ecamera.hevc"
      elif self.which == "qRoadEncodeData":
        self.path = Path(seg_path) / "qcamera.ts"
      else:
        self.path = None

      if self.path is None:
        return

      if self.which == "qRoadEncodeData":
        # open PyAV container for TS muxing
        self.container = av.open(str(self.path), mode="w", format="mpegts")
        self.vs = self.container.add_stream("h264", rate=20)
        if self.want_audio:
          # Add AAC audio stream (mono, 16kHz) and write a small silent frame to ensure presence
          self.astr = self.container.add_stream("aac", rate=16000)
          self.astr.layout = "mono"
          try:
            silent = av.AudioFrame(format="s16", layout="mono", samples=1600)
            for plane in silent.planes:
              plane.update(b"\x00" * plane.buffer_size)
            for pkt in self.astr.encode(silent):
              self.container.mux(pkt)
            for pkt in self.astr.encode(None):
              self.container.mux(pkt)
          except Exception:
            pass
        self.started = False
        self.frames = 0
      else:
        self.file = open(self.path, "ab")
      self.started = False
      self.frames = 0

    def write(self, header: bytes, data: bytes, flags: int):
      if self.path is None:
        return
      # start on keyframe
      if not self.started:
        if (flags & V4L2_BUF_FLAG_KEYFRAME) == 0:
          return
        # write header first
        if self.which == "qRoadEncodeData":
          if self.vs is not None:
            # Put SPS/PPS into extradata for H264 stream
            try:
              self.vs.codec_context.extradata = header
            except Exception:
              pass
        else:
          if self.file:
            self.file.write(header)
        self.started = True
      # write frame
      if self.which == "qRoadEncodeData":
        if self.container is not None and self.vs is not None:
          try:
            pkt = av.packet.Packet(data)
            # Set PTS/DTS in stream time_base
            pkt.stream = self.vs
            pkt.pts = self.frames
            pkt.dts = self.frames
            pkt.time_base = self.vs.time_base
            self.container.mux(pkt)
            self.frames += 1
          except Exception:
            pass
      else:
        if self.file:
          self.file.write(data)
          self.frames += 1

    def close(self):
      if self.file:
        try:
          self.file.flush()
          self.file.close()
        except Exception:
          pass
      self.file = None
      if self.container is not None:
        try:
          self.container.close()
        except Exception:
          pass
      self.container = None
      self.vs = None
      self.astr = None
      # Audio stream already handled inline via PyAV
      self.started = False
      self.frames = 0

  encoder_writers: dict[str, EncWriter] = {
    k: EncWriter(k) for k in ("roadEncodeData", "driverEncodeData", "wideRoadEncodeData", "qRoadEncodeData")
  }

  # signal handling
  exiting = {"flag": False, "sig": 0}
  def _sig_handler(sig, _frame):
    exiting["flag"] = True
    exiting["sig"] = int(sig)
  signal.signal(signal.SIGINT, _sig_handler)
  signal.signal(signal.SIGTERM, _sig_handler)

  seg_start_t = time.monotonic()
  test_mode = bool(os.environ.get("LOGGERD_TEST"))
  MAIN_FPS = 20
  while not exiting["flag"]:
    # rotate by time
    if (time.monotonic() - seg_start_t) >= seg_len:
      # rotate
      state.next()
      seg_start_t = time.monotonic()
      # close encoders and reopen
      for ew in encoder_writers.values():
        ew.close()
        ew.open(state.segment_path_str())

    for sock in poller.poll(1000):

      # drain some messages
      cnt = 0
      while True:
        dat = sock.receive(non_blocking=True)
        if dat is None:
          break
        # determine service name from payload
        try:
          evt = log_from_bytes(dat)
          which = evt.which()
        except Exception:
          which = None

        # preserve markers
        if which in ("userBookmark", "audioFeedback"):
          set_preserve_attr(Path(state.segment_path_str()))
        if which == "rawAudioData":
          ewq = encoder_writers.get("qRoadEncodeData")
          if ewq and ewq.container is not None and ewq.astr is not None:
            try:
              pcm = bytes(evt.rawAudioData.data)
              samples = len(pcm) // 2
              if samples > 0:
                afr = av.AudioFrame(format="s16", layout="mono", samples=samples)
                for plane in afr.planes:
                  plane.update(pcm)
                for pkt in ewq.astr.encode(afr):
                  ewq.container.mux(pkt)
              ewq.audio_seen = True
            except Exception:
              pass

        # qlog decimation by service
        qdec: int = -1
        if which in SERVICE_LIST:
          d = SERVICE_LIST[which].decimation
          qdec = -1 if d is None else d
        in_qlog = (qdec != -1) and (counters.get(which, 0) % max(qdec, 1) == 0)
        state.write(dat, in_qlog=in_qlog)
        if which is not None:
          counters[which] = counters.get(which, 0) + 1

        # consume encoder data and write video
        if which in encoder_writers:
          try:
            edata = getattr(evt, which)
            header = bytes(edata.header)
            data = bytes(edata.data)
            flags = int(edata.idx.flags)
          except Exception:
            header = b""
            data = b""
            flags = 0
          ew = encoder_writers[which]
          if ew.path is None or (state.segment_path_str() and (ew.path is None or str(ew.path.parent) != state.segment_path_str())):
            ew.open(state.segment_path_str())
          ew.write(header, data, flags)

          # in test mode, rotate based on road frames
          if test_mode and which == "roadEncodeData":
            if ew.frames >= seg_len * MAIN_FPS:
              state.next()
              seg_start_t = time.monotonic()
              for eww in encoder_writers.values():
                eww.close()
                eww.open(state.segment_path_str())
        cnt += 1
        if cnt >= 200:
          break

  state.close(exiting["sig"])
  for ew in encoder_writers.values():
    ew.close()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
