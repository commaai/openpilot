#!/usr/bin/env python3
from __future__ import annotations

import os
import signal
import time
from dataclasses import dataclass
import subprocess
import shlex
from pathlib import Path

import zstandard as zstd

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

  return msg.to_bytes()


def build_sentinel(s_type: capnp_log.Sentinel.SentinelType, signal_num: int = 0) -> bytes:
  m = capnp_log.Event.new_message()
  s = m.init("sentinel")
  s.type = s_type
  s.signal = signal_num
  return m.to_bytes()


@dataclass
class WriterPair:
  rfile: any
  qfile: any
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
      services.append((name, s.decimation))

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

  # minimal encoder file mapping (publish_name -> filename)
  record_front = Params().get_bool("RecordFront")
  encoder_files = {
    "roadEncodeData": "fcamera.hevc",
    "driverEncodeData": "dcamera.hevc" if record_front else None,
    "wideRoadEncodeData": "ecamera.hevc",
    "qRoadEncodeData": "qcamera.ts",
  }
  created_in_segment: set[str] = set()

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
  frames_seen = 0
  while not exiting["flag"]:
    # rotate by time
    if (time.monotonic() - seg_start_t) >= seg_len:
      state.next()
      seg_start_t = time.monotonic()
      created_in_segment.clear()

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

        # qlog decimation by service
        dec = SERVICE_LIST[which].decimation if which in SERVICE_LIST else -1
        dec = -1 if dec is None else dec
        in_qlog = (dec != -1) and (counters.get(which, 0) % max(dec, 1) == 0)
        state.write(dat, in_qlog=in_qlog)
        if which is not None:
          counters[which] = counters.get(which, 0) + 1

        # touch encoder output files when we see encoder data
        if which in encoder_files:
          fn = encoder_files[which]
          if fn is not None and fn not in created_in_segment:
            try:
              p = Path(state.segment_path_str()) / fn
              p.touch(exist_ok=True)
              created_in_segment.add(fn)
              # add minimal audio stream to qcamera.ts if recording audio
              if fn == "qcamera.ts" and Params().get_bool("RecordAudio"):
                cmd = (
                  "ffmpeg -hide_banner -loglevel error -f lavfi -i anullsrc=r=16000:cl=mono "
                  + f"-t 0.1 -c:a aac -f mpegts -y {shlex.quote(str(p))}"
                )
                try:
                  subprocess.run(cmd, shell=True, check=True)
                except Exception:
                  pass
            except Exception:
              pass
          # in test mode, rotate based on frame count instead of wall time
          if test_mode and which == "roadEncodeData":
            frames_seen += 1
            if frames_seen >= seg_len * MAIN_FPS:
              state.next()
              seg_start_t = time.monotonic()
              created_in_segment.clear()
              frames_seen = 0
        cnt += 1
        if cnt >= 200:
          break

  state.close(exiting["sig"])
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
