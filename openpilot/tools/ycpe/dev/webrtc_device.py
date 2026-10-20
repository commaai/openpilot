#!/usr/bin/env python3
"""The device side of dev/webrtc_test.py, in openpilot's venv: writes a log root with one synthetic route, then
publishes the same kind of messages live under OPENPILOT_PREFIX (carState and can at 100 Hz, deviceState at 2 Hz, and
the road and wide road cameras as H.264 at 20 fps when ffmpeg is there). Prints "video <n> frames", then "wall <ms>": wall clock minus
monotonic, for latency."""
import argparse
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import zstandard

from openpilot.cereal import messaging

KEYFRAME = 0x8  # V4L2_BUF_FLAG_KEYFRAME


def events(t):
  cs = messaging.new_message("carState")
  cs.carState.vEgo = 10 + 5 * math.sin(t)
  can = messaging.new_message("can", 1)
  can.can[0].address, can.can[0].src, can.can[0].dat = 0x123, 0, int(t * 100).to_bytes(8, "little")
  return [cs, can] + ([messaging.new_message("deviceState")] if int(t * 100) % 50 == 0 else [])


def write_route(root: Path, log_id: str, segments: int):
  for n in range(segments):
    msgs = [messaging.new_message("initData")]
    for k in range(6000):
      for m in events(t := n * 60 + k / 100):
        m.logMonoTime = int((1000 + t) * 1e9)
        msgs.append(m)
    msgs[0].logMonoTime = msgs[1].logMonoTime
    (root / f"{log_id}--{n}").mkdir(parents=True, exist_ok=True)
    (root / f"{log_id}--{n}" / "rlog.zst").write_bytes(zstandard.compress(b"".join(m.to_bytes() for m in msgs)))
  (root / f"{log_id}--0" / "fcamera.hevc").write_bytes(os.urandom(20 << 20))  # stands in for a camera file, to time reads


def h264_frames():
  """access units of a test pattern, start codes all 4 bytes long (teleoprtc's packetizer splits on those). Tries every
  ffmpeg on PATH: openpilot's venv has one without lavfi or the h264 muxer"""
  args = ["-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=20", "-t", "5", "-c:v", "libx264", "-pix_fmt", "yuv420p",
          "-profile:v", "baseline", "-g", "20", "-bf", "0", "-x264-params", "aud=1", "-f", "h264", "-"]
  for d in dict.fromkeys(os.environ.get("PATH", "").split(os.pathsep)):
    try:
      raw = subprocess.run([os.path.join(d, "ffmpeg"), *args], capture_output=True, check=True).stdout
      break
    except (OSError, subprocess.CalledProcessError):
      continue
  else:
    return []
  raw = raw.replace(b"\x00\x00\x01", b"\x00\x00\x00\x01").replace(b"\x00\x00\x00\x00\x01", b"\x00\x00\x00\x01")
  starts = [m.start() for m in re.finditer(b"\x00\x00\x00\x01\x09", raw)] + [len(raw)]
  return [raw[a:b] for a, b in zip(starts, starts[1:], strict=False)]


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--log-root", required=True)
  p.add_argument("--log-id", default="00000001--0123456789")
  p.add_argument("--segments", type=int, default=2)
  p.add_argument("--seconds", type=float, default=120)
  args = p.parse_args()
  assert os.environ.get("OPENPILOT_PREFIX"), "needs an OPENPILOT_PREFIX of its own"

  write_route(Path(args.log_root), args.log_id, args.segments)
  frames = h264_frames()
  socks = {s: messaging.pub_sock(s) for s in ("carState", "can", "deviceState", "livestreamNarrowRoadEncodeData", "livestreamWideRoadEncodeData")}
  print(f"video {len(frames)} frames", flush=True)
  print(f"wall {time.time() * 1000 - time.monotonic_ns() / 1e6:.3f}", flush=True)  # noqa: TID251  # the page's clock is wall time
  t0, k = time.monotonic(), 0
  while (t := time.monotonic() - t0) < args.seconds:
    for m in events(t):
      socks[m.which()].send(m.to_bytes())
    if frames and k % 5 == 0:
      f = frames[k // 5 % len(frames)]
      for s in ("livestreamNarrowRoadEncodeData", "livestreamWideRoadEncodeData"):
        v = messaging.new_message(s)
        getattr(v, s).data = f
        getattr(v, s).idx.flags = KEYFRAME if b"\x00\x00\x00\x01\x67" in f[:64] else 0
        socks[s].send(v.to_bytes())
    k += 1
    time.sleep(max(0, t0 + k / 100 - time.monotonic()))
  sys.exit(0)


if __name__ == "__main__":
  main()
