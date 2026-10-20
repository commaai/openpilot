#!/usr/bin/env python3
import json
import re
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from openpilot.tools.ycpe.daemon import transcode
from openpilot.tools.ycpe.daemon.tests.common import DaemonCase
from openpilot.tools.ycpe.daemon.transcode import Encode, Pictures, Transcoder, backends, encode

FRAMES = 50  # 2.5 s at 20 fps: a few GOPs and a short last one


def hevc_clip(path: Path, frames: int = FRAMES) -> Path:
  """An HEVC elementary stream like the camera files: 20 fps, keyframes far apart."""
  subprocess.run(transcode.ffmpeg("-f", "lavfi", "-i", f"testsrc2=s=320x192:r=20:d={frames / 20}", "-c:v", "libx265",
                                  "-x265-params", "log-level=none:keyint=600", "-f", "hevc", str(path)), check=True)
  return path


def nal_types(data: bytes) -> list[int]:
  return [data[m.end()] & 31 for m in re.finditer(b"\x00\x00\x01", data)]


class TestPictures(unittest.TestCase):
  def test_counts_across_chunk_boundaries(self):
    # SPS, PPS, an IDR, then two P pictures of two slices each (the second slice has first_mb != 0)
    stream = b"".join(b"\x00\x00\x00\x01" + nal for nal in [b"\x67\x42", b"\x68\xce", b"\x65\x88\x80", b"\x41\x9a\x00", b"\x01\x02",
                                                             b"\x41\x80\x00", b"\x41\x00\x00"])
    for size in (1, 2, 3, 5, len(stream)):
      p = Pictures()
      for i in range(0, len(stream), size):
        p.feed(stream[i:i + size])
      p.feed(b"\x00\x00\x01\x09\x00")  # an access unit delimiter flushes the tail
      self.assertEqual((p.count, p.keys), (3, 1), size)


class TestEncode(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.dir = Path(tempfile.mkdtemp())
    cls.src = hevc_clip(cls.dir / "fcamera.hevc")

  def check_output(self, base: Path, frames: int):
    h264 = base.with_suffix(".h264").read_bytes()
    types = nal_types(h264)
    pictures = [t for t, nxt in zip(types, types[1:] + [0], strict=True) if t in (1, 5)]
    self.assertEqual(len(pictures), frames)
    # an IDR with its parameter sets every GOP frames, nothing else is a keyframe
    self.assertEqual([i for i, t in enumerate(pictures) if t == 5], list(range(0, frames, transcode.GOP)))
    idr = [i for i, t in enumerate(types) if t == 5]
    for i in idr:
      self.assertIn(7, types[max(0, i - 4):i])
      self.assertIn(8, types[max(0, i - 4):i])
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "frame=pts_time,key_frame", "-of", "json",
                            str(base.with_suffix(".mp4"))], capture_output=True, check=True, text=True)
    mp4 = json.loads(probe.stdout)["frames"]
    self.assertEqual(len(mp4), frames)
    self.assertEqual([round(float(f["pts_time"]) * 20) for f in mp4], list(range(frames)))  # frame i at i / 20 s
    self.assertEqual([i for i, f in enumerate(mp4) if f["key_frame"]], list(range(0, frames, transcode.GOP)))

  def test_every_backend_keeps_frames_and_timing(self):
    for b in backends():
      with self.subTest(backend=b.name):
        job = Encode("k", self.dir / b.name / "fcamera-medium", "medium", 0, src=str(self.src))
        info = encode(job, b)
        self.assertEqual(info["frames"], FRAMES)
        self.assertEqual(info["keyframes"], 3)
        self.assertTrue(job.base.with_suffix(".json").exists())
        self.check_output(job.base, FRAMES)

  def test_failure_leaves_nothing(self):
    job = Encode("k", self.dir / "bad" / "fcamera-medium", "medium", 0, src=str(self.dir / "nope.hevc"))
    with self.assertRaises(RuntimeError):
      encode(job, transcode.BACKENDS[-1])
    self.assertEqual(list((self.dir / "bad").iterdir()), [])


class TestTranscoder(unittest.TestCase):
  def setUp(self):
    self.dir = Path(tempfile.mkdtemp())
    self.src = hevc_clip(self.dir / "src.hevc")
    self.resolved = []

  def resolve(self, dongle, log_id, seg, cam):
    self.resolved.append(seg)
    if seg > 3:
      raise FileNotFoundError(f"no segment {seg}")
    return str(self.src)

  def test_unwatched_prefetches_are_dropped(self):
    t = Transcoder(self.dir / "cache", self.resolve, workers=1)
    try:
      t.prefetch("d", "r", [1, 2], "fcamera")  # nothing asked for this route's camera recently
      for _ in range(100):
        if not t.jobs:
          break
        time.sleep(0.01)
      self.assertEqual((t.jobs, self.resolved), ({}, []))
    finally:
      t.close()

  def test_cache_prefetch_and_missing(self):
    t = Transcoder(self.dir / "cache", self.resolve, workers=1)
    try:
      job = t.get("d", "r", 0, "fcamera")
      t.prefetch("d", "r", [1, 2, 5], "fcamera")
      self.assertTrue(job.finished.wait(60))
      self.assertEqual(job.state, "done", job.error)
      for seg, state in [(1, "done"), (2, "done"), (5, "missing")]:
        j = t.get("d", "r", seg, "fcamera", priority=9)
        self.assertTrue(j.finished.wait(60))
        self.assertEqual(j.state, state, j.error)
      self.assertEqual(self.resolved, [0, 1, 2, 5])
      self.assertIs(t.get("d", "r", 0, "fcamera"), job)
    finally:
      t.close()
    # a new transcoder (a restart) serves it from the cache without encoding
    t = Transcoder(self.dir / "cache", self.resolve, workers=1)
    try:
      again = t.get("d", "r", 0, "fcamera")
      self.assertEqual((again.state, again.frames), ("done", FRAMES))
      self.assertEqual(self.resolved, [0, 1, 2, 5])
    finally:
      t.close()


class TestRoute(DaemonCase):
  @classmethod
  def setUpClass(cls):
    cls.dir = Path(tempfile.mkdtemp())
    cls.src = hevc_clip(cls.dir / "src.hevc", 200)
    super().setUpClass()
    transcode._transcoder = Transcoder(cls.dir / "cache", lambda d, r, seg, cam: str(cls.src) if seg < 2 else cls.missing(seg),
                                       run=transcode.run_as_job)

  @staticmethod
  def missing(seg):
    raise FileNotFoundError(f"no segment {seg}")

  @classmethod
  def tearDownClass(cls):
    transcode._transcoder.close()
    super().tearDownClass()

  def test_streams_then_serves_ranges(self):
    start = time.monotonic()
    r, data = self.req("/local/video/d/r/0/fcamera.h264?ahead=1")
    self.assertEqual(r.status, 200)
    self.assertEqual(len([t for t in nal_types(data) if t in (1, 5)]), 200)
    self.assertLess(time.monotonic() - start, 60)
    r, info = self.req("/local/video/d/r/0/fcamera.json")
    self.assertEqual(json.loads(info)["state"], "done")
    r, part = self.req("/local/video/d/r/0/fcamera.mp4")
    self.assertEqual(r.status, 200)
    r, head = self.req("/local/video/d/r/0/fcamera.mp4", headers={"Range": "bytes=0-99"})
    self.assertEqual((r.status, head), (206, part[:100]))

  def test_errors(self):
    self.assertEqual(self.req("/local/video/d/r/7/fcamera.h264")[0].status, 404)
    self.assertEqual(self.req("/local/video/d/r/0/qcamera.h264")[0].status, 404)
    self.assertEqual(self.req("/local/video/d/r/0/fcamera.exe")[0].status, 404)
    self.assertEqual(self.req("/local/video/d/r/0/fcamera.h264?preset=ultra")[0].status, 404)


if __name__ == "__main__":
  unittest.main()
