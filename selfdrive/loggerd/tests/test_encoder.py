#!/usr/bin/env python3
import math
import os
import random
import shutil
import subprocess
import time
import unittest
from pathlib import Path

from parameterized import parameterized
from tqdm import trange

from common.params import Params
from common.timeout import Timeout
from selfdrive.hardware import EON, TICI
from selfdrive.loggerd.config import ROOT
from selfdrive.manager.process_config import managed_processes
from tools.lib.logreader import LogReader

SEGMENT_LENGTH = 2
if EON:
  FULL_SIZE = 1253786 # file size for a 2s segment in bytes
  CAMERAS = [
    ("fcamera.hevc", 20, FULL_SIZE, "roadEncodeIdx"),
    ("dcamera.hevc", 10, 770920, "driverEncodeIdx"),
    ("qcamera.ts", 20, 77066, None),
  ]
else:
  FULL_SIZE = 2507572
  CAMERAS = [
    ("fcamera.hevc", 20, FULL_SIZE, "roadEncodeIdx"),
    ("dcamera.hevc", 20, FULL_SIZE, "driverEncodeIdx"),
    ("ecamera.hevc", 20, FULL_SIZE, "wideRoadEncodeIdx"),
    ("qcamera.ts", 20, 77066, None),
  ]

# we check frame count, so we don't have to be too strict on size
FILE_SIZE_TOLERANCE = 0.5


class TestEncoder(unittest.TestCase):

  # TODO: all of loggerd should work on PC
  @classmethod
  def setUpClass(cls):
    if not (EON or TICI):
      raise unittest.SkipTest

  def setUp(self):
    self._clear_logs()
    os.environ["LOGGERD_TEST"] = "1"
    os.environ["LOGGERD_SEGMENT_LENGTH"] = str(SEGMENT_LENGTH)

  def tearDown(self):
    self._clear_logs()

  def _clear_logs(self):
    if os.path.exists(ROOT):
      shutil.rmtree(ROOT)

  def _get_latest_segment_path(self):
    last_route = sorted(Path(ROOT).iterdir())[-1]
    return os.path.join(ROOT, last_route)

  # TODO: this should run faster than real time
  @parameterized.expand([(True, ), (False, )])
  def test_log_rotation(self, record_front):
    Params().put_bool("RecordFront", record_front)

    managed_processes['sensord'].start()
    managed_processes['loggerd'].start()

    time.sleep(1.0)
    managed_processes['camerad'].start()

    num_segments = int(os.getenv("SEGMENTS", random.randint(10, 15)))

    # wait for loggerd to make the dir for first segment
    route_prefix_path = None
    with Timeout(int(SEGMENT_LENGTH*3)):
      while route_prefix_path is None:
        try:
          route_prefix_path = self._get_latest_segment_path().rsplit("--", 1)[0]
        except Exception:
          time.sleep(0.1)

    def check_seg(i):
      # check each camera file size
      counts = []
      first_frames = []
      for camera, fps, size, encode_idx_name in CAMERAS:
        if not record_front and "dcamera" in camera:
          continue

        eon_dcam = EON and (camera == 'dcamera.hevc')

        file_path = f"{route_prefix_path}--{i}/{camera}"

        # check file exists
        self.assertTrue(os.path.exists(file_path), f"segment #{i}: '{file_path}' missing")

        # TODO: this ffprobe call is really slow
        # check frame count
        cmd = f"ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 {file_path}"
        if TICI:
          cmd = "LD_LIBRARY_PATH=/usr/local/lib " + cmd

        expected_frames = fps * SEGMENT_LENGTH
        frame_tolerance = 1 if eon_dcam else 0
        probe = subprocess.check_output(cmd, shell=True, encoding='utf8')
        frame_count = int(probe.split('\n')[0].strip())
        counts.append(frame_count)

        self.assertTrue(abs(expected_frames - frame_count) <= frame_tolerance,
                        f"segment #{i}: {camera} failed frame count check: expected {expected_frames}, got {frame_count}")

        # sanity check file size
        file_size = os.path.getsize(file_path)
        self.assertTrue(math.isclose(file_size, size, rel_tol=FILE_SIZE_TOLERANCE))

        # Check encodeIdx
        if encode_idx_name is not None:
          rlog_path = f"{route_prefix_path}--{i}/rlog.bz2"
          msgs = [m for m in LogReader(rlog_path) if m.which() == encode_idx_name]
          encode_msgs = [getattr(m, encode_idx_name) for m in msgs]

          valid = [m.valid for m in msgs]
          segment_idxs = [m.segmentId for m in encode_msgs]
          encode_idxs = [m.encodeId for m in encode_msgs]
          frame_idxs = [m.frameId for m in encode_msgs]

          # Check frame count
          self.assertEqual(frame_count, len(segment_idxs))
          self.assertEqual(frame_count, len(encode_idxs))

          # Check for duplicates or skips
          self.assertEqual(0, segment_idxs[0])
          self.assertEqual(len(set(segment_idxs)), len(segment_idxs))

          self.assertTrue(all(valid))

          if not eon_dcam:
            self.assertEqual(expected_frames * i, encode_idxs[0])
            first_frames.append(frame_idxs[0])
          self.assertEqual(len(set(encode_idxs)), len(encode_idxs))

      self.assertEqual(1, len(set(first_frames)))

      if TICI:
        expected_frames = fps * SEGMENT_LENGTH
        self.assertEqual(min(counts), expected_frames)
      shutil.rmtree(f"{route_prefix_path}--{i}")

    try:
      for i in trange(num_segments + 1):
        # poll for next segment
        with Timeout(int(SEGMENT_LENGTH*10), error_msg=f"timed out waiting for segment {i}"):
          while Path(f"{route_prefix_path}--{i}") not in Path(ROOT).iterdir():
            time.sleep(0.1)
    finally:
      managed_processes['loggerd'].stop()
      managed_processes['camerad'].stop()
      managed_processes['sensord'].stop()

    for i in trange(num_segments):
      check_seg(i)


if __name__ == "__main__":
  unittest.main()
