#!/usr/bin/env python3
import os
import sys
import unittest

from parameterized import parameterized
from typing import Optional, Union, List


from openpilot.selfdrive.test.openpilotci import get_url, upload_file
from openpilot.selfdrive.test.process_replay.compare_logs import compare_logs, format_process_diff
from openpilot.selfdrive.test.process_replay.process_replay import CONFIGS, PROC_REPLAY_DIR, FAKEDATA, replay_process
from openpilot.system.version import get_commit
from openpilot.tools.lib.filereader import FileReader
from openpilot.tools.lib.helpers import save_log
from openpilot.tools.lib.logreader import LogReader, LogIterable


BASE_URL = "https://commadataci.blob.core.windows.net/openpilotci/"
REF_COMMIT_FN = os.path.join(PROC_REPLAY_DIR, "ref_commit")
EXCLUDED_PROCS = {"modeld", "dmonitoringmodeld"}


def get_log_data(segment):
  r, n = segment.rsplit("--", 1)
  with FileReader(get_url(r, n)) as f:
    return f.read()


ALL_PROCS = sorted({cfg.proc_name for cfg in CONFIGS if cfg.proc_name not in EXCLUDED_PROCS})
PROC_TO_CFG = {cfg.proc_name: cfg for cfg in CONFIGS}

cpu_count = os.cpu_count() or 1


class TestProcessReplayBase(unittest.TestCase):
  """
  Base class that replays all processes within test_proceses from a segment,
  and puts the log messages in self.log_msgs for analysis by other tests.
  """
  segment: Optional[Union[str, LogIterable]] = None
  tested_procs: List[str] = ALL_PROCS

  @classmethod
  def setUpClass(cls, create_logs=True):
    if "Base" in cls.__name__:
      raise unittest.SkipTest("skipping base class")

    if isinstance(cls.segment, str):
      cls.log_reader = LogReader.from_bytes(get_log_data(cls.segment))
    else:
      cls.log_reader = cls.segment

    if create_logs:
      cls._create_log_msgs()

  @classmethod
  def _run_replay(cls, cfg):
    try:
      return replay_process(cfg, cls.log_reader, disable_progress=True)
    except Exception as e:
      raise Exception(f"failed on segment: {cls.segment} \n{e}") from e

  @classmethod
  def _create_log_msgs(cls):
    cls.log_msgs = {}
    cls.proc_cfg = {}

    for proc in cls.tested_procs:
      cfg = PROC_TO_CFG[proc]

      log_msgs = cls._run_replay(cfg)

      cls.log_msgs[proc] = log_msgs
      cls.proc_cfg[proc] = cfg


class TestProcessReplayDiffBase(TestProcessReplayBase):
  """
  Base class for checking for diff between process outputs.
  """
  update_refs = False
  upload_only = False
  long_diff = False
  ignore_msgs: List[str] = []
  ignore_fields: List[str] = []

  def setUp(self):
    super().setUp()
    if self.upload_only:
      raise unittest.SkipTest("skipping test, uploading only")

  @classmethod
  def setUpClass(cls):
    super().setUpClass(not cls.upload_only)

    if cls.long_diff:
      cls.maxDiff = None

    os.makedirs(os.path.dirname(FAKEDATA), exist_ok=True)

    cls.cur_commit = get_commit()
    cls.assertNotEqual(cls.cur_commit, None, "Couldn't get current commit")

    cls.upload = cls.update_refs or cls.upload_only

    try:
      with open(REF_COMMIT_FN) as f:
        cls.ref_commit = f.read().strip()
    except FileNotFoundError:
      print("Couldn't find reference commit")
      sys.exit(1)

    cls._create_ref_log_msgs()

  @classmethod
  def _create_ref_log_msgs(cls):
    cls.ref_log_msgs = {}

    for proc in cls.tested_procs:
      cur_log_fn = os.path.join(FAKEDATA, f"{cls.segment}_{proc}_{cls.cur_commit}.bz2")
      if cls.update_refs:  # reference logs will not exist if routes were just regenerated
        ref_log_path = get_url(*cls.segment.rsplit("--", 1))
      else:
        ref_log_fn = os.path.join(FAKEDATA, f"{cls.segment}_{proc}_{cls.ref_commit}.bz2")
        ref_log_path = ref_log_fn if os.path.exists(ref_log_fn) else BASE_URL + os.path.basename(ref_log_fn)

      if not cls.upload_only:
        save_log(cur_log_fn, cls.log_msgs[proc])
        cls.ref_log_msgs[proc] = list(LogReader(ref_log_path))

      if cls.upload:
        assert os.path.exists(cur_log_fn), f"Cannot find log to upload: {cur_log_fn}"
        upload_file(cur_log_fn, os.path.basename(cur_log_fn))
        os.remove(cur_log_fn)

  @parameterized.expand(ALL_PROCS)
  def test_process_diff(self, proc):
    if proc not in self.tested_procs:
      raise unittest.SkipTest(f"{proc} was not requested to be tested")

    cfg = self.proc_cfg[proc]
    log_msgs = self.log_msgs[proc]
    ref_log_msgs = self.ref_log_msgs[proc]

    diff = compare_logs(ref_log_msgs, log_msgs, self.ignore_fields + cfg.ignore, self.ignore_msgs)

    diff_short, diff_long = format_process_diff(diff)

    self.assertEqual(len(diff), 0, "\n" + diff_long if self.long_diff else diff_short)
