#!/usr/bin/env python3
import copy
import sys
import math
import capnp
import numbers
from cereal import log, messaging
import unittest
import dictdiffer
from collections import defaultdict
from typing import Dict
from openpilot.selfdrive.test.process_replay.compare_logs import compare_logs, format_diff

IGNORE_FIELDS = ["logMonoTime"]


class TestCompareLogs(unittest.TestCase):
  def setUp(self):
    self.ref_logs = [self._msg('controlsState'), self._msg('carState'), self._msg('carControl'),
                     self._msg('controlsState'), self._msg('carState'), self._msg('carControl')]

  @staticmethod
  def _msg(which: str, data: None | dict = None, size: None | int = None):
    msg = messaging.new_message(which, size=size)
    if data is not None:
      getattr(msg, which).from_dict(data)
    return msg.as_reader()

  @staticmethod
  def _get_failed(diff) -> bool:
    _, _, failed = format_diff({"": {"": diff}}, {"": {"": {"ref": "", "new": ""}}}, {})
    return failed

  def test_no_diff(self):
    diff = compare_logs(self.ref_logs, self.ref_logs, ignore_fields=IGNORE_FIELDS, ignore_msgs=[], tolerance=None)
    self.assertFalse(self._get_failed(diff))

  def test_addition(self):
    # Adding any single message should fail
    for which in ('controlsState', 'carState', 'carControl'):
      with self.subTest(which=which):
        new_logs = self.ref_logs + [self._msg(which)]
        diff = compare_logs(self.ref_logs, new_logs, ignore_fields=IGNORE_FIELDS, ignore_msgs=[], tolerance=None)
        print(diff)
        self.assertTrue(self._get_failed(diff))

  def test_removal(self):
    # Removing any message should fail
    for idx in range(len(self.ref_logs)):
      with self.subTest(remove_idx=idx):
        new_logs = self.ref_logs[0:idx] + self.ref_logs[idx + 1:len(self.ref_logs)]
        diff = compare_logs(self.ref_logs, new_logs, ignore_fields=IGNORE_FIELDS, ignore_msgs=[], tolerance=None)
        self.assertTrue(self._get_failed(diff))

  def test_alignment(self):
    # Reverse ref logs and compare: overall alignment should fail
    new_logs = self.ref_logs[::-1]

    diff = compare_logs(self.ref_logs, new_logs, ignore_fields=IGNORE_FIELDS, ignore_msgs=[], tolerance=None)
    print(diff)
    self.assertTrue(self._get_failed(diff))


if __name__ == "__main__":
  unittest.main()
