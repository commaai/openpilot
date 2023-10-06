#!/usr/bin/env python3
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

from openpilot.tools.lib.logreader import LogReader

IGNORE_FIELDS = ["logMonoTime"]


class TestCompareLogs(unittest.TestCase):
  def setUp(self):
    pass

  @staticmethod
  def _msg(which: str, data: None | dict = None):
    msg = messaging.new_message(which)
    if data is not None:
      getattr(msg, which).from_dict(data)
    return msg.as_reader()

  @staticmethod
  def _get_failed(diff) -> bool:
    _, _, failed = format_diff({"": {"": diff}}, {"": {"": {"ref": "", "new": ""}}}, {})
    return failed

  def test_no_diff(self):
    print('hi')

    # Test no diff
    log1 = [self._msg('controlsState', {'vCruise': 255}), self._msg('carControl', {'enabled': True})]

    diff = compare_logs(log1, log1, ignore_fields=IGNORE_FIELDS, ignore_msgs=[], tolerance=None)
    print(diff)
    self.assertFalse(self._get_failed(diff))

    # Test msgs out of order
    log1 = [self._msg('controlsState', {'vCruise': 255}), self._msg('carControl', {'enabled': True})]

    diff = compare_logs(log1, log1[::-1], ignore_fields=IGNORE_FIELDS, ignore_msgs=[], tolerance=None)
    print(diff)
    self.assertTrue(self._get_failed(diff))


if __name__ == "__main__":
  unittest.main()
