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

from openpilot.tools.lib.logreader import LogReader

IGNORE_FIELDS = ["logMonoTime"]


class TestCompareLogs(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.ref_logs = [
      # cls._msg('sendcan', [(0, 0, 0, 0)], size=1),
      cls._msg('controlsState', {'vCruise': 255}), cls._msg('carState', {'vEgo': 40.}), cls._msg('carControl', {'enabled': False}),
      cls._msg('controlsState', {'vCruise': 255}), cls._msg('carState', {'vEgo': 41.}), cls._msg('carControl', {'enabled': False}),
      cls._msg('controlsState', {'vCruise': 42.}), cls._msg('carState', {'vEgo': 42.}), cls._msg('carControl', {'enabled': True}),
      cls._msg('controlsState', {'vCruise': 43.}), cls._msg('carState', {'vEgo': 42.}), cls._msg('carControl', {'enabled': True}),
      cls._msg('controlsState', {'vCruise': 44.}), cls._msg('carState', {'vEgo': 43.}), cls._msg('carControl', {'enabled': True}),
      cls._msg('controlsState', {'vCruise': 45.}), cls._msg('carState', {'vEgo': 44.}), cls._msg('carControl', {'enabled': True}),
      cls._msg('controlsState', {'vCruise': 45.}), cls._msg('carState', {'vEgo': 45.}), cls._msg('carControl', {'enabled': True}),
      cls._msg('controlsState', {'vCruise': 45.}), cls._msg('carState', {'vEgo': 45.}), cls._msg('carControl', {'enabled': True}),
      cls._msg('controlsState', {'vCruise': 45.}), cls._msg('carState', {'vEgo': 43.}), cls._msg('carControl', {'enabled': False}),
      cls._msg('controlsState', {'vCruise': 45.}), cls._msg('carState', {'vEgo': 39.}), cls._msg('carControl', {'enabled': False}),
    ]
    # print(cls.ref_logs)
    # raise unittest.SkipTest

  def setUp(self):
    self.new_logs = copy.deepcopy(self.ref_logs)

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

  # def test_no_diff(self):
  #   diff = compare_logs(self.ref_logs, self.new_logs, ignore_fields=IGNORE_FIELDS, ignore_msgs=[], tolerance=None)
  #   print(diff)
  #   self.assertFalse(self._get_failed(diff))

  def test_no_diff(self):
    ref_logs = [self._msg('controlsState'), self._msg('carState'), self._msg('carControl'),
                self._msg('controlsState'), self._msg('carState'), self._msg('carControl')]
    diff = compare_logs(ref_logs, ref_logs, ignore_fields=IGNORE_FIELDS, ignore_msgs=[], tolerance=None)
    self.assertFalse(self._get_failed(diff))

  def test_addition(self):
    # Adding any single message should fail
    ref_logs = [self._msg('controlsState'), self._msg('carState'), self._msg('carControl'),
                self._msg('controlsState'), self._msg('carState'), self._msg('carControl')]
    for which in ('controlsState', 'carState', 'carControl'):
      with self.subTest(which=which):
        new_logs = ref_logs + [self._msg(which)]
        diff = compare_logs(ref_logs, new_logs, ignore_fields=IGNORE_FIELDS, ignore_msgs=[], tolerance=None)
        print(diff)
        self.assertTrue(self._get_failed(diff))

  def test_removal(self):
    # Removing any message should fail
    ref_logs = [self._msg('controlsState'), self._msg('carState'), self._msg('carControl'),
                self._msg('controlsState'), self._msg('carState'), self._msg('carControl')]
    for idx in range(len(ref_logs)):
      with self.subTest(remove_idx=idx):
        new_logs = ref_logs[0:idx] + ref_logs[idx + 1:len(ref_logs)]
        diff = compare_logs(ref_logs, new_logs, ignore_fields=IGNORE_FIELDS, ignore_msgs=[], tolerance=None)
        self.assertTrue(self._get_failed(diff))

  def test_alignment(self):
    # Reverse ref logs and compare: overall alignment should fail
    ref_logs = [self._msg('controlsState'), self._msg('carState'), self._msg('carControl'),
                self._msg('controlsState'), self._msg('carState'), self._msg('carControl')]

    new_logs = ref_logs[::-1]

    diff = compare_logs(ref_logs, new_logs, ignore_fields=IGNORE_FIELDS, ignore_msgs=[], tolerance=None)
    print(diff)
    self.assertTrue(self._get_failed(diff))


if __name__ == "__main__":
  unittest.main()
