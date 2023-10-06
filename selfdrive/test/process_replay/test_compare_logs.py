#!/usr/bin/env python3
import unittest

from cereal import messaging
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
  def _get_failed(ref_logs: list, new_logs: list) -> bool:
    diff = compare_logs(ref_logs, new_logs, ignore_fields=IGNORE_FIELDS, ignore_msgs=[], tolerance=None)
    _, _, failed = format_diff({"": {"": diff}}, {"": {"": {"ref": "", "new": ""}}}, {})
    return failed

  def test_no_diff(self):
    self.assertFalse(self._get_failed(self.ref_logs, self.ref_logs))

  def test_addition(self):
    # Adding any single message should fail
    for which in ('controlsState', 'carState', 'carControl'):
      with self.subTest(which=which):
        new_logs = self.ref_logs + [self._msg(which)]
        self.assertTrue(self._get_failed(self.ref_logs, new_logs))

  def test_removal(self):
    # Removing any message should fail
    for idx in range(len(self.ref_logs)):
      with self.subTest(remove_idx=idx):
        new_logs = self.ref_logs[0:idx] + self.ref_logs[idx + 1:len(self.ref_logs)]
        self.assertTrue(self._get_failed(self.ref_logs, new_logs))

  def test_alignment(self):
    # Reverse ref logs and compare: overall alignment should fail
    new_logs = self.ref_logs[::-1]
    self.assertTrue(self._get_failed(self.ref_logs, new_logs))


if __name__ == "__main__":
  unittest.main()
