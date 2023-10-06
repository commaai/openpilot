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
  def _msg(which: str, data: None | dict = None):
    msg = messaging.new_message(which)
    if data is not None:
      getattr(msg, which).from_dict(data)
    return msg.as_reader()

  @staticmethod
  def _get_failed(ref_logs: list, new_logs: list) -> bool:
    diff = compare_logs(ref_logs, new_logs, ignore_fields=IGNORE_FIELDS)
    _, _, failed = format_diff({"": {"": diff}}, {"": {"": {"ref": "", "new": ""}}}, {})
    return failed

  def _assertException(self, ref_logs: list, new_logs: list, msg: str):
    self.assertTrue(len(msg))
    with self.assertRaises(Exception) as cm:
      self._get_failed(ref_logs, new_logs)
    self.assertIn(msg, str(cm.exception))

  def test_no_diff(self):
    self.assertFalse(self._get_failed(self.ref_logs, self.ref_logs))

  def test_addition(self):
    # Adding any single message should fail
    for which in ('controlsState', 'carState', 'carControl'):
      with self.subTest(which=which):
        new_logs = self.ref_logs + [self._msg(which)]
        self._assertException(self.ref_logs, new_logs, "logs are not same length")

  def test_removal(self):
    # Removing any message should fail
    for idx in range(len(self.ref_logs)):
      with self.subTest(remove_idx=idx):
        new_logs = self.ref_logs[0:idx] + self.ref_logs[idx + 1:len(self.ref_logs)]
        self._assertException(self.ref_logs, new_logs, "logs are not same length")

  def test_alignment(self):
    # Msgs within each service type will match, but overall alignment should fail
    new_logs = self.ref_logs[::-1]
    self._assertException(self.ref_logs, new_logs, "msgs not aligned between logs")

    # Try with different length
    self._assertException(self.ref_logs, new_logs + [self._msg('controlsState')], "logs are not same length")

    # Try with replaced msg
    new_logs[0] = self._msg('controlsState')
    self._assertException(self.ref_logs, new_logs, "msgs not aligned between logs")

  def test_alignment_with_data(self):
    # Now order of msgs within controlsState shouldn't match
    self.ref_logs[0] = self._msg('controlsState', {'vCruise': 255})
    new_logs = self.ref_logs[::-1]
    self._assertException(self.ref_logs, new_logs, "msgs not aligned between logs")


if __name__ == "__main__":
  unittest.main()
