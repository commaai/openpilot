#!/usr/bin/env python3
import hypothesis.strategies as st
from hypothesis import given, settings
import unittest

from cereal import messaging
from openpilot.selfdrive.test.process_replay.compare_logs import compare_logs, format_diff, remove_ignored_fields

IGNORE_FIELDS = ["logMonoTime"]


class TestCompareLogs(unittest.TestCase):
  @staticmethod
  def _msg(which: str, data: None | dict = None):
    msg = messaging.new_message(which)
    if data is not None:
      getattr(msg, which).from_dict(data)
    return msg.as_reader()

  @staticmethod
  def _get_failed(ref_logs: list, new_logs: list) -> bool:
    try:
      diff = compare_logs(ref_logs, new_logs, ignore_fields=IGNORE_FIELDS, ignore_msgs=[], tolerance=None)
      _, _, failed = format_diff({"": {"": diff}}, {"": {"": {"ref": "", "new": ""}}}, {})
    except Exception:
      return True
    return failed

  @settings(max_examples=1000, deadline=None)
  @given(data=st.data())
  def test_fuzzy_diff(self, data):
    """compare_logs tries to format the diff nicely. check it fails for additions, removals, alignment"""

    services = ['controlsState', 'carState', 'carControl']

    ref_logs = data.draw(st.lists(st.sampled_from(services).map(lambda x: self._msg(x)), max_size=6))
    new_logs = data.draw(st.lists(st.sampled_from(services).map(lambda x: self._msg(x)), max_size=6))

    any_diff = [remove_ignored_fields(m, IGNORE_FIELDS).as_reader().to_dict(verbose=True) for m in ref_logs] != \
               [remove_ignored_fields(m, IGNORE_FIELDS).as_reader().to_dict(verbose=True) for m in new_logs]

    failed = self._get_failed(ref_logs, new_logs)
    self.assertEqual(failed, any_diff, "compare_logs didn't catch diff")


if __name__ == "__main__":
  unittest.main()
